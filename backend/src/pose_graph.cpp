#include "msckf/pose_graph.hpp"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace msckf {

namespace {

double wrapAngle(double angle) { return std::atan2(std::sin(angle), std::cos(angle)); }

}  // namespace

double yawOf(const Mat3& R) { return std::atan2(R(1, 0), R(0, 0)); }

Mat3 rotationZ(double yaw) {
    Mat3 R;
    R << std::cos(yaw), -std::sin(yaw), 0.0,
         std::sin(yaw), std::cos(yaw), 0.0,
         0.0, 0.0, 1.0;
    return R;
}

int PoseGraph4Dof::addNode(const Pose& T_W_I) {
    Node node;
    node.p = T_W_I.t;
    node.yaw = yawOf(T_W_I.R);
    node.R_roll_pitch = rotationZ(node.yaw).transpose() * T_W_I.R;
    nodes_.push_back(node);
    return static_cast<int>(nodes_.size()) - 1;
}

void PoseGraph4Dof::addEdge(int i, int j, const Pose& T_i_j, double sigma_p, double sigma_yaw, bool robust) {
    // With R_i = Rz(yaw_i) R_rp_i: p_j - p_i = R_i t_ij, so in the yaw frame of i it is R_rp_i t_ij
    const Node& node_i = nodes_[i];
    const Mat3 R_i = rotationZ(node_i.yaw) * node_i.R_roll_pitch;
    Edge edge;
    edge.i = i;
    edge.j = j;
    edge.delta_p = node_i.R_roll_pitch * T_i_j.t;
    edge.delta_yaw = wrapAngle(yawOf(R_i * T_i_j.R) - node_i.yaw);
    edge.sigma_p = sigma_p;
    edge.sigma_yaw = sigma_yaw;
    edge.robust = robust;
    edges_.push_back(edge);
}

void PoseGraph4Dof::optimize(int iterations, double huber_threshold) {
    const int n = static_cast<int>(nodes_.size());
    if (n < 2) return;
    const int dim = 4 * (n - 1);  // node 0 is fixed
    auto column = [](int node) { return 4 * (node - 1); };
    const Vec3 e_z(0.0, 0.0, 1.0);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<Eigen::Triplet<double>> triplets;
        VecX b = VecX::Zero(dim);

        for (const Edge& edge : edges_) {
            const Node& a = nodes_[edge.i];
            const Node& c = nodes_[edge.j];
            const Mat3 Rz_i_T = rotationZ(a.yaw).transpose();

            Eigen::Matrix<double, 4, 1> r;
            r.head<3>() = Rz_i_T * (c.p - a.p) - edge.delta_p;
            r(3) = wrapAngle(c.yaw - a.yaw - edge.delta_yaw);

            // Jacobians w.r.t. [p, yaw] of i and j
            Eigen::Matrix<double, 4, 4> J_i = Eigen::Matrix<double, 4, 4>::Zero();
            Eigen::Matrix<double, 4, 4> J_j = Eigen::Matrix<double, 4, 4>::Zero();
            J_i.block<3, 3>(0, 0) = -Rz_i_T;
            J_i.block<3, 1>(0, 3) = -skew(e_z) * Rz_i_T * (c.p - a.p);  // d(Rz^T)/dyaw = -[e_z]x Rz^T
            J_i(3, 3) = -1.0;
            J_j.block<3, 3>(0, 0) = Rz_i_T;
            J_j(3, 3) = 1.0;

            Eigen::Matrix<double, 4, 1> information;
            information << Vec3::Constant(1.0 / (edge.sigma_p * edge.sigma_p)), 1.0 / (edge.sigma_yaw * edge.sigma_yaw);
            double weight = 1.0;
            if (edge.robust) {
                const double normalized = std::sqrt(r.dot(information.asDiagonal() * r));
                if (normalized > huber_threshold) weight = huber_threshold / normalized;
            }
            const Eigen::Matrix<double, 4, 4> W = weight * information.asDiagonal().toDenseMatrix();

            const int nodes[2] = {edge.i, edge.j};
            const Eigen::Matrix<double, 4, 4>* jacobians[2] = {&J_i, &J_j};
            for (int u = 0; u < 2; ++u) {
                if (nodes[u] == 0) continue;
                b.segment<4>(column(nodes[u])) -= jacobians[u]->transpose() * W * r;
                for (int v = 0; v < 2; ++v) {
                    if (nodes[v] == 0) continue;
                    const Eigen::Matrix<double, 4, 4> block = jacobians[u]->transpose() * W * (*jacobians[v]);
                    for (int row = 0; row < 4; ++row)
                        for (int col = 0; col < 4; ++col)
                            triplets.emplace_back(column(nodes[u]) + row, column(nodes[v]) + col, block(row, col));
                }
            }
        }

        Eigen::SparseMatrix<double> H(dim, dim);
        H.setFromTriplets(triplets.begin(), triplets.end());
        const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(H);
        if (solver.info() != Eigen::Success) return;
        const VecX delta = solver.solve(b);
        if (!delta.allFinite()) return;

        for (int k = 1; k < n; ++k) {
            nodes_[k].p += delta.segment<3>(column(k));
            nodes_[k].yaw = wrapAngle(nodes_[k].yaw + delta(column(k) + 3));
        }
        if (delta.norm() < 1e-8) break;
    }
}

}  // namespace msckf
