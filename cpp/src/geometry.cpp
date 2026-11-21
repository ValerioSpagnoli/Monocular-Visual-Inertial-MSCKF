#include "msckf/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace msckf {

Vec3 intersectionOfLines(const std::vector<Line>& lines) {
    Mat3 X = Mat3::Zero();
    Vec3 y = Vec3::Zero();
    for (const auto& line : lines) {
        const Vec3 d = line.direction.normalized();
        const Mat3 P = Mat3::Identity() - d * d.transpose();
        X += line.confidence * P;
        y += line.confidence * P * line.base;
    }
    return X.completeOrthogonalDecomposition().pseudoInverse() * y;
}

double angleBetweenDirections(const Vec3& d1, const Vec3& d2) {
    const double dot = std::clamp(d1.normalized().dot(d2.normalized()), -1.0, 1.0);
    return std::acos(dot);
}

Mat3 fundamentalMatrix(const Pose& T_C1_C2, const Mat3& invK) {
    return invK.transpose() * skew(T_C1_C2.t) * T_C1_C2.R * invK;
}

double sampsonDistance(const Mat3& F, const Vec2& keypoint_1, const Vec2& keypoint_2) {
    const Vec3 x1(keypoint_1.x(), keypoint_1.y(), 1.0);
    const Vec3 x2(keypoint_2.x(), keypoint_2.y(), 1.0);
    const Vec3 F_x2 = F * x2;
    const Vec3 Ft_x1 = F.transpose() * x1;
    const double denominator = F_x2(0) * F_x2(0) + F_x2(1) * F_x2(1) + Ft_x1(0) * Ft_x1(0) + Ft_x1(1) * Ft_x1(1);
    return denominator > 0.0 ? std::abs(x1.dot(F_x2)) / std::sqrt(denominator) : 0.0;
}

TriangulationResult triangulateFeature(const std::vector<Pose>& poses, const std::vector<Vec2>& keypoints, const Mat3& invK,
                                       const Vec3& initial_point, int max_iterations) {
    const std::size_t n = poses.size();
    const Pose& anchor = poses.front();

    std::vector<Vec2> observations(n);
    std::vector<Mat3> R(n);
    std::vector<Vec3> t(n);
    for (std::size_t i = 0; i < n; ++i) {
        observations[i] = (invK * Vec3(keypoints[i].x(), keypoints[i].y(), 1.0)).head<2>();
        const Pose relative = poses[i].inverse() * anchor;  // anchor camera in camera i: p_Ci = R p_C0 + t
        R[i] = relative.R;
        t[i] = relative.t;
    }

    const Vec3 p0 = anchor.R.transpose() * (initial_point - anchor.t);
    Vec3 x = p0.z() > 1e-3 ? Vec3(p0.x() / p0.z(), p0.y() / p0.z(), 1.0 / p0.z())
                           : Vec3(observations[0].x(), observations[0].y(), 0.1);

    VecX residuals(2 * n);
    Eigen::Matrix<double, Eigen::Dynamic, 3> J(2 * n, 3);
    VecX depths(n);
    auto evaluate = [&](const Vec3& x_, VecX& r_, Eigen::Matrix<double, Eigen::Dynamic, 3>& J_, VecX& depths_) {
        for (std::size_t i = 0; i < n; ++i) {
            const Vec3 h = R[i].col(0) * x_(0) + R[i].col(1) * x_(1) + R[i].col(2) + t[i] * x_(2);
            depths_(i) = h.z();
            r_.segment<2>(2 * i) = observations[i] - h.head<2>() / h.z();
            Eigen::Matrix<double, 2, 3> dz_dh;
            dz_dh << 1.0 / h.z(), 0.0, -h.x() / (h.z() * h.z()),
                     0.0, 1.0 / h.z(), -h.y() / (h.z() * h.z());
            Mat3 dh_dx;
            dh_dx << R[i].col(0), R[i].col(1), t[i];
            J_.block<2, 3>(2 * i, 0) = dz_dh * dh_dx;
        }
    };

    double damping = 1e-3;
    evaluate(x, residuals, J, depths);
    double cost = residuals.squaredNorm();
    VecX new_residuals(2 * n);
    Eigen::Matrix<double, Eigen::Dynamic, 3> new_J(2 * n, 3);
    VecX new_depths(n);
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        const Mat3 JtJ = J.transpose() * J;
        Mat3 A = JtJ;
        A.diagonal() += damping * (JtJ.diagonal().array() + 1e-12).matrix();
        const Vec3 step = A.ldlt().solve(J.transpose() * residuals);
        const Vec3 candidate = x + step;
        evaluate(candidate, new_residuals, new_J, new_depths);
        const double new_cost = new_residuals.squaredNorm();
        if (new_cost < cost) {
            x = candidate;
            residuals = new_residuals;
            J = new_J;
            depths = new_depths;
            cost = new_cost;
            damping = std::max(damping / 10.0, 1e-9);
            if (step.norm() < 1e-8) break;
        } else {
            damping *= 10.0;
        }
    }

    TriangulationResult result;
    result.rms = std::sqrt(cost / static_cast<double>(n));
    if (x(2) <= 0.0 || (depths.array() <= 0.0).any()) return result;  // behind a camera (or at infinity)
    result.point = anchor.R * (Vec3(x(0), x(1), 1.0) / x(2)) + anchor.t;
    return result;
}

MatX observabilityConstrainedTransition(const MatX& Phi_in, double dt, const Pose& T_W_I_null, const Vec3& v_W_I_null,
                                        const Pose& T_W_I, const Vec3& v_W_I, const Vec3& gravity) {
    MatX Phi = Phi_in;
    Phi.block<3, 3>(0, 0) = T_W_I.R.transpose() * T_W_I_null.R;

    const Vec3 u = T_W_I_null.R.transpose() * gravity;
    const Vec3 s = u / u.dot(u);

    const Mat3 A_vel = Phi.block<3, 3>(6, 0);
    const Mat3 A_pos = Phi.block<3, 3>(12, 0);
    const Vec3 w1 = skew(v_W_I_null - v_W_I) * gravity;
    const Vec3 w2 = skew(dt * v_W_I_null + T_W_I_null.t - T_W_I.t) * gravity;

    Phi.block<3, 3>(6, 0) = A_vel - (A_vel * u - w1) * s.transpose();
    Phi.block<3, 3>(12, 0) = A_pos - (A_pos * u - w2) * s.transpose();
    return Phi;
}

namespace {

/// Regularized lower incomplete gamma function P(a, x) (series and continued fraction, Numerical Recipes).
double regularizedGammaP(double a, double x) {
    if (x <= 0.0) return 0.0;
    const double log_prefactor = -x + a * std::log(x) - std::lgamma(a);
    if (x < a + 1.0) {
        double term = 1.0 / a, sum = term, ap = a;
        for (int n = 0; n < 500; ++n) {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if (std::abs(term) < std::abs(sum) * 1e-15) break;
        }
        return sum * std::exp(log_prefactor);
    }
    const double tiny = std::numeric_limits<double>::min() / std::numeric_limits<double>::epsilon();
    double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int i = 1; i < 500; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < tiny) d = tiny;
        c = b + an / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::abs(delta - 1.0) < 1e-15) break;
    }
    return 1.0 - std::exp(log_prefactor) * h;
}

}  // namespace

double chi2Quantile(int dof, double probability) {
    // CDF(x) = P(dof / 2, x / 2), inverted by bisection
    const double a = 0.5 * dof;
    double low = 0.0, high = std::max(1.0, static_cast<double>(dof));
    while (regularizedGammaP(a, 0.5 * high) < probability) high *= 2.0;
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (low + high);
        (regularizedGammaP(a, 0.5 * mid) < probability ? low : high) = mid;
        if (high - low < 1e-12 * high) break;
    }
    return 0.5 * (low + high);
}

}  // namespace msckf
