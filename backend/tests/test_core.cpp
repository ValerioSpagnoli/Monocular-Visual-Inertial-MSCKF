#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>

#include "msckf/msckf.hpp"

using namespace msckf;

namespace {

std::mt19937 rng(0);  // reseeded at the start of every test case, so results do not depend on the test order

Vec3 randomVector(double scale = 1.0) {
    std::normal_distribution<double> normal(0.0, scale);
    return {normal(rng), normal(rng), normal(rng)};
}

Mat3 randomRotation() { return expSO3(randomVector(1.5)); }

bool near(const MatX& a, const MatX& b, double tol) { return (a - b).cwiseAbs().maxCoeff() <= tol; }

const Mat3 K = (Mat3() << 400.0, 0.0, 320.0, 0.0, 400.0, 240.0, 0.0, 0.0, 1.0).finished();

Vec2 pixel(const Pose& T_W_C, const Vec3& W_f) {
    const Vec3 C_f = T_W_C.R.transpose() * (W_f - T_W_C.t);
    return (K * (C_f / C_f.z())).head<2>();
}

}  // namespace

TEST_CASE("SO(3) helpers") {
    rng.seed(42);
    const Vec3 a = randomVector(), b = randomVector();
    CHECK(near(skew(a) * b, a.cross(b), 1e-12));
    const Vec3 phi = randomVector(0.7);
    CHECK(near(logSO3(expSO3(phi)), phi, 1e-12));
    const Mat3 R = expSO3(phi);
    CHECK(near(R * R.transpose(), Mat3::Identity(), 1e-12));
}

TEST_CASE("Pose composition and inverse") {
    rng.seed(42);
    const Pose T(randomRotation(), randomVector());
    const Pose I = T * T.inverse();
    CHECK(near(I.R, Mat3::Identity(), 1e-12));
    CHECK(near(I.t, Vec3::Zero(), 1e-12));
}

TEST_CASE("Static IMU does not move") {
    rng.seed(42);
    ImuState imu;
    imu.T_W_I = Pose(randomRotation(), Vec3::Zero());
    const Vec3 f = imu.T_W_I.R.transpose() * (-imu.gravity);
    for (int k = 0; k < 200; ++k) imu.integrate(f, Vec3::Zero(), 0.005);
    CHECK(near(imu.v_W_I, Vec3::Zero(), 1e-9));
    CHECK(near(imu.T_W_I.t, Vec3::Zero(), 1e-9));
}

TEST_CASE("Initialization recovers the tilt, also upside down") {
    rng.seed(42);
    for (const Mat3& R_W_I : {Mat3(Mat3::Identity()), Mat3((Eigen::Vector3d(1.0, -1.0, -1.0)).asDiagonal()), randomRotation()}) {
        ImuState imu;
        const Vec3 f = R_W_I.transpose() * (-imu.gravity);
        imu.initialize(std::vector<ImuMeasurement>(10, ImuMeasurement{0.0, Vec3::Zero(), f}), false);
        const Mat3& R = imu.T_W_I.R;
        CHECK(R.determinant() == Catch::Approx(1.0).margin(1e-9));
        CHECK(near(R.transpose() * imu.gravity, R_W_I.transpose() * imu.gravity, 1e-9));
    }
}

TEST_CASE("Observability-constrained transition matches the error dynamics") {
    rng.seed(42);
    const double dt = 0.005;
    const Vec3 w(0.3, -0.2, 0.5), f(0.4, 0.1, 9.6), g(0.0, 0.0, -9.81);
    const Pose T0(randomRotation(), randomVector());
    const Vec3 v0 = randomVector();

    auto propagate = [&](const Pose& T, const Vec3& v) {
        ImuState imu;
        imu.T_W_I = T;
        imu.v_W_I = v;
        imu.gravity = g;
        imu.integrate(f, w, dt);
        return std::make_pair(imu.T_W_I, imu.v_W_I);
    };
    const auto [T1, v1] = propagate(T0, v0);

    MatX F = MatX::Zero(15, 15);
    F.block<3, 3>(0, 0) = -skew(w);
    F.block<3, 3>(0, 3) = -Mat3::Identity();
    F.block<3, 3>(6, 0) = -T0.R * skew(f);
    F.block<3, 3>(6, 9) = -T0.R;
    F.block<3, 3>(12, 6) = Mat3::Identity();
    const MatX Phi = observabilityConstrainedTransition(MatX::Identity(15, 15) + F * dt, dt, T0, v0, T1, v1, g);

    // Numerical derivative of the propagated error w.r.t. the initial orientation error (R_true = R exp(δθ))
    const double eps = 1e-6;
    for (int k = 0; k < 3; ++k) {
        Vec3 d = Vec3::Zero();
        d(k) = eps;
        const auto [Tp, vp] = propagate(Pose(T0.R * expSO3(d), T0.t), v0);
        const auto [Tm, vm] = propagate(Pose(T0.R * expSO3(-d), T0.t), v0);
        const Vec3 dtheta = (logSO3(T1.R.transpose() * Tp.R) - logSO3(T1.R.transpose() * Tm.R)) / (2 * eps);
        const Vec3 dv = (vp - vm) / (2 * eps);
        const Vec3 dp = (Tp.t - Tm.t) / (2 * eps);
        CHECK(near(Phi.block<3, 1>(0, k), dtheta, 1e-6));
        // The OC modification changes the velocity and position blocks by design: only a loose agreement
        CHECK(near(Phi.block<3, 1>(6, k), dv, 1e-3));
        CHECK(near(Phi.block<3, 1>(12, k), dp, 1e-3));
    }

    // Defining property of the constrained blocks (Hesch et al. 2014): Phi_v u = w1, Phi_p u = w2
    const Vec3 u = T0.R.transpose() * g;
    CHECK(near(Phi.block<3, 3>(6, 0) * u, skew(v0 - v1) * g, 1e-9));
    CHECK(near(Phi.block<3, 3>(12, 0) * u, skew(dt * v0 + T0.t - T1.t) * g, 1e-9));
}

TEST_CASE("Triangulation recovers the point and rejects points behind the cameras") {
    rng.seed(42);
    const Pose T0(randomRotation(), randomVector());
    std::vector<Pose> poses;
    for (int k = 0; k < 5; ++k) poses.push_back(T0 * Pose(expSO3(randomVector(0.05)), randomVector(0.3)));
    const Vec3 W_f = T0.transform(Vec3(0.4, -0.3, 4.5));
    std::vector<Vec2> keypoints;
    for (const auto& T : poses) keypoints.push_back(pixel(T, W_f));

    const TriangulationResult result = triangulateFeature(poses, keypoints, K.inverse(), T0.t + 1.4 * (W_f - T0.t));
    REQUIRE(result.point.has_value());
    CHECK(near(*result.point, W_f, 1e-6));
    CHECK(result.rms < 1e-9);

    const Vec3 behind = T0.transform(Vec3(0.3, -0.2, -4.0));
    std::vector<Vec2> behind_keypoints;
    for (const auto& T : poses) behind_keypoints.push_back(pixel(T, behind));
    CHECK_FALSE(triangulateFeature(poses, behind_keypoints, K.inverse(), behind).point.has_value());
}

TEST_CASE("Sampson distance is zero for a correct match") {
    rng.seed(42);
    const Pose T1(randomRotation(), randomVector());
    const Pose T2 = T1 * Pose(expSO3(randomVector(0.1)), randomVector(0.3));
    const Vec3 W_f = T1.transform(Vec3(0.2, 0.1, 5.0));
    const Mat3 F = fundamentalMatrix(T1.inverse() * T2, K.inverse());
    CHECK(sampsonDistance(F, pixel(T1, W_f), pixel(T2, W_f)) < 1e-9);
    CHECK(sampsonDistance(F, pixel(T1, W_f), pixel(T2, W_f) + Vec2(0.0, 20.0)) > 1.0);
}

TEST_CASE("Homography transfer error for a pure rotation") {
    rng.seed(42);
    const Pose T1(randomRotation(), randomVector());
    const Pose T2(T1.R * expSO3(randomVector(0.1)), T1.t);  // same center
    const Pose T_C1_C2 = T1.inverse() * T2;
    const Mat3 H = K * T_C1_C2.R.transpose() * K.inverse();  // pixels of camera 1 -> camera 2
    const Vec3 W_f = T1.transform(Vec3(0.2, 0.1, 5.0));
    CHECK(homographyTransferError(H, H.inverse(), pixel(T1, W_f), pixel(T2, W_f)) < 1e-9);
    CHECK(homographyTransferError(H, H.inverse(), pixel(T1, W_f), pixel(T2, W_f) + Vec2(0.0, 20.0)) > 5.0);
}

TEST_CASE("Chi-square quantiles") {
    rng.seed(42);
    CHECK(chi2Quantile(1) == Catch::Approx(3.841458820694124).epsilon(1e-10));
    CHECK(chi2Quantile(2) == Catch::Approx(5.991464547107979).epsilon(1e-10));
    CHECK(chi2Quantile(10) == Catch::Approx(18.307038053275146).epsilon(1e-10));
    CHECK(chi2Quantile(60) == Catch::Approx(79.08194448784878).epsilon(1e-10));
}

TEST_CASE("Filter with a static IMU stays at rest") {
    rng.seed(42);
    Parameters parameters;
    parameters.K = K;
    Msckf filter(parameters);
    const Vec3 f = -parameters.gravity;
    for (int k = 0; k < 10; ++k) filter.addImu({0.005 * k, Vec3::Zero(), f});  // static samples for the initialization
    filter.addImage(FrameFeatures{});
    for (int k = 10; k <= 400; ++k) filter.addImu({0.005 * k, Vec3::Zero(), f});
    CHECK(filter.imu().initialized);
    CHECK(near(filter.imu().T_W_I.t, Vec3::Zero(), 1e-6));
    CHECK(near(filter.imu().v_W_I, Vec3::Zero(), 1e-6));
}

TEST_CASE("Static initialization waits for still features") {
    rng.seed(7);
    Parameters parameters;
    parameters.K = K;
    parameters.static_window = 0.5;
    Msckf filter(parameters);
    const Vec3 f = -parameters.gravity;

    // Random descriptors of 50 points, shifted by 5 px per image while moving, still afterwards
    const int n = 50;
    std::vector<Vec2> base;
    Eigen::MatrixXf descriptors = Eigen::MatrixXf::Random(n, 64);
    for (int i = 0; i < n; ++i) base.push_back(Vec2(320.0, 240.0) + 100.0 * randomVector().head<2>());
    auto frame = [&](double shift) {
        FrameFeatures features;
        for (const Vec2& u : base) features.keypoints.push_back(u + Vec2(shift, 0.0));
        features.descriptors = descriptors;
        features.scores.assign(n, 1.0f);
        return features;
    };

    double t = 0.0, shift = 0.0;
    auto image = [&](bool moving) {  // 20 Hz images, 200 Hz IMU (tilted specific force while moving)
        for (int k = 0; k < 10; ++k, t += 0.005) filter.addImu({t, Vec3::Zero(), moving ? Vec3(1.0, 0.0, 9.81) : f});
        if (moving) shift += 5.0;
        filter.addImage(frame(shift));
    };
    for (int k = 0; k < 20; ++k) image(true);
    CHECK_FALSE(filter.imu().initialized);
    for (int k = 0; k < 8; ++k) image(false);
    CHECK_FALSE(filter.imu().initialized);  // still for 0.35 s
    for (int k = 0; k < 5; ++k) image(false);
    REQUIRE(filter.imu().initialized);      // still for 0.6 s
    // Gravity from the static samples only (the moving ones would tilt it by ~6 deg)
    CHECK(near(filter.imu().T_W_I.R.transpose() * Vec3::UnitZ(), Vec3::UnitZ(), 1e-6));
}
