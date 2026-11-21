"""Statistics of the IMU data produced by PhotorealisticGenerator on a static trajectory."""
import numpy as np
import pandas as pd

from dataset.tools.dataset_generators.photorealistic_generator import PhotorealisticGenerator

N_POSES = 400
SIGMA_ACC, SIGMA_GYRO = 0.01, 0.001          # white noise densities
WALK_ACC, WALK_GYRO = 0.001, 0.0001          # bias random-walk densities


def generate_static_sequence(tmp_path, monkeypatch, walk_acc: float = WALK_ACC) -> tuple[pd.DataFrame, pd.DataFrame]:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / 'data' / 'tartanair'
    (root / 'S000' / 'cam').mkdir(parents=True)
    pd.DataFrame([dict(R00=0, R01=0, R02=1, R10=-1, R11=0, R12=0, R20=0, R21=-1, R22=0, fx=320, fy=320, px=320, py=240, w=640, h=480)]) \
        .to_csv(root / 'camera_info.csv', index=False)
    for k in range(N_POSES): (root / 'S000' / 'cam' / f'{k:06d}.png').touch()
    with open(root / 'S000' / 'trajectory.csv', 'w') as f:
        f.write('x y z qx qy qz qw\n' + '1.0 2.0 3.0 0.0 0.0 0.0 1.0\n' * N_POSES)

    np.random.seed(0)
    PhotorealisticGenerator('tartanair', 'S000', accelerometer_noise_density=SIGMA_ACC, gyroscope_noise_density=SIGMA_GYRO,
                            accelerometer_random_walk=walk_acc, gyroscope_random_walk=WALK_GYRO).process_data()
    return pd.read_csv(root / 'S000' / 'imu.csv'), pd.read_csv(root / 'S000' / 'imu_gt.csv')


def test_white_noise_is_discretized(tmp_path, monkeypatch):
    noisy, true = generate_static_sequence(tmp_path, monkeypatch)
    dt = np.median(np.diff(noisy['timestamp']))
    error = (noisy - true).iloc[1:]
    # High-pass (first differences) removes the slowly varying bias: var(diff) = 2 sigma^2 / dt
    for columns, sigma in ((['ax', 'ay', 'az'], SIGMA_ACC), (['wx', 'wy', 'wz'], SIGMA_GYRO)):
        measured = np.diff(error[columns].values, axis=0).std() / np.sqrt(2)
        assert abs(measured / (sigma / np.sqrt(dt)) - 1) < 0.1


def test_bias_is_accumulated(tmp_path, monkeypatch):
    walk = 0.05 # large enough for the bias drift to stand out of the white noise in 20 s
    noisy, true = generate_static_sequence(tmp_path, monkeypatch, walk_acc=walk)
    error = (noisy - true).iloc[1:]
    # With the bias accumulated, the low-frequency part of the error drifts: the mean over the last samples
    # differs from the mean over the first ones much more than the averaged white noise could explain
    dt = np.median(np.diff(noisy['timestamp']))
    window = 500
    drift = np.abs(error[['ax', 'ay', 'az']].values[-window:].mean(0) - error[['ax', 'ay', 'az']].values[:window].mean(0))
    white_noise_of_mean = SIGMA_ACC / np.sqrt(dt) / np.sqrt(window)
    assert drift.max() > 3 * white_noise_of_mean
