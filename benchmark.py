"""
Run main.py on a set of sequences without the viewer and print a table of accuracy, consistency and speed.

    pixi run python benchmark.py                      # default set
    pixi run python benchmark.py --only V1_01_easy    # subset (substring match on the sequence name)
    pixi run python benchmark.py -- --sigma_image 1.5 # extra arguments forwarded to main.py
"""
import argparse
import re
import subprocess
import sys
import time

SEQUENCES = [
    ('euroc', 'V1_01_easy', []),
    ('euroc', 'V1_02_medium', []),
    ('euroc', 'V1_03_difficult', []),
    ('tartanair', 'P001', ['--noise_level', 'mid']),
]

PATTERNS = {
    'distance': r'ATE \(evaluated over \d+ steps, ([\d.]+) m\)',
    'ate': r'Position RMSE:\s+([\d.]+) m',
    'aoe': r'Orient\. RMSE:\s+([\d.]+) deg',
    'inside': r'Inside 3σ:\s+([\d.]+) %',
    'nees_p': r'Mean NEES:\s+position ([\d.]+)',
    'nees_r': r'orientation ([\d.]+) \(consistent',
    'hz_camera': r'With Camera:\s+[\d.]+ s - ([\d.]+) Hz',
    'hz_imu': r'Without Camera:\s+[\d.]+ s - ([\d.]+) Hz',
}


def run(source: str, sequence: str, extra: list) -> dict:
    command = [sys.executable, 'main.py', '--source', source, '--sequence', sequence, '--max_frames', '100000', '--no_viewer'] + extra
    start = time.time()
    output = subprocess.run(command, capture_output=True, text=True).stdout
    result = {key: float(m.group(1)) if (m := re.search(pattern, output)) else float('nan') for key, pattern in PATTERNS.items()}
    result['wall'] = time.time() - start
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', type=str, default=None, help='Run only sequences whose name contains this string')
    args, extra = parser.parse_known_args()
    extra = [a for a in extra if a != '--']

    print(f'| Sequence | Distance | ATE RMSE | Orient. RMSE | Inside 3σ | NEES pos / orient | Camera loop | IMU loop | Wall time |')
    print(f'|---|---|---|---|---|---|---|---|---|')
    for source, sequence, sequence_args in SEQUENCES:
        if args.only and args.only not in sequence: continue
        r = run(source, sequence, sequence_args + extra)
        print(f"| {sequence} | {r['distance']:.1f} m | {r['ate']:.3f} m | {r['aoe']:.2f}° | {r['inside']:.1f} % | "
              f"{r['nees_p']:.1f} / {r['nees_r']:.1f} | {r['hz_camera']:.1f} Hz | {r['hz_imu']:.0f} Hz | {r['wall']:.0f} s |", flush=True)
