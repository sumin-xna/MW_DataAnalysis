from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'code'

TASKS = [
    ('fast solver', CODE / 'global_game_hedge_run_fast.py'),
    ('exact solver', CODE / 'global_game_hedge_run_exact.py'),
    ('fully nested solver', CODE / 'global_game_hedge_run_fully_nested.py'),
]


def run_task(label, script_path):
    print(f'Running {label}: {script_path.name}')
    if not script_path.exists():
        print(f'  skipped: file not found')
        return
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f'{label} failed with exit code {result.returncode}')


if __name__ == '__main__':
    for label, script in TASKS:
        run_task(label, script)
    print('All available model scripts completed.')
