"""Validation-only fairness coefficient sweep; no test selection."""
import argparse
import subprocess
import sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--data', required=True); p.add_argument('--semantics', required=True)
p.add_argument('--backbone-checkpoint', required=True); p.add_argument('--out', required=True)
p.add_argument('--config', default='configs/default.json')
p.add_argument('--parameter', choices=['lambda1', 'lambda2', 'source_lambda', 'beta'], required=True)
p.add_argument('--values', nargs='+', type=float, required=True)
p.add_argument('--set', action='append', default=[])
a = p.parse_args()
if a.parameter == 'source_lambda':
    p.error('source_lambda requires retraining source semantics separately, not a calibration sweep')
for value in a.values:
    cmd = [sys.executable, '-m', 'stagefairrec', 'train', '--data', a.data,
           '--semantics', a.semantics, '--backbone-checkpoint', a.backbone_checkpoint,
           '--config', a.config, '--out', str(Path(a.out) / f'{a.parameter}_{value:g}'),
           '--set', f'{a.parameter}={value}']
    for entry in a.set:
        cmd += ['--set', entry]
    subprocess.run(cmd, check=True)
