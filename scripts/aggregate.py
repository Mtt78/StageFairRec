"""Aggregate actual run files; never contains hard-coded manuscript results."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser(); p.add_argument('--runs', required=True); p.add_argument('--out', required=True)
a = p.parse_args(); grouped = defaultdict(list)
for path in Path(a.runs).rglob('test.json'):
    r = json.loads(path.read_text())
    if 'config' not in r:
        continue
    values = {k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    attack = path.parent / 'attack.json'
    if attack.exists():
        for kind, result in json.loads(attack.read_text())['attacks'].items():
            values['AUC_' + kind] = result['test_auc']
    key = (r['data_hash'], r['config']['backbone'], r['config']['variant'], r['mock_semantics'])
    grouped[key].append((r['config']['seed'], values))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['data_hash', 'backbone', 'variant', 'mock', 'metric', 'seeds', 'mean', 'std'])
    for key, entries in sorted(grouped.items()):
        seeds = [x[0] for x in entries]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f'Duplicate seeds under {key}; separate parameter/encoder experiments')
        for metric in sorted(set.intersection(*(set(e[1]) for e in entries))):
            vals = [e[1][metric] for e in entries]
            w.writerow([*key, metric, len(vals), np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.])
