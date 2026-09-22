"""Run matched RQ1/RQ3 comparisons across three seeds; caches must exist per seed."""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--data', required=True); p.add_argument('--cache-root', required=True)
p.add_argument('--out', required=True); p.add_argument('--config', default='configs/default.json')
p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
p.add_argument('--backbones', nargs='+', default=['pmf', 'deepmodel', 'sasrec', 'bert4rec'])
args = p.parse_args()
def call(*cmd):
    subprocess.run([sys.executable, '-m', 'stagefairrec', *map(str, cmd)], check=True)
for seed in args.seeds:
    for backbone in args.backbones:
        base = Path(args.out) / str(seed) / backbone / 'base'
        mode = 'sequential' if backbone in ('sasrec', 'bert4rec') else 'collaborative'
        cache = Path(args.cache_root) / str(seed) / mode
        common = ['--data', args.data, '--config', args.config, '--set', f'seed={seed}',
                  '--set', f'backbone={backbone}']
        call('train', *common, '--set', 'variant=base', '--out', base)
        for variant in ('base', 'llm', 'source_only', 'post_only', 'wo_tlgc', 'full'):
            run = base.parent / variant
            sem = [] if variant == 'base' else ['--semantics', cache]
            if variant != 'base':
                call('train', *common, '--set', f'variant={variant}', '--out', run,
                     '--backbone-checkpoint', base / 'best.pt', *sem)
            call('evaluate', '--data', args.data, '--checkpoint', run / 'best.pt',
                 *sem, '--out', run / 'test.json')
            call('export', '--data', args.data, '--checkpoint', run / 'best.pt',
                 *sem, '--out', run / 'representations')
            call('attack', '--representations', run / 'representations', '--seed', seed,
                 '--out', run / 'attack.json')
