"""CPU integration run. All outputs are synthetic; never paper reproduction numbers."""
import argparse
import tempfile
from pathlib import Path
import torch
from stagefairrec.cli import main


def run(root, backbones):
    torch.set_num_threads(1)
    root = Path(root)
    def call(*args):
        main([str(x) for x in args])
    raw, data = root / 'raw', root / 'prepared'
    call('demo-data', '--out', raw)
    call('prepare', '--raw', raw, '--out', data)
    for mode in ('collaborative', 'sequential'):
        call('encode', '--data', data, '--out', root / mode, '--mode', mode,
             '--mock', '--config', 'configs/smoke.json')
    for backbone in backbones:
        semantic = root / ('sequential' if backbone in ('sasrec', 'bert4rec') else 'collaborative')
        base = root / backbone / 'base'
        common = ['--data', data, '--config', 'configs/smoke.json', '--set', f'backbone={backbone}']
        call('train', *common, '--set', 'variant=base', '--out', base)
        for variant in ('llm', 'source_only', 'post_only', 'wo_tlgc', 'full'):
            out = root / backbone / variant
            call('train', *common, '--set', f'variant={variant}', '--out', out,
                 '--backbone-checkpoint', base / 'best.pt', '--semantics', semantic)
            call('evaluate', '--data', data, '--checkpoint', out / 'best.pt',
                 '--semantics', semantic, '--out', out / 'test.json')
        full = root / backbone / 'full'
        call('export', '--data', data, '--checkpoint', full / 'best.pt',
             '--semantics', semantic, '--out', full / 'representations')
        call('attack', '--representations', full / 'representations', '--kinds', 'logistic',
             '--protocol', 'user_disjoint', '--out', full / 'attack.json')
        call('recommend', '--data', data, '--checkpoint', full / 'best.pt',
             '--semantics', semantic, '--row', '11', '--out', full / 'recommendations.json')
    print('SMOKE PASSED: ' + str(root))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out'); p.add_argument('--backbones', nargs='+',
                           default=['pmf', 'deepmodel', 'sasrec', 'bert4rec'])
    args = p.parse_args()
    if args.out:
        run(args.out, args.backbones)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            run(tmp, args.backbones)
