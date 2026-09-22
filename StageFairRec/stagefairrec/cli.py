import argparse
import json
from pathlib import Path
from .utils import read_json, write_json


def config(args):
    cfg = read_json(args.config)
    for entry in getattr(args, 'set', []):
        key, value = entry.split('=', 1)
        if key not in cfg:
            raise ValueError('Unknown configuration key: ' + key)
        try:
            cfg[key] = json.loads(value)
        except json.JSONDecodeError:
            cfg[key] = value
    if cfg['dim'] % cfg['heads']:
        raise ValueError('dim must be divisible by heads')
    for k in ('epochs', 'batch_size', 'max_len', 'patience', 'negatives', 'source_epochs',
              'source_batch_size', 'encoder_batch_size', 'lora_rank', 'neighbors'):
        if cfg[k] < 1:
            raise ValueError(k + ' must be positive')
    if cfg['temperature'] <= 0:
        raise ValueError('temperature must be positive')
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description='StageFairRec paper-based research implementation')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('demo-data'); p.add_argument('--out', required=True)
    p.add_argument('--users', type=int, default=60); p.add_argument('--items', type=int, default=140)
    p = sub.add_parser('prepare'); p.add_argument('--raw', required=True); p.add_argument('--out', required=True)
    p.add_argument('--attribute', choices=['gender', 'age'], default='gender')
    p.add_argument('--age-bins', type=float, nargs='+')
    p.add_argument('--min-interactions', type=int, default=10)
    p.add_argument('--candidate-seed', type=int, default=2026)
    p.add_argument('--eval-negatives', type=int, default=99)
    for command in ('encode', 'train'):
        p = sub.add_parser(command)
        p.add_argument('--data', required=True); p.add_argument('--out', required=True)
        p.add_argument('--config', default='configs/default.json')
        p.add_argument('--set', action='append', default=[], metavar='KEY=JSON')
        if command == 'encode':
            p.add_argument('--mode', choices=['collaborative', 'sequential'], required=True)
            p.add_argument('--mock', action='store_true', help='Toy encoder; not valid for paper results')
        else:
            p.add_argument('--semantics'); p.add_argument('--backbone-checkpoint'); p.add_argument('--resume')
    for command in ('evaluate', 'export', 'recommend'):
        p = sub.add_parser(command)
        p.add_argument('--data', required=True); p.add_argument('--checkpoint', required=True)
        p.add_argument('--semantics'); p.add_argument('--out', required=True)
        p.add_argument('--device', default='auto')
        if command == 'evaluate':
            p.add_argument('--split', choices=['valid', 'test'], default='test')
        if command == 'recommend':
            p.add_argument('--row', type=int, required=True)
            p.add_argument('--k', type=int, default=10); p.add_argument('--exclude-seen', action='store_true')
    p = sub.add_parser('attack'); p.add_argument('--representations', required=True)
    p.add_argument('--out', required=True); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--protocol', choices=['temporal', 'user_disjoint'], default='temporal')
    p.add_argument('--kinds', nargs='+', choices=['mlp', 'logistic', 'random_forest', 'xgboost'], default=['mlp'])
    args = parser.parse_args(argv)
    if args.command == 'demo-data':
        from .data import synthetic
        synthetic(args.out, args.users, args.items)
    elif args.command == 'prepare':
        from .data import prepare
        if args.age_bins and (sorted(set(args.age_bins)) != args.age_bins):
            parser.error('Age bins must be strictly increasing')
        prepare(args.raw, args.out, args.attribute, args.min_interactions,
                args.candidate_seed, args.eval_negatives, args.age_bins)
    elif args.command in ('encode', 'train'):
        from .data import Corpus
        corpus, cfg = Corpus(args.data), config(args)
        if args.command == 'encode':
            from .semantics import encode
            encode(corpus, cfg, args.out, args.mode, args.mock)
        else:
            from .engine import fit
            fit(corpus, cfg, args.out, args.backbone_checkpoint, args.semantics, args.resume)
    elif args.command in ('evaluate', 'export', 'recommend'):
        from .data import Corpus
        from .engine import restore, evaluate, export_representations, recommend
        corpus = Corpus(args.data)
        model, cfg, cache = restore(corpus, args.checkpoint, args.semantics, args.device)
        if args.command == 'evaluate':
            result = evaluate(model, corpus, cfg, cache, 1 if args.split == 'valid' else 2)
            result.update(split=args.split, data_hash=corpus.hash, config=cfg,
                          mock_semantics=bool(cache and cache.manifest['mock']))
            write_json(args.out, result); print(json.dumps(result, indent=2))
        elif args.command == 'export':
            export_representations(model, corpus, cfg, cache, args.out)
        else:
            write_json(args.out, recommend(model, corpus, cfg, cache, args.row, args.k, args.exclude_seen))
    elif args.command == 'attack':
        from .attacks import audit
        print(json.dumps(audit(args.representations, args.out, args.kinds, args.seed, args.protocol), indent=2))


if __name__ == '__main__':
    main()
