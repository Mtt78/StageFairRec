"""Backbone pretraining, frozen-backbone calibration, ranking and representation export."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import Examples, negatives
from .models import Backbone, StageFairRec
from .losses import recommendation_loss, tlgc_loss
from .semantics import SemanticCache
from .utils import seed_all, device_of, freeze, save_checkpoint, load_checkpoint, write_json


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def forward(model, batch, ids, cache, device):
    h = cache.learners(batch, device) if cache else None
    e = cache.courses(ids, device) if cache else None
    final, z, residual = model.learner(batch, h)
    course = model.courses(ids, e)
    return model.backbone.score(final, course), final, z, residual


def summarize_ranks(ranks, labels, ks):
    result = {}
    for k in ks:
        hit = (ranks <= k).astype(float)
        ndcg = hit / np.log2(ranks + 1)
        groups = {}
        for group in sorted(set(labels.tolist())):
            selected = labels == group
            groups[str(group)] = {'count': int(selected.sum()), 'HR': float(hit[selected].mean()),
                                   'NDCG': float(ndcg[selected].mean())}
        result[f'HR@{k}'] = float(hit.mean())
        result[f'NDCG@{k}'] = float(ndcg.mean())
        result[f'groups@{k}'] = groups
        for name in ('HR', 'NDCG'):
            values = [g[name] for g in groups.values()]
            result[f'gap_{name}@{k}'] = max(values) - min(values)
    result['instances'] = len(ranks)
    return result


@torch.inference_mode()
def evaluate(model, corpus, cfg, cache, split=1):
    model.eval()
    device = next(model.parameters()).device
    ranks, labels = [], []
    for batch in DataLoader(Examples(corpus, split, cfg), batch_size=cfg['batch_size']):
        ids = torch.tensor(np.array(corpus.candidates[batch['candidate_index'].numpy()]), device=device)
        batch = move(batch, device)
        scores, *_ = forward(model, batch, ids, cache, device)
        # Target is index 0; pessimistic tie handling avoids target-first tie advantage.
        rank = 1 + (scores[:, 1:] >= scores[:, :1]).sum(1)
        ranks.extend(rank.cpu().tolist()); labels.extend(batch['label'].cpu().tolist())
    if not ranks:
        raise ValueError('Empty evaluation split')
    return summarize_ranks(np.asarray(ranks), np.asarray(labels), sorted(set(cfg['ks'] + [10])))


def build(corpus, cfg, semantic_dim=1):
    backbone = Backbone(corpus.meta['num_users'], corpus.meta['num_items'], cfg)
    return StageFairRec(backbone, semantic_dim, corpus.meta['classes'], cfg)


def fit(corpus, cfg, out, backbone_path=None, semantics=None, resume=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and not resume:
        raise ValueError('Run exists; choose a new output directory or use --resume')
    seed_all(cfg['seed'])
    device = device_of(cfg['device'])
    calibrate = cfg['variant'] != 'base'
    cache = SemanticCache(semantics, corpus, cfg) if calibrate else None
    model = build(corpus, cfg, cache.dim if cache else 1).to(device)
    if calibrate:
        if not backbone_path:
            raise ValueError('Calibrated variants require --backbone-checkpoint')
        original = load_checkpoint(backbone_path)
        if original['data_hash'] != corpus.hash or original['config']['variant'] != 'base':
            raise ValueError('Expected base checkpoint from same dataset')
        for key in ('backbone', 'dim', 'heads', 'layers', 'max_len', 'hidden'):
            if original['config'][key] != cfg[key]:
                raise ValueError('Backbone configuration differs: ' + key)
        base_weights = {k.removeprefix('backbone.'): v for k, v in original['model'].items()
                        if k.startswith('backbone.')}
        model.backbone.load_state_dict(base_weights)
        freeze(model.backbone)
    else:
        # Unused fairness heads should not appear in the backbone optimizer.
        freeze(model)
        freeze(model.backbone, False)
    parameters = [p for name, p in model.named_parameters()
                  if p.requires_grad and not name.startswith('discriminator.')]
    opt = torch.optim.Adam(parameters, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    adv_opt = torch.optim.Adam(model.discriminator.parameters(), lr=cfg['lr']) if model.use_residual else None
    rng = np.random.default_rng(cfg['seed'])
    start, best, stale = 0, -1., 0
    if resume:
        state = load_checkpoint(resume, device)
        if state['config'] != cfg or state['data_hash'] != corpus.hash:
            raise ValueError('Resume configuration/data mismatch')
        if state.get('cache_hash') != (cache.fingerprint if cache else None):
            raise ValueError('Resume semantic cache mismatch')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        if adv_opt:
            adv_opt.load_state_dict(state['adv_optimizer'])
        start, best, stale = state['epoch'] + 1, state['best'], state['stale']
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available() and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        if not (out / 'best.pt').exists():
            raise ValueError('Resume in original output directory containing best.pt')
    write_json(out / 'config.json', cfg)
    data = Examples(corpus, 0, cfg)
    for epoch in range(start, cfg['epochs']):
        model.train()
        if calibrate:
            model.backbone.eval()  # frozen dropout stays disabled
        summed, steps, aligned = 0., 0, 0
        for batch in DataLoader(data, batch_size=cfg['batch_size'], shuffle=True):
            neg = negatives(corpus, batch['user'], cfg['negatives'], rng).to(device)
            batch = move(batch, device)
            ids = torch.cat([batch['target'][:, None], neg], dim=1)
            opt.zero_grad(set_to_none=True)
            if not calibrate and cfg['backbone'] == 'bert4rec':
                loss = model.backbone.bert_loss(batch)
            else:
                scores, final, z, residual = forward(model, batch, ids, cache, device)
                if adv_opt:
                    adv_opt.zero_grad(set_to_none=True)
                    F.cross_entropy(model.discriminator(final.detach()), batch['label']).backward()
                    adv_opt.step()
                    freeze(model.discriminator)
                loss = recommendation_loss(cfg['backbone'], scores)
                if adv_opt:
                    # Eq. (13): PFD updates only residual extractor/classifier.
                    r_detached_input = model.residual(z.detach())
                    fair_detached_input = F.normalize(z.detach() - r_detached_input, dim=-1)
                    pfd = F.cross_entropy(model.residual_classifier(r_detached_input), batch['label'])
                    pfd -= cfg['beta'] * F.cross_entropy(model.discriminator(fair_detached_input), batch['label'])
                    loss = loss + cfg['lambda1'] * pfd
                if model.use_tlgc:
                    local = tlgc_loss(z, final, batch['user'], batch['target'], batch['label'],
                              cfg['neighbors'], cfg['similarity_threshold'], cfg['temperature'])
                    loss = loss + cfg['lambda2'] * local
                    aligned += int(float(local.detach()) != 0.)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            opt.step()
            if adv_opt:
                freeze(model.discriminator, False)
            summed += float(loss.detach()); steps += 1
        metrics = evaluate(model, corpus, cfg, cache)
        score = metrics['HR@10']
        if score > best:
            best, stale = score, 0
            save_checkpoint(out / 'best.pt', model, cfg, corpus.hash, epoch=epoch,
                            cache_hash=cache.fingerprint if cache else None,
                            semantic_dim=cache.dim if cache else 1)
        else:
            stale += 1
        record = dict(epoch=epoch + 1, train_loss=summed / max(1, steps),
                      batches_with_tlgc=aligned, valid=metrics)
        with open(out / 'history.jsonl', 'a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        save_checkpoint(out / 'last.pt', model, cfg, corpus.hash, epoch=epoch, best=best, stale=stale,
            semantic_dim=cache.dim if cache else 1, cache_hash=cache.fingerprint if cache else None,
            optimizer=opt.state_dict(), adv_optimizer=adv_opt.state_dict() if adv_opt else None,
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        if stale >= cfg['patience']:
            break
    if not (out / 'best.pt').exists():
        raise ValueError('No checkpoint: epochs must be positive')
    model.load_state_dict(load_checkpoint(out / 'best.pt', device)['model'])
    write_json(out / 'validation.json', evaluate(model, corpus, cfg, cache))
    return model


def restore(corpus, checkpoint, semantics=None, device='auto'):
    state = load_checkpoint(checkpoint)
    cfg = state['config']
    if state['data_hash'] != corpus.hash:
        raise ValueError('Checkpoint data mismatch')
    cache = SemanticCache(semantics, corpus, cfg) if cfg['variant'] != 'base' else None
    if state.get('cache_hash') != (cache.fingerprint if cache else None):
        raise ValueError('Checkpoint cache mismatch')
    model = build(corpus, cfg, state['semantic_dim']).to(device_of(device))
    model.load_state_dict(state['model']); model.eval()
    return model, cfg, cache


@torch.inference_mode()
def export_representations(model, corpus, cfg, cache, out):
    # One average scoring representation per learner per temporal split.
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model.eval(); device = next(model.parameters()).device
    for split, name in enumerate(('train', 'valid', 'test')):
        sums = np.zeros((corpus.meta['num_users'], cfg['dim']), dtype=np.float64)
        counts = np.zeros(corpus.meta['num_users'], dtype=np.int64)
        for batch in DataLoader(Examples(corpus, split, cfg), batch_size=cfg['batch_size']):
            batch = move(batch, device)
            h = cache.learners(batch, device) if cache else None
            representation = model.learner(batch, h)[0].cpu().numpy()
            users = batch['user'].cpu().numpy()
            np.add.at(sums, users, representation); np.add.at(counts, users, 1)
        ids = np.flatnonzero(counts)
        np.savez(out / f'{name}.npz', x=(sums[ids] / counts[ids, None]).astype(np.float32),
                 y=np.array(corpus.meta['labels'])[ids], users=ids)
    write_json(out / 'manifest.json', dict(data_hash=corpus.hash, config=cfg,
        cache_hash=cache.fingerprint if cache else None,
        aggregation='mean scoring representation per user per temporal split'))


@torch.inference_mode()
def recommend(model, corpus, cfg, cache, row, k=10, exclude_seen=False):
    if not 0 <= row < len(corpus.rows):
        raise ValueError('Invalid row index')
    split = int(corpus.rows[row, 3])
    dataset = Examples(corpus, split, cfg)
    found = np.flatnonzero(dataset.ids == row)
    if not len(found):
        raise ValueError('No nonempty sequential history for this row')
    batch = {key: torch.as_tensor(value).unsqueeze(0) for key, value in dataset[int(found[0])].items()}
    device = next(model.parameters()).device; batch = move(batch, device)
    ids = torch.arange(1, corpus.meta['num_items'] + 1, device=device)[None]
    scores = forward(model, batch, ids, cache, device)[0][0]
    if exclude_seen:
        mode = 'sequential' if cfg['backbone'] in ('sasrec', 'bert4rec') else 'collaborative'
        history = corpus.history(row, mode)
        if history:
            scores[torch.tensor(history, device=device) - 1] = -torch.inf
    eligible = torch.where(torch.isfinite(scores))[0]
    chosen = eligible[torch.argsort(scores[eligible], descending=True, stable=True)[:k]]
    return [{'course_id': corpus.meta['course_ids'][int(i) + 1], 'score': float(scores[i])}
            for i in chosen]
