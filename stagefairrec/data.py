"""Canonical CSV ingestion; no silent demographic inference or downloaded data."""
import csv
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import read_json, write_json, data_digest


def prepare(raw, out, attribute='gender', min_interactions=10, candidate_seed=2026,
            eval_negatives=99, age_bins=None):
    raw, out = Path(raw), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty; preserve fixed candidates.')
    out.mkdir(parents=True, exist_ok=True)
    with open(raw / 'courses.csv', newline='', encoding='utf-8') as f:
        courses = list(csv.DictReader(f))
    ids = [r['course_id'] for r in courses]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate course_id')
    item_map = {x: i + 1 for i, x in enumerate(ids)}
    with open(raw / 'users.csv', newline='', encoding='utf-8') as f:
        users = list(csv.DictReader(f))
    if len({r['user_id'] for r in users}) != len(users):
        raise ValueError('Duplicate user_id')
    labels = {}
    for r in users:
        value = r.get(attribute, '').strip()
        if attribute == 'gender':
            if value not in {'0', '1'}:
                continue
            value = int(value)
        elif attribute == 'age':
            if not age_bins:
                raise ValueError('Age boundaries must be supplied explicitly via --age-bins')
            try:
                age = float(value)
            except ValueError:
                continue
            if not math.isfinite(age) or not 0 < age < 120:
                continue
            value = int(np.searchsorted(age_bins, age, side='right'))
        else:
            raise ValueError('Supported attributes: gender, age')
        labels[r['user_id']] = value
    histories = defaultdict(list)
    with open(raw / 'interactions.csv', newline='', encoding='utf-8') as f:
        for order, r in enumerate(csv.DictReader(f)):
            if r['course_id'] not in item_map:
                raise ValueError('Interaction references unknown course: ' + r['course_id'])
            if r['user_id'] not in labels:
                continue
            t = float(r['timestamp'])
            if not math.isfinite(t):
                raise ValueError('timestamp must be finite numeric seconds')
            histories[r['user_id']].append((t, order, item_map[r['course_id']]))
    kept = sorted(u for u, h in histories.items() if len(h) >= min_interactions)
    if not kept:
        raise ValueError('No eligible learners remain')
    groups = sorted({labels[u] for u in kept})
    if len(groups) < 2:
        raise ValueError('Fairness training needs at least two nonempty groups')
    sequences, cuts, rows, candidates = [], [], [], []
    rng = np.random.default_rng(candidate_seed)
    for uid, original in enumerate(kept):
        seq = [x[2] for x in sorted(histories[original])]
        n = len(seq)
        nt, nv = int(n * 0.8), max(1, int(n * 0.1))
        if nt < 2 or nt + nv >= n:
            raise ValueError('Insufficient interactions for chronological 8:1:1 split')
        sequences.append(seq)
        cuts.append([nt, nt + nv])
        unseen = np.array(sorted(set(item_map.values()) - set(seq)), dtype=np.int64)
        if len(unseen) < eval_negatives:
            raise ValueError(f'Learner {original}: fewer than {eval_negatives} unseen courses; '
                             'reduce candidates explicitly for a non-paper smoke test')
        for t, target in enumerate(seq):
            split = 0 if t < nt else 1 if t < nt + nv else 2
            ci = -1
            if split:
                ci = len(candidates)
                candidates.append([target, *rng.choice(unseen, eval_negatives, replace=False).tolist()])
            rows.append([uid, t, target, split, ci])
    metadata = dict(user_ids=kept, course_ids=['<PAD>', *ids],
        course_texts=['', *['\n'.join(f'{k}: {r[k]}' for k in sorted(r)
                          if k != 'course_id' and r[k]) for r in courses]],
        sequences=sequences, cuts=cuts, labels=[groups.index(labels[u]) for u in kept],
        group_values=groups, attribute=attribute, age_bins=age_bins,
        num_users=len(kept), num_items=len(ids), classes=len(groups),
        candidate_seed=candidate_seed, eval_negatives=eval_negatives,
        duplicate_policy='retain events; sort by timestamp then input row order',
        split_policy='per-user floor(0.8*n), floor(0.1*n), remainder')
    write_json(out / 'metadata.json', metadata)
    np.save(out / 'rows.npy', np.asarray(rows, dtype=np.int64))
    np.save(out / 'candidates.npy', np.asarray(candidates, dtype=np.int64))
    write_json(out / 'manifest.json', dict(data_hash=data_digest(out), rows=len(rows),
               users=len(kept), courses=len(ids), synthetic=(raw / 'SYNTHETIC').exists()))


class Corpus:
    def __init__(self, root):
        self.root = Path(root)
        self.meta = read_json(self.root / 'metadata.json')
        self.rows = np.load(self.root / 'rows.npy', mmap_mode='r')
        self.candidates = np.load(self.root / 'candidates.npy', mmap_mode='r')
        self.hash = data_digest(root)
        self.unseen = [np.array(sorted(set(range(1, self.meta['num_items'] + 1)) - set(s)),
                                  dtype=np.int64) for s in self.meta['sequences']]

    def history(self, row, mode):
        u, t = map(int, self.rows[row, :2])
        end = t if mode == 'sequential' else self.meta['cuts'][u][0]
        return self.meta['sequences'][u][:end]


class Examples(Dataset):
    def __init__(self, corpus, split, config):
        self.c = corpus
        self.config = config
        self.sequential = config['backbone'] in ('sasrec', 'bert4rec')
        mask = corpus.rows[:, 3] == split
        if self.sequential:
            mask &= corpus.rows[:, 1] > 0
        self.ids = np.flatnonzero(mask)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        rid = int(self.ids[index])
        u, t, target, split, ci = map(int, self.c.rows[rid])
        # Right padding avoids all-masked queries in causal attention.
        history = self.c.meta['sequences'][u][:t][-self.config['max_len']:]
        seq = np.zeros(self.config['max_len'], dtype=np.int64)
        seq[:len(history)] = history
        return dict(row=rid, user=u, seq=torch.from_numpy(seq), length=len(history),
                    target=target, label=self.c.meta['labels'][u], candidate_index=ci)


def negatives(corpus, users, count, rng):
    return torch.tensor(np.stack([rng.choice(corpus.unseen[int(u)], count, replace=True)
                                 for u in users]), dtype=torch.long)


def synthetic(out, users=60, items=140, seed=7):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    def write(name, fields, rows):
        with open(out / name, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(fields); w.writerows(rows)
    write('courses.csv', ['course_id', 'name', 'about', 'field'],
          [[i, f'Course {i}', f'Introduction to topic {i % 7}', f'Topic {i % 7}']
           for i in range(items)])
    write('users.csv', ['user_id', 'gender', 'age'],
          [[u, u % 2, 18 + u % 40] for u in range(users)])
    rows = []
    for u in range(users):
        for t, item in enumerate(rng.choice(items, 12, replace=False)):
            rows.append([u, item, 1_600_000_000 + t * 3600])
    write('interactions.csv', ['user_id', 'course_id', 'timestamp'], rows)
    (out / 'SYNTHETIC').write_text('Generated toy data. Not paper experiments.\n')
