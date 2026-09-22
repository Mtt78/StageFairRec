"""Post-hoc attackers. Validation-only selection; never update the recommender."""
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from .utils import write_json


def auc(y, probability):
    if probability.shape[1] == 2:
        return float(roc_auc_score(y, probability[:, 1]))
    return float(roc_auc_score(y, probability, multi_class='ovr', average='macro'))


def attackers(kind, seed):
    if kind == 'logistic':
        return [(f'C={c}', make_pipeline(StandardScaler(), LogisticRegression(C=c,
                    max_iter=500, random_state=seed))) for c in (.1, 1., 10.)]
    if kind == 'mlp':
        return [(f'hidden={h}', make_pipeline(StandardScaler(), MLPClassifier(
                 hidden_layer_sizes=h, max_iter=300, random_state=seed,
                 early_stopping=False))) for h in ((64,), (128, 64))]
    if kind == 'random_forest':
        return [(f'depth={d}', RandomForestClassifier(n_estimators=200, max_depth=d,
                  n_jobs=1, random_state=seed)) for d in (5, None)]
    if kind == 'xgboost':
        from xgboost import XGBClassifier
        return [(f'depth={d}', XGBClassifier(n_estimators=200, max_depth=d, n_jobs=1,
                  learning_rate=.05, random_state=seed)) for d in (3, 6)]
    raise ValueError('Unknown attacker ' + kind)


def audit(root, out, kinds, seed=42, protocol='temporal'):
    root = Path(root)
    data = [dict(np.load(root / f'{name}.npz', allow_pickle=False))
            for name in ('train', 'valid', 'test')]
    if protocol == 'user_disjoint':
        # Same seeded user partition is used across methods. Each side retains its own temporal state.
        common = sorted(set(data[0]['users']) & set(data[1]['users']) & set(data[2]['users']))
        labels = dict(zip(data[0]['users'], data[0]['y']))
        train_ids, holdout = train_test_split(common, test_size=.4, random_state=seed,
                                            stratify=[labels[i] for i in common])
        valid_ids, test_ids = train_test_split(holdout, test_size=.5, random_state=seed,
                                             stratify=[labels[i] for i in holdout])
        for i, ids in enumerate((train_ids, valid_ids, test_ids)):
            keep = np.isin(data[i]['users'], ids)
            data[i] = {k: v[keep] for k, v in data[i].items()}
    classes = np.unique(data[0]['y'])
    for d in data:
        if not np.array_equal(np.unique(d['y']), classes):
            raise ValueError('Every attack split must contain all classes; adjust data or split size')
        if not np.isfinite(d['x']).all():
            raise ValueError('Non-finite representations')
    train, valid, test = data
    result = dict(protocol=protocol, seed=seed, sizes=[len(d['y']) for d in data], attacks={})
    for kind in kinds:
        best, selected, classifier = -1., None, None
        tuning = []
        for name, candidate in attackers(kind, seed):
            candidate.fit(train['x'], train['y'])
            score = auc(valid['y'], candidate.predict_proba(valid['x']))
            tuning.append(dict(parameters=name, validation_auc=score))
            if score > best:
                best, selected, classifier = score, name, candidate
        test_score = auc(test['y'], classifier.predict_proba(test['x']))
        result['attacks'][kind] = dict(selected=selected, validation_auc=best,
            test_auc=test_score, distance_from_chance=abs(test_score - .5), tuning=tuning)
    result['note'] = ('Temporal attack partitions share learner identities; collaborative states '
        'can be identical across splits. Use user_disjoint as an additional unseen-user audit. '
        'AUC below .5 is not stronger fairness: label inversion may recover information.')
    write_json(out, result)
    return result
