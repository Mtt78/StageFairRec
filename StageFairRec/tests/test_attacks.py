import numpy as np
import pytest
from stagefairrec.attacks import auc, audit


def test_auc_binary_and_macro_ovr():
    assert auc(np.array([0, 1, 0, 1]), np.array([[.9, .1], [.1, .9], [.8, .2], [.2, .8]])) == 1.
    y = np.array([0, 1, 2, 0, 1, 2])
    assert auc(y, np.eye(3)[y] * .8 + .2 / 3) == 1.


def test_user_disjoint_attack(tmp_path):
    users = np.arange(60)
    labels = users % 3
    for i, split in enumerate(('train', 'valid', 'test')):
        x = np.eye(3)[labels] + np.random.default_rng(i).normal(0, .01, (60, 3))
        np.savez(tmp_path / f'{split}.npz', users=users, x=x, y=labels)
    result = audit(tmp_path, tmp_path / 'attack.json', ['logistic'], protocol='user_disjoint')
    assert result['sizes'] == [36, 12, 12]
    assert result['attacks']['logistic']['test_auc'] == pytest.approx(1.)
