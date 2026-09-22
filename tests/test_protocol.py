import numpy as np
import pytest
from stagefairrec.data import Corpus, prepare, synthetic


@pytest.fixture
def corpus(tmp_path):
    raw = tmp_path / 'raw'
    synthetic(raw, users=10)
    prepare(raw, tmp_path / 'data')
    return Corpus(tmp_path / 'data')


def test_fixed_candidates_and_prefix_visibility(corpus, tmp_path):
    c = corpus
    for row, (u, t, target, split, ci) in enumerate(c.rows):
        assert c.history(row, 'sequential') == c.meta['sequences'][u][:t]
        assert len(c.history(row, 'sequential')) == t
        assert c.history(row, 'collaborative') == c.meta['sequences'][u][:c.meta['cuts'][u][0]]
        if split:
            ids = c.candidates[ci]
            assert ids[0] == target and len(ids) == 100 and len(set(ids)) == 100
            assert not set(ids[1:]) & set(c.meta['sequences'][u])
    prepare(c.root.parent / 'raw', tmp_path / 'again')
    other = Corpus(tmp_path / 'again')
    assert c.hash == other.hash
    assert np.array_equal(c.candidates, other.candidates)


def test_age_requires_explicit_boundaries(tmp_path):
    synthetic(tmp_path / 'raw')
    with pytest.raises(ValueError, match='boundaries'):
        prepare(tmp_path / 'raw', tmp_path / 'age', attribute='age')


def test_nonempty_output_is_protected(corpus):
    with pytest.raises(ValueError, match='empty'):
        prepare(corpus.root.parent / 'raw', corpus.root)
