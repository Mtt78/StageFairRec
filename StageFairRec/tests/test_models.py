import math
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from stagefairrec.models import Backbone, StageFairRec
from stagefairrec.losses import tlgc_loss
from stagefairrec.engine import summarize_ranks
from stagefairrec.semantics import mean_pool
from stagefairrec.utils import read_json, freeze


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    return read_json('configs/smoke.json')


@pytest.mark.parametrize('kind', ['pmf', 'deepmodel', 'sasrec', 'bert4rec'])
def test_backbone_scoring_and_backward(cfg, kind):
    cfg['backbone'] = kind
    model = Backbone(5, 20, cfg)
    batch = dict(user=torch.tensor([0, 1]), seq=torch.tensor([[1, 0, 0], [2, 3, 0]]),
                 target=torch.tensor([2, 4]))
    z = model.learner(batch)
    scores = model.score(z, model.item(torch.tensor([[2, 3], [4, 5]])))
    assert scores.shape == (2, 2) and torch.isfinite(scores).all()
    loss = model.bert_loss(batch) if kind == 'bert4rec' else scores.square().mean()
    loss.backward()
    assert model.item.weight.grad is not None


def test_causal_prefix_invariance(cfg):
    cfg['backbone'] = 'sasrec'
    model = Backbone(3, 20, cfg).eval()
    a = model.encode_tokens(torch.tensor([[1, 2, 3]]))
    b = model.encode_tokens(torch.tensor([[1, 2, 9]]))
    assert torch.allclose(a[:, :2], b[:, :2], atol=1e-6)


def test_post_fusion_freezes_backbone_and_adversary(cfg):
    cfg['backbone'] = 'pmf'
    m = StageFairRec(Backbone(4, 20, cfg), 24, 2, cfg)
    freeze(m.backbone); freeze(m.discriminator)
    batch = dict(user=torch.tensor([0, 1]))
    final, z, _ = m.learner(batch, torch.randn(2, 24))
    r = m.residual(z.detach())
    loss = F.cross_entropy(m.residual_classifier(r), torch.tensor([0, 1]))
    loss -= .1 * F.cross_entropy(m.discriminator(F.normalize(z.detach() - r, dim=-1)), torch.tensor([0, 1]))
    loss.backward()
    assert all(p.grad is None for p in m.backbone.parameters())
    assert all(p.grad is None for p in m.discriminator.parameters())
    assert all(p.grad is None for p in m.user_fusion.parameters())
    assert any(p.grad is not None for p in m.residual.parameters())


def test_tlgc_exact_and_empty():
    z = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], requires_grad=True)
    users, target, group = torch.tensor([0, 1, 2]), torch.tensor([7, 7, 9]), torch.tensor([0, 1, 0])
    loss = tlgc_loss(z, z, users, target, group, neighbors=1, threshold=.5, temperature=1.)
    assert float(loss.detach()) == pytest.approx(math.log(math.e + 1) - 1)
    loss.backward(); assert torch.isfinite(z.grad).all()
    empty = tlgc_loss(z, z, users, target, torch.zeros(3, dtype=torch.long))
    assert float(empty.detach()) == 0 and empty.requires_grad


def test_tlgc_excludes_same_learner():
    z = torch.randn(2, 4, requires_grad=True)
    result = tlgc_loss(z, z, torch.tensor([1, 1]), torch.tensor([2, 2]), torch.tensor([0, 1]), threshold=-1)
    assert float(result.detach()) == 0


def test_masked_pooling_and_metrics():
    h = torch.tensor([[[2., 4.], [4., 6.], [999., 999.]]])
    assert torch.equal(mean_pool(h, torch.tensor([[1, 1, 0]])), torch.tensor([[3., 5.]]))
    metrics = summarize_ranks(np.array([1, 2, 11]), np.array([0, 0, 1]), [10])
    assert metrics['HR@10'] == pytest.approx(2/3)
    assert metrics['gap_HR@10'] == 1.
    assert metrics['NDCG@10'] == pytest.approx((1 + 1/math.log2(3))/3)
