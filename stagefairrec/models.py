"""Independent backbone implementations and Eqs. (4), (7)--(13)."""
import torch
from torch import nn
from torch.nn import functional as F


def mlp(din, hidden, dout):
    return nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, dout))


class Backbone(nn.Module):
    def __init__(self, users, items, cfg):
        super().__init__()
        self.kind, self.items, self.cfg = cfg['backbone'], items, cfg
        d = cfg['dim']
        self.item = nn.Embedding(items + 2, d, padding_idx=0)  # last ID = BERT mask
        self.user = nn.Embedding(users, d)
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.user.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()
        if self.kind in ('sasrec', 'bert4rec'):
            self.position = nn.Embedding(cfg['max_len'] + 1, d)
            layer = nn.TransformerEncoderLayer(d, cfg['heads'], d * 4, cfg['dropout'],
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, cfg['layers'], enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
        elif self.kind == 'deepmodel':
            # Concrete reconstruction: wide dot product plus a nonlinear matching head.
            self.deep = mlp(2 * d, cfg['hidden'], 1)
        elif self.kind != 'pmf':
            raise ValueError('Unknown backbone ' + self.kind)

    def encode_tokens(self, seq):
        pos = torch.arange(seq.shape[1], device=seq.device)
        x = self.item(seq) + self.position(pos)[None]
        causal = None
        if self.kind == 'sasrec':
            causal = torch.ones(seq.shape[1], seq.shape[1], device=seq.device,
                                dtype=torch.bool).triu(1)
        x = self.encoder(x, mask=causal, src_key_padding_mask=seq.eq(0))
        return self.norm(x)

    def learner(self, batch):
        if self.kind in ('pmf', 'deepmodel'):
            return self.user(batch['user'])
        seq = batch['seq']
        lengths = seq.ne(0).sum(1)
        if (lengths == 0).any():
            raise ValueError('Sequential models require a nonempty prefix')
        if self.kind == 'bert4rec':
            seq = F.pad(seq, (0, 1))
            seq = seq.clone()
            seq[torch.arange(len(seq), device=seq.device), lengths] = self.items + 1
            index = lengths
        else:
            index = lengths - 1
        states = self.encode_tokens(seq)
        return states[torch.arange(len(seq), device=seq.device), index]

    def score(self, learner, course):
        if course.ndim == 3:
            learner = learner[:, None, :].expand_as(course)
        score = (learner * course).sum(-1)
        if self.kind == 'deepmodel':
            score = score + self.deep(torch.cat([learner, course], -1)).squeeze(-1)
        return score

    def bert_loss(self, batch):
        # Bidirectional Cloze pretraining, 80% mask / 10% random / 10% unchanged.
        seq = F.pad(batch['seq'], (0, 1)).clone()
        lengths = seq.ne(0).sum(1)
        seq[torch.arange(len(seq), device=seq.device), lengths] = batch['target']
        active = seq.ne(0)
        chosen = (torch.rand(seq.shape, device=seq.device) < self.cfg['mask_probability']) & active
        missing = ~chosen.any(1)
        chosen[torch.arange(len(seq), device=seq.device)[missing], lengths[missing]] = True
        labels = seq[chosen].clone() - 1
        draw = torch.rand(seq.shape, device=seq.device)
        corrupted = seq.clone()
        corrupted[chosen & (draw < .8)] = self.items + 1
        random_mask = chosen & (draw >= .8) & (draw < .9)
        corrupted[random_mask] = torch.randint(1, self.items + 1,
                                   (int(random_mask.sum()),), device=seq.device)
        states = self.encode_tokens(corrupted)[chosen]
        return F.cross_entropy(states @ self.item.weight[1:self.items + 1].T, labels)


class SourceFair(nn.Module):
    def __init__(self, d, hidden, classes):
        super().__init__()
        self.extractor = mlp(d, hidden, d)
        self.classifier = mlp(d, hidden, classes)
        self.discriminator = mlp(d, hidden, classes)

    def forward(self, raw):
        sensitive = self.extractor(raw)
        return raw - sensitive, sensitive


class Fusion(nn.Module):
    def __init__(self, semantic_dim, d):
        super().__init__()
        self.project = nn.Linear(semantic_dim, d)
        self.gate = nn.Linear(2 * d, d)

    def forward(self, base, semantic):
        semantic = self.project(semantic)
        gate = torch.sigmoid(self.gate(torch.cat([base, semantic], -1)))
        return F.normalize(base + gate * semantic, dim=-1)


VARIANTS = {
    'base': (False, False, False, False),
    'llm': (True, False, False, False),
    'source_only': (True, True, False, False),
    'post_only': (True, False, True, True),
    'wo_tlgc': (True, True, True, False),
    'full': (True, True, True, True),
}


class StageFairRec(nn.Module):
    def __init__(self, backbone, semantic_dim, classes, cfg):
        super().__init__()
        self.backbone, self.cfg = backbone, cfg
        self.use_semantics, self.use_source, self.use_residual, self.use_tlgc = VARIANTS[cfg['variant']]
        d, h = cfg['dim'], cfg['hidden']
        self.user_fusion, self.item_fusion = Fusion(semantic_dim, d), Fusion(semantic_dim, d)
        self.residual = mlp(d, h, d)
        self.residual_classifier = mlp(d, h, classes)
        self.discriminator = mlp(d, h, classes)

    def learner(self, batch, semantics=None):
        p = self.backbone.learner(batch)
        if not self.use_semantics:
            return p, p, torch.zeros_like(p)
        z = self.user_fusion(p, semantics)
        r = self.residual(z) if self.use_residual else torch.zeros_like(z)
        final = F.normalize(z - r, dim=-1) if self.use_residual else z
        return final, z, r

    def courses(self, ids, semantics=None):
        p = self.backbone.item(ids)
        return self.item_fusion(p, semantics) if self.use_semantics else p
