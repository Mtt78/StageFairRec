"""Target-conditioned cross-group InfoNCE, Eq. (14)--(16)."""
import torch
from torch.nn import functional as F


def recommendation_loss(kind, scores, learner=None, course=None):
    # PMF is an explicit squared-error implicit-feedback adaptation.
    if kind == 'pmf':
        target = torch.zeros_like(scores); target[:, 0] = 1
        return F.mse_loss(scores, target)
    if kind == 'bert4rec':
        return F.cross_entropy(scores, torch.zeros(len(scores), dtype=torch.long, device=scores.device))
    target = torch.zeros_like(scores); target[:, 0] = 1
    return F.binary_cross_entropy_with_logits(scores, target)


def tlgc_loss(z, final, users, targets, groups, neighbors=10, threshold=0., temperature=.1):
    if temperature <= 0 or neighbors < 1:
        raise ValueError('temperature and neighbors must be positive')
    # Neighbor membership is discrete; stop gradients through neighbor selection.
    with torch.no_grad():
        sim = F.normalize(z, dim=-1) @ F.normalize(z, dim=-1).T
        eligible = (users[:, None] != users[None]) & (targets[:, None] == targets[None])
        eligible &= groups[:, None] != groups[None]
        eligible &= sim >= threshold
        # At most one reference row per learner, even if a batch contains repeated users.
        refs = []
        for i in range(len(z)):
            indices = torch.where(eligible[i])[0]
            indices = indices[torch.argsort(sim[i, indices], descending=True, stable=True)]
            selected, seen = [], set()
            for j in indices.tolist():
                u = int(users[j])
                if u not in seen:
                    selected.append(j); seen.add(u)
                if len(selected) >= neighbors:
                    break
            refs.append(selected)
    logits = F.normalize(final, dim=-1) @ F.normalize(final, dim=-1).T / temperature
    # U_B is a set: deterministic first row per distinct learner in denominator.
    unique_columns, seen = [], set()
    for j, user in enumerate(users.tolist()):
        if user not in seen:
            unique_columns.append(j); seen.add(user)
    values = []
    for i, selected in enumerate(refs):
        if not selected:
            continue
        columns = [j for j in unique_columns if int(users[j]) != int(users[i])]
        # For selected users, use the matched-target row as their representative.
        replace = {int(users[j]): j for j in selected}
        columns = [replace.get(int(users[j]), j) for j in columns]
        denom = torch.logsumexp(logits[i, columns], dim=0)
        values.append(denom - logits[i, selected].mean())
    return torch.stack(values).mean() if values else final.sum() * 0
