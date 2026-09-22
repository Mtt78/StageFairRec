import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_of(name):
    return torch.device('cuda' if name == 'auto' and torch.cuda.is_available()
                        else 'cpu' if name == 'auto' else name)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def data_digest(root):
    return hashlib.sha256(''.join(digest(Path(root) / x) for x in
        ['metadata.json', 'rows.npy', 'candidates.npy']).encode()).hexdigest()


def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=True)


def freeze(module, frozen=True):
    for p in module.parameters():
        p.requires_grad_(not frozen)


def save_checkpoint(path, model, config, data_hash, **extra):
    # CPU tensors make the artifacts portable. Only load checkpoints you trust.
    torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    config=config, data_hash=data_hash, **extra), path)
