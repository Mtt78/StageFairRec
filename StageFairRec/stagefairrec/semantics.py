"""Offline frozen course encoding, learner LoRA adversarial training and prefix caches."""
import contextlib
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .models import SourceFair
from .utils import device_of, seed_all, freeze, write_json, read_json, digest


def mean_pool(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


class MockEncoder(nn.Module):
    """Deterministic toy token features + trainable low-rank adapter. NOT an LLM."""
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg['mock_dim']
        self.down = nn.Linear(self.dim, cfg['lora_rank'], bias=False)
        self.up = nn.Linear(cfg['lora_rank'], self.dim, bias=False)
        nn.init.zeros_(self.up.weight)
        self.scale = cfg['lora_alpha'] / cfg['lora_rank']

    def forward(self, texts, limit, adapt=True, history=True):
        vectors = []
        for text in texts:
            words = text.split()
            words = words[-limit:] if history else words[:limit]
            vec = np.zeros(self.dim, dtype=np.float32)
            for word in words:
                h = hashlib.sha256(word.encode()).digest()
                vec[int.from_bytes(h[:4], 'little') % self.dim] += 1 if h[4] % 2 else -1
            vec /= max(1, len(words)) ** .5
            vectors.append(vec)
        raw = torch.as_tensor(np.stack(vectors), device=self.down.weight.device)
        return raw + self.up(self.down(raw)) * self.scale if adapt else raw


class HFEncoder(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        from peft import LoraConfig, TaskType, get_peft_model
        common = dict(revision=cfg['revision'], trust_remote_code=cfg['trust_remote_code'])
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], **common)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError('Tokenizer requires a pad or EOS token')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        base = AutoModel.from_pretrained(cfg['model_name'], torch_dtype=dtype, **common)
        self.dim = base.config.hidden_size
        base.config.use_cache = False
        self.model = get_peft_model(base, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
            r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'],
            target_modules=cfg['lora_targets'], bias='none'))
        if cfg['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.to(device)

    def forward(self, texts, limit, adapt=True, history=True):
        self.tokenizer.truncation_side = 'left' if history else 'right'
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=limit,
                                return_tensors='pt').to(next(self.model.parameters()).device)
        context = contextlib.nullcontext() if adapt else self.model.disable_adapter()
        with context:
            output = self.model(**tokens, return_dict=True)
        if output.last_hidden_state.shape[:2] != tokens['input_ids'].shape:
            raise ValueError('Encoder hidden state layout is not batch-first; add an explicit adapter')
        return mean_pool(output.last_hidden_state.float(), tokens['attention_mask'])


def prompt(corpus, history):
    texts = corpus.meta['course_texts']
    # No sensitive labels or target field appears in prompts.
    return 'Interacted courses in chronological order:\n' + '\n\n'.join(texts[i] for i in history)


def encode(corpus, cfg, out, mode, mock=False):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Semantic output must be empty to avoid stale cache mixing')
    out.mkdir(parents=True, exist_ok=True)
    seed_all(cfg['seed'])
    device = device_of(cfg['device'])
    encoder = (MockEncoder(cfg).to(device) if mock else HFEncoder(cfg, device))
    source = SourceFair(encoder.dim, cfg['hidden'], corpus.meta['classes']).to(device)
    main_parameters = [p for p in encoder.parameters() if p.requires_grad]
    main_parameters += list(source.extractor.parameters()) + list(source.classifier.parameters())
    opt = torch.optim.Adam(main_parameters, lr=cfg['source_lr'])
    adv_opt = torch.optim.Adam(source.discriminator.parameters(), lr=cfg['source_lr'])
    labels = torch.tensor(corpus.meta['labels'], dtype=torch.long, device=device)
    terminal = [prompt(corpus, s[:corpus.meta['cuts'][u][0]])
                for u, s in enumerate(corpus.meta['sequences'])]
    rng = np.random.default_rng(cfg['seed'])
    for epoch in range(cfg['source_epochs']):
        encoder.train(); source.train()
        total = 0.
        order = rng.permutation(len(terminal))
        for offset in range(0, len(order), cfg['source_batch_size']):
            ix = order[offset:offset + cfg['source_batch_size']]
            raw = encoder([terminal[i] for i in ix], cfg['learner_tokens'])
            fair, sensitive = source(raw)
            adv_opt.zero_grad(set_to_none=True)
            F.cross_entropy(source.discriminator(fair.detach()), labels[ix]).backward()
            adv_opt.step()
            freeze(source.discriminator)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(source.classifier(sensitive), labels[ix])
            loss -= cfg['source_lambda'] * F.cross_entropy(source.discriminator(fair), labels[ix])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite source loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_parameters, 5.)
            opt.step()
            freeze(source.discriminator, False)
            total += float(loss.detach())
        print(f'source epoch={epoch + 1} summed_loss={total:.6f}', flush=True)
    encoder.eval(); source.eval()
    torch.save(source.state_dict(), out / 'source.pt')
    if mock:
        torch.save(encoder.state_dict(), out / 'mock_encoder.pt')
    else:
        encoder.model.save_pretrained(out / 'learner_adapter')
        encoder.tokenizer.save_pretrained(out / 'learner_adapter')
    size = len(corpus.rows) if mode == 'sequential' else corpus.meta['num_users']
    arrays = {name: np.lib.format.open_memmap(out / f'{name}.npy', mode='w+',
              dtype=np.float32, shape=(size, encoder.dim)) for name in ('plain', 'fair')}
    courses = np.lib.format.open_memmap(out / 'courses.npy', mode='w+', dtype=np.float32,
                      shape=(corpus.meta['num_items'] + 1, encoder.dim))
    courses[0] = 0
    bs = cfg['encoder_batch_size'] if not mock else 128
    with torch.inference_mode():
        for start in range(1, len(courses), bs):
            texts = corpus.meta['course_texts'][start:start + bs]
            courses[start:start + len(texts)] = encoder(texts, cfg['course_tokens'],
                                                       adapt=False, history=False).cpu().numpy()
        for start in range(0, size, bs):
            stop = min(start + bs, size)
            texts = ([prompt(corpus, corpus.history(i, mode)) for i in range(start, stop)]
                     if mode == 'sequential' else terminal[start:stop])
            arrays['plain'][start:stop] = encoder(texts, cfg['learner_tokens'], adapt=False).cpu().numpy()
            adapted = encoder(texts, cfg['learner_tokens'])
            arrays['fair'][start:stop] = source(adapted)[0].cpu().numpy()
    for arr in [courses, *arrays.values()]:
        arr.flush()
    write_json(out / 'manifest.json', dict(data_hash=corpus.hash, mode=mode, mock=mock,
        semantic_dim=encoder.dim, config=cfg,
        source_history='one training-terminal prompt per learner',
        files={name: digest(out / name) for name in ('courses.npy', 'plain.npy', 'fair.npy', 'source.pt')}))


class SemanticCache:
    def __init__(self, root, corpus, cfg):
        root = Path(root)
        self.manifest = read_json(root / 'manifest.json')
        if self.manifest['data_hash'] != corpus.hash:
            raise ValueError('Semantic cache belongs to another prepared dataset')
        mode = 'sequential' if cfg['backbone'] in ('sasrec', 'bert4rec') else 'collaborative'
        if self.manifest['mode'] != mode:
            raise ValueError('Semantic cache history mode does not match backbone')
        from .models import VARIANTS
        self.mode = mode
        source = VARIANTS[cfg['variant']][1]
        self.name = 'fair' if source else 'plain'
        self.course = np.load(root / 'courses.npy', mmap_mode='r')
        self.learner = np.load(root / f'{self.name}.npy', mmap_mode='r')
        self.dim = self.manifest['semantic_dim']
        for filename in ('courses.npy', f'{self.name}.npy'):
            if digest(root / filename) != self.manifest['files'][filename]:
                raise ValueError('Semantic cache checksum mismatch: ' + filename)
        self.fingerprint = hashlib.sha256((self.manifest['files']['courses.npy'] +
                           self.manifest['files'][f'{self.name}.npy']).encode()).hexdigest()

    def learners(self, batch, device):
        index = batch['row'] if self.mode == 'sequential' else batch['user']
        return torch.tensor(np.array(self.learner[index.detach().cpu().numpy()]), device=device)

    def courses(self, ids, device):
        return torch.tensor(np.array(self.course[ids.detach().cpu().numpy()]), device=device)
