"""Offline tiny random Llama checks the actual HF/PEFT path; no weights are downloaded."""
import pytest
import torch


def test_real_hf_lora_path(tmp_path):
    pytest.importorskip('transformers')
    pytest.importorskip('peft')
    from transformers import LlamaConfig, LlamaModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from stagefairrec.semantics import HFEncoder
    from stagefairrec.utils import read_json
    torch.set_num_threads(1)
    model_path = tmp_path / 'tiny'
    tokenizer = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, 'course': 2, 'math': 3, 'code': 4},
                                    unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='[PAD]', unk_token='[UNK]')
    fast.save_pretrained(model_path)
    LlamaModel(LlamaConfig(vocab_size=5, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=1, num_attention_heads=2,
                          num_key_value_heads=2, max_position_embeddings=64)).save_pretrained(model_path)
    cfg = read_json('configs/smoke.json') | {'model_name': str(model_path), 'gradient_checkpointing': True}
    encoder = HFEncoder(cfg, torch.device('cpu'))
    encoder.train()
    value = encoder(['course math', 'course'], 16)
    assert value.shape == (2, 16) and torch.isfinite(value).all()
    value.square().mean().backward()
    lora = [(name, p) for name, p in encoder.named_parameters() if 'lora_' in name]
    assert lora and any(p.grad is not None for _, p in lora)
    assert all(not p.requires_grad for name, p in encoder.named_parameters() if 'lora_' not in name)
    encoder.eval()
    before = encoder(['course math'], 16, adapt=False).detach()
    with torch.no_grad():
        for name, param in lora:
            if 'lora_B' in name:
                param.add_(.1)
    after = encoder(['course math'], 16, adapt=False).detach()
    assert torch.allclose(before, after)
