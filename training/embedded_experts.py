"""Learned, checkpoint-owned experts with explicit per-conversation state.

The frozen Qwen body produces causal hidden states. Four experts read user-only
memory and a separate recurrent working state before the existing vocabulary
head. No state is hidden on the module: callers own, carry, and reset it.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


@dataclass(frozen=True)
class ExpertConfig:
    hidden_size: int
    experts: int = 4
    rank: int = 16
    max_delta_rms: float = 0.1
    format_version: int = 1

    def __post_init__(self):
        if min(self.hidden_size, self.experts, self.rank) < 1:
            raise ValueError('Expert dimensions must be positive')
        if not math.isfinite(self.max_delta_rms) or self.max_delta_rms <= 0:
            raise ValueError('Expert output bound must be finite and positive')
        if self.format_version != 1:
            raise ValueError('Unsupported embedded expert format')


@dataclass(frozen=True)
class ExpertState:
    user: torch.Tensor
    working: torch.Tensor
    positions: int = 0

    def detached(self):
        return ExpertState(self.user.detach(), self.working.detach(), self.positions)


class EmbeddedExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        e, d, r = config.experts, config.hidden_size, config.rank
        self.source = nn.Parameter(torch.randn(e, d, r) / math.sqrt(d))
        self.input = nn.Parameter(torch.randn(e, d, r) / math.sqrt(d))
        self.memory_read = nn.Parameter(torch.eye(r).repeat(e, 1, 1))
        self.recurrence = nn.Parameter(torch.full((e, r), 0.5))
        # Gates see current context, old memory, fresh content, and their difference.
        self.retain = nn.Parameter(torch.zeros(e, d + 3 * r))
        self.write = nn.Parameter(torch.zeros(e, d + 3 * r))
        self.retain_bias = nn.Parameter(torch.full((e,), 2.0))
        self.write_bias = nn.Parameter(torch.full((e,), -2.0))
        self.router = nn.Linear(d, e)
        self.router_memory = nn.Parameter(torch.zeros(e, r))
        self.router_working = nn.Parameter(torch.zeros(e, r))
        # Only the output is zero. Earlier parameters get gradients once it moves.
        self.output = nn.Parameter(torch.zeros(e, r, d))

    def initial_state(self, batch_size):
        shape = (batch_size, self.config.experts, self.config.rank)
        return ExpertState(self.source.new_zeros(shape), self.source.new_zeros(shape))

    def forward(self, hidden, user_mask, state=None, *, memory_enabled=True,
                enabled=True):
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size:
            raise ValueError('Expected hidden states [batch, tokens, hidden_size]')
        if hidden.shape[1] == 0 or user_mask.shape != hidden.shape[:2] or user_mask.dtype != torch.bool:
            raise ValueError('A boolean user-content mask is required for every token')
        if user_mask.device != hidden.device or hidden.device != self.source.device:
            raise ValueError('Hidden states, masks and experts must share a device')
        state = self.initial_state(hidden.shape[0]) if state is None else state
        expected = (hidden.shape[0], self.config.experts, self.config.rank)
        for tensor in (state.user, state.working):
            if tensor.shape != expected or tensor.device != hidden.device or tensor.dtype != self.source.dtype:
                raise ValueError('Memory state does not match the expert block')
        if state.positions < 0:
            raise ValueError('Invalid memory position')
        bank, working = state.user, state.working
        values, votes = [], []
        for pos in range(hidden.shape[1]):
            raw = hidden[:, pos]
            x = raw.to(self.source.dtype)
            x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
            fresh = torch.tanh(torch.einsum('bd,edr->ber', x, self.source))
            features = torch.cat((x[:, None].expand(-1, self.config.experts, -1),
                                  bank, fresh, fresh - bank), dim=-1)
            retain = torch.sigmoid((features * self.retain).sum(-1) + self.retain_bias)
            write = torch.sigmoid((features * self.write).sum(-1) + self.write_bias)
            candidate = torch.tanh(retain[..., None] * bank + write[..., None] * fresh)
            bank = torch.where(user_mask[:, pos, None, None], candidate, bank)
            if not memory_enabled:
                bank = torch.zeros_like(bank)
            working = torch.tanh(torch.einsum('bd,edr->ber', x, self.input)
                + torch.einsum('ber,ers->bes', bank, self.memory_read)
                + torch.tanh(self.recurrence) * working)
            gates = torch.softmax(self.router(x) + (bank * self.router_memory).sum(-1)
                                  + (working * self.router_working).sum(-1), dim=-1)
            delta = torch.einsum('be,ber,erd->bd', gates, working, self.output)
            rms = torch.sqrt(delta.square().mean(-1, keepdim=True) + 1e-12)
            delta = delta * (self.config.max_delta_rms / rms).clamp(max=1.0)
            values.append(raw + delta.to(raw.dtype) if enabled else raw)
            votes.append(gates)
        return (torch.stack(values, dim=1),
                ExpertState(bank, working, state.positions + hidden.shape[1]),
                torch.stack(votes, dim=1))


def tensor_hashes(module):
    """Hash every state tensor by logical bytes, preserving frozen-body evidence."""
    hashes = {}
    for name, value in module.state_dict().items():
        h = hashlib.sha256()
        flat = value.detach().reshape(-1)
        for chunk in flat.split(1024 * 1024):
            h.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        hashes[name] = h.hexdigest()
    return hashes


class ExpertQwen(nn.Module):
    """Frozen Qwen3.5 VLM body plus a trainable pre-head expert block.

    The first pilot accepts unpadded text. Vision weights remain part of the
    checkpoint; multimodal expert masks will need their own parity evaluation.
    """
    def __init__(self, base, config=None, base_identity=None):
        super().__init__()
        if base.config.model_type != 'qwen3_5':
            raise ValueError('Expected Qwen3.5 conditional-generation model')
        self.base = base.requires_grad_(False).eval()
        if (base.config.text_config.tie_word_embeddings and
                base.get_input_embeddings().weight.data_ptr() !=
                base.get_output_embeddings().weight.data_ptr()):
            raise ValueError('Checkpoint promises tied language matrices but they are untied')
        self.expert_config = config or ExpertConfig(base.config.text_config.hidden_size)
        if self.expert_config.hidden_size != base.config.text_config.hidden_size:
            raise ValueError('Expert width differs from the base model')
        self.experts = EmbeddedExperts(self.expert_config).to(next(base.parameters()).device)
        self.base_identity = dict(base_identity or {})

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def hidden(self, input_ids, user_mask, *, state=None, past_key_values=None,
               use_cache=False, memory_enabled=True, enabled=True):
        if input_ids.ndim != 2 or user_mask.shape != input_ids.shape:
            raise ValueError('Expected unpadded input IDs and matching user mask')
        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        if past_length != (0 if state is None else state.positions):
            raise ValueError('Qwen cache and expert memory must describe the same prefix')
        with torch.no_grad():
            result = self.base.model(input_ids=input_ids, use_cache=use_cache,
                                     past_key_values=past_key_values, return_dict=True)
        original = result.last_hidden_state.detach()
        corrected, state, votes = self.experts(original, user_mask, state,
                            memory_enabled=memory_enabled, enabled=enabled)
        return corrected, original, state, result.past_key_values, votes

    def forward(self, input_ids, user_mask, *, logits_to_keep=0, **kwargs):
        corrected, _, state, cache, votes = self.hidden(input_ids, user_mask, **kwargs)
        if logits_to_keep < 0:
            raise ValueError('logits_to_keep must be nonnegative')
        selected = corrected[:, -logits_to_keep:] if logits_to_keep else corrected
        return SimpleNamespace(logits=self.base.lm_head(selected), state=state,
                               past_key_values=cache, votes=votes)

    def save_pretrained(self, destination, *, tokenizer=None, receipt=None):
        from safetensors.torch import save_file
        path = Path(destination)
        if path.exists():
            raise ValueError('Preserve existing checkpoint: ' + str(path))
        path.mkdir(parents=True)
        self.base.save_pretrained(path, safe_serialization=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.experts.state_dict().items()},
                  str(path / 'experts.safetensors'))
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        with (path/'experts.safetensors').open('rb') as stream:
            expert_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        metadata = dict(config=asdict(self.expert_config), base_identity=self.base_identity,
                        experts_sha256=expert_sha,
                        state_policy='user writes only; working updates every token; reset per dialogue',
                        receipt=receipt or {}, frozen_body_hashes=tensor_hashes(self.base))
        # Metadata is the completion marker: partial directories cannot load.
        (path / 'embedded_experts.json').write_text(json.dumps(metadata, indent=2) + '\n')

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        from safetensors.torch import load_file
        from transformers import Qwen3_5ForConditionalGeneration
        path = Path(path)
        metadata = json.loads((path / 'embedded_experts.json').read_text())
        with (path/'experts.safetensors').open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != metadata['experts_sha256']:
                raise ValueError('Expert tensors differ from the saved checkpoint receipt')
        kwargs.setdefault('dtype', 'auto')
        base = Qwen3_5ForConditionalGeneration.from_pretrained(path, **kwargs)
        model = cls(base, ExpertConfig(**metadata['config']), metadata['base_identity'])
        if tensor_hashes(base) != metadata['frozen_body_hashes']:
            raise ValueError('Frozen base differs from the saved checkpoint receipt')
        model.experts.load_state_dict(load_file(str(path / 'experts.safetensors')), strict=True)
        return model
