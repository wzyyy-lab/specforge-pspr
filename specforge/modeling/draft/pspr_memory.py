"""PSPR-Memory: opt-in warm-startable residual reader of verified context.

Uses the LAST ALREADY CAPTURED target layer strictly before the block anchor.
No additional target call or vocabulary matrix. The mature cloze scorer is kept
and a zero-output residual is added. Identically shaped blind/noisy-future
controls distinguish context value from capacity. This is an experiment, not a
claim that bidirectionality is impossible or that oracle information is free.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .pspr_cloze import ClozeCorrector, PSPRClozeDraftModel
from .registry import register_draft


def gather_verified_memory(target_hidden, anchor_positions, length, hidden_size):
    """[B,S,layers*D], [B,N] -> [B,N,L,D], [B,N,L]; STRICTLY positions < a."""
    if length < 1 or target_hidden.shape[-1] % hidden_size:
        raise ValueError("invalid memory length or captured feature width")
    positions = anchor_positions[..., None] + torch.arange(
        -length, 0, device=anchor_positions.device)
    valid = (positions >= 0) & (positions < target_hidden.shape[1])
    safe = positions.clamp(0, target_hidden.shape[1] - 1)
    batch = torch.arange(target_hidden.shape[0], device=safe.device)[:, None, None]
    memory = target_hidden[..., -hidden_size:][batch, safe].detach()
    return memory.masked_fill(~valid[..., None], 0), valid


class VerifiedMemoryCache:
    """Request-local rolling accepted-feature cache. Never store proposal states.

    start is the absolute start of the newly verified chunk. Zero resets the
    request. Gaps/overlaps raise instead of silently duplicating a prefix.
    Training does not use this mutable cache; its gather is explicit and pure.
    """
    def __init__(self, length, hidden_size):
        self.length, self.hidden_size = int(length), int(hidden_size)
        self.end, self.values = 0, None

    def append(self, target_hidden, start):
        start = int(start)
        if start == 0:
            self.end, self.values = 0, None
        if start != self.end:
            raise ValueError(f"verified memory discontinuity: {start=} != end={self.end}")
        chunk = target_hidden[..., -self.hidden_size:].detach()
        self.values = chunk if self.values is None else torch.cat([self.values, chunk], 1)
        self.values = self.values[:, -self.length:]
        self.end = start + chunk.shape[1]

    def read(self):
        if self.values is None:
            raise RuntimeError("verified memory was not populated before proposal")
        pad = self.length - self.values.shape[1]
        memory = F.pad(self.values, (0, 0, pad, 0))
        valid = torch.arange(self.length, device=memory.device)[None, :] >= pad
        return memory, valid.expand(memory.shape[0], -1)


class MemoryResidual(nn.Module):
    def __init__(self, hidden_size, d, state_dim, n_heads, length, dropout):
        super().__init__()
        self.memory_ln = nn.LayerNorm(hidden_size)
        self.memory_in = nn.Linear(hidden_size, d, bias=False)
        self.pos = nn.Parameter(torch.empty(length, d))
        nn.init.normal_(self.pos, std=0.02)
        self.query_ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ff_ln = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        self.read_ln = nn.LayerNorm(d)
        self.h_ln = nn.LayerNorm(hidden_size)
        self.s_ln = nn.LayerNorm(state_dim)
        self.fuse = nn.Linear(d + hidden_size + state_dim, d)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, hidden_size, bias=False)
        self.out._pspr_zero_init = True
        nn.init.zeros_(self.out.weight)

    def read(self, query, memory, valid, attention_mask=None):
        lead, H, d = query.shape[:-2], query.shape[-2], query.shape[-1]
        L = memory.shape[-2]
        valid = valid.bool().reshape(-1, L)
        nonempty = valid.any(-1)
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~nonempty  # avoid all-masked softmax NaNs
        kv = self.memory_in(self.memory_ln(memory.to(self.memory_in.weight.dtype)))
        kv = kv + self.pos[-L:]
        q, kv = self.query_ln(query).reshape(-1, H, d), kv.reshape(-1, L, d)
        read, _ = self.attn(q, kv, kv, key_padding_mask=~safe_valid,
                            attn_mask=attention_mask, need_weights=False)
        read = read + self.ff(self.ff_ln(read))
        read = read * nonempty[:, None, None]  # null-memory has no context contribution
        return read.reshape(*lead, H, d)

    def delta(self, read, hidden, state):
        dtype = self.out.weight.dtype
        x = torch.cat([self.read_ln(read.to(dtype)), self.h_ln(hidden.to(dtype)),
                       self.s_ln(state.to(dtype))], -1)
        return self.out(self.drop(F.silu(self.fuse(x))))


class MemoryCorrector(ClozeCorrector):
    requires_verified_memory = True

    def __init__(self, *, memory_length=64, memory_source="verified",
                 memory_future_causal=False, memory_dropout=0.05, **kwargs):
        super().__init__(**kwargs)
        if not self.use_state or not self.direct_hidden:
            raise ValueError("PSPR-Memory requires full-width h and committed-prefix GRU")
        if memory_source not in {"verified", "draft_future", "blind"}:
            raise ValueError(f"unsupported memory source: {memory_source}")
        if memory_length < 2:
            raise ValueError("memory_length must be >= 2")
        self.memory_length, self.memory_source = int(memory_length), str(memory_source)
        self.memory_future_causal = bool(memory_future_causal)
        self.memory_refiner = MemoryResidual(
            self.hidden_size, self.d, self.state_dim, kwargs.get("n_heads", 8),
            self.memory_length, float(memory_dropout))
        self.freeze_base_runtime = False
        self._config.update(model_type="MemoryCorrector", memory_length=self.memory_length,
                            memory_source=self.memory_source,
                            memory_future_causal=self.memory_future_causal,
                            memory_dropout=float(memory_dropout))

    @classmethod
    def from_reference_config(cls, c):
        return cls(
            hidden_size=c["hidden_dim"], vocab_size=c["vocab_size"], top_k=c["K"],
            d=c["d"], n_layers=c["n_layers"], n_heads=c["n_heads"], state_dim=c["ds"],
            delta_hidden=c["delta_h"], max_slots=c["max_slots"], dropout=c["dropout"],
            direct_hidden=c.get("direct_hidden", True), use_state=c.get("use_state", True),
            bidirectional=c.get("bidirectional", True), err_use_state=c.get("err_use_state", False),
            memory_length=c.get("memory_length", 64), memory_source=c.get("memory_source", "verified"),
            memory_future_causal=c.get("memory_future_causal", False),
            memory_dropout=c.get("memory_dropout", 0.05))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base_runtime:
            # Frozen dropout must also stay in eval; do not train against a noisy
            # mature scorer whose noise disappears at serving time.
            for name, child in self.named_children():
                if name != "memory_refiner":
                    child.train(False)
        return self

    def restore_zero_init_contract(self):
        super().restore_zero_init_contract()
        nn.init.zeros_(self.memory_refiner.out.weight)

    def cloze_states(self, hidden_states, candidate_ids, anchor_ids, log_probs, scalars,
                     *, target_memory=None, memory_mask=None):
        if target_memory is None or memory_mask is None:
            raise RuntimeError("PSPR-Memory needs explicit verified memory and mask")
        if target_memory.shape[:-2] != hidden_states.shape[:-2]:
            raise ValueError("memory and proposal block batch dimensions must match")
        z = super().cloze_states(hidden_states, candidate_ids, anchor_ids, log_probs, scalars)
        # Keep this reader's query slot-local. Feeding z here would let the
        # frozen bidirectional branch relay future seed information around the
        # new reader's causal mask, confounding that control.
        query = self.h_in(self.h_ln(hidden_states.to(self.h_in.weight.dtype)))
        attn_mask = None
        if self.memory_source == "blind":
            # Identical trainable shapes/compute, no target-context information.
            # Positions and query still work, making this a capacity control.
            target_memory, memory_mask = torch.zeros_like(target_memory), torch.ones_like(memory_mask)
        elif self.memory_source == "draft_future":
            H = hidden_states.shape[-2]
            if H >= self.memory_length:
                raise ValueError("draft-future control needs memory_length > proposal slots")
            seed = F.embedding(candidate_ids[..., 0], self._embedding())
            feature = F.layer_norm(hidden_states.float(), (self.hidden_size,)) + F.layer_norm(
                seed.float(), (self.hidden_size,))
            pad = self.memory_length - H
            target_memory = F.pad(feature, (0, 0, pad, 0))
            memory_mask = torch.zeros_like(memory_mask)
            memory_mask[..., 0], memory_mask[..., pad:] = True, True
            query_i = torch.arange(H, device=feature.device)[:, None]
            key_j = torch.arange(self.memory_length, device=feature.device)[None, :] - pad
            # Each query excludes its own concrete seed; a null key is always
            # visible so the causal control's first query is well-defined.
            attn_mask = key_j.eq(query_i)
            if self.memory_future_causal:
                attn_mask = attn_mask | key_j.gt(query_i)
        read = self.memory_refiner.read(query, target_memory, memory_mask, attn_mask)
        return torch.cat([z, read], -1)

    def hidden_delta(self, z, hidden_states=None, state=None):
        if z.shape[-1] != 2 * self.d:
            raise ValueError("PSPR-Memory requires packed [cloze_z, memory_read]")
        old = super().hidden_delta(z[..., :self.d], hidden_states, state)
        return old + self.memory_refiner.delta(z[..., self.d:], hidden_states, state)

    def err_logits(self, z, *args, **kwargs):
        return super().err_logits(z[..., :self.d], *args, **kwargs)

    def score_candidates(self, *, candidate_ids, unary_logits, hidden_states,
                         predecessor_ids, lattice_scalars=None, return_err=False,
                         return_full=False, target_memory=None, memory_mask=None):
        if lattice_scalars is None:
            raise ValueError("PSPR-Memory needs exact lattice uncertainty features")
        embedding = self._embedding()
        candidates = F.embedding(candidate_ids, embedding)
        z = self.cloze_states(hidden_states, candidate_ids, predecessor_ids[..., 0],
                              unary_logits, lattice_scalars,
                              target_memory=target_memory, memory_mask=memory_mask)
        state = self.causal_states(F.embedding(predecessor_ids, embedding))
        scores = self.score(z, hidden_states, candidates, unary_logits, state)
        if not (return_err or return_full):
            return scores
        out = [scores]
        if return_err:
            out.append(self.err_logits(z, candidates[..., 0, :], unary_logits,
                                       lattice_scalars, hidden_states, state))
        if return_full:
            out.append(self.full_vocab_logits(z, hidden_states, state))
        return tuple(out)


@register_draft
class PSPRMemoryDraftModel(PSPRClozeDraftModel):
    warm_start_optional_prefixes = ("candidate_selector.", "candidate_selector.memory_refiner.")

    def _init_draft_head(self, config, dflash_config):
        super()._init_draft_head(config, {**dflash_config, "selector_train_scope": "all"})
        c = self.candidate_selector.reference_config()
        for key, default in (("memory_length", 64), ("memory_source", "verified"),
                             ("memory_future_causal", False), ("memory_dropout", 0.05)):
            c[key] = dflash_config.get(key, default)
        self.candidate_selector = MemoryCorrector.from_reference_config(c)
        self.memory_train_scope = dflash_config.get("memory_train_scope", "adapter")
        if self.memory_train_scope not in {"adapter", "all"}:
            raise ValueError("memory_train_scope must be adapter or all")
        self._verified_memory_cache = None

    def apply_backbone_freeze(self):
        frozen = super().apply_backbone_freeze()
        if self.memory_train_scope == "adapter":
            for name, p in self.candidate_selector.named_parameters():
                if not name.startswith("memory_refiner.") and p.requires_grad:
                    p.requires_grad_(False)
                    frozen += p.numel()
            self.candidate_selector.freeze_base_runtime = True
        return frozen

    def forward(self, position_ids, attention_mask=None, noise_embedding=None,
                target_hidden=None, past_key_values=None, use_cache=False, **kwargs):
        if use_cache:
            if self._verified_memory_cache is None:
                self._verified_memory_cache = VerifiedMemoryCache(
                    self.candidate_selector.memory_length, self.config.hidden_size)
            self._verified_memory_cache.append(target_hidden, position_ids[0, 0].item())
        return super().forward(position_ids=position_ids, attention_mask=attention_mask,
                               noise_embedding=noise_embedding, target_hidden=target_hidden,
                               past_key_values=past_key_values, use_cache=use_cache, **kwargs)

    def _sample_draft_tokens(self, target, draft_hidden, block_output_ids):
        selector = self.candidate_selector
        if selector.target_embedding.numel() == 0:
            self.bind_target_decoder(target.model.embed_tokens)
        hidden = draft_hidden[:, -self.block_size + 1:]
        lp, candidates, scalars = selector.extract_lattice(target.lm_head(hidden))
        embedding = selector._embedding()
        memory, valid = self._verified_memory_cache.read()
        z = selector.cloze_states(hidden, candidates, block_output_ids[:, 0], lp, scalars,
                                  target_memory=memory, memory_mask=valid)
        prev, recurrent, path = block_output_ids[:, 0], None, []
        for i in range(candidates.shape[1]):
            step, recurrent = selector.gru(F.embedding(prev, embedding)[:, None], recurrent)
            state = step[:, 0]
            scores = selector.score(z[:, i], hidden[:, i], F.embedding(candidates[:, i], embedding),
                                     lp[:, i], state)
            if self.selector_decision_mode != "margin_gate":
                raise ValueError("PSPR-Memory currently validates margin_gate serving only")
            err = None
            if self.selector_gate_theta > 0:
                err = selector.err_logits(z[:, i], F.embedding(candidates[:, i, 0], embedding),
                                          lp[:, i], scalars[:, i], hidden[:, i], state)
            force_keep = torch.full(scores.shape[:-1], self.selector_gate_skip_first and i == 0,
                                    dtype=torch.bool, device=scores.device)
            index = selector.select_margin_gate(scores, err_logits=err, rho=self.selector_gate_rho,
                                                 tau=self.selector_gate_tau, theta=self.selector_gate_theta,
                                                 force_keep_mask=force_keep)
            prev = candidates[:, i].gather(-1, index).squeeze(-1)
            path.append(prev)
        return torch.stack(path, 1)


__all__ = ["MemoryCorrector", "PSPRMemoryDraftModel", "VerifiedMemoryCache", "gather_verified_memory"]
