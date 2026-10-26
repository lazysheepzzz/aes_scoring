"""RoME-inspired last-layer V/O experts for masked text, not a vision reproduction."""
from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from paer.modeling_paer_v3 import PAERV3ForEssayScoring, PAERV3Output

MODEL_TYPE = 'aes_low_rank_experts_v1'
EXPERT_STATE = 'paer_experts.pt'
_CONTEXT = ContextVar('aes_expert_forward_context', default=None)


@dataclass
class ExpertOutput(PAERV3Output):
    expert_gates: tuple[torch.Tensor, ...] | None = None


class LowRankExpertLinear(nn.Module):
    """Shared projection plus dense, differentiable low-rank expert residuals."""
    def __init__(self, base, *, rank, experts):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0 or experts < 2:
            raise ValueError('Need a linear projection, positive rank and at least two experts')
        self.base = base
        self.down = nn.ModuleList(nn.Linear(base.in_features, rank, bias=False) for _ in range(experts))
        self.up = nn.ModuleList(nn.Linear(rank, base.out_features, bias=False) for _ in range(experts))
        self.local_gate = nn.Linear(base.in_features, experts)
        self.global_gate = nn.Linear(base.in_features, experts)
        for down, up in zip(self.down, self.up):
            nn.init.normal_(down.weight, std=.01)
            nn.init.zeros_(up.weight)
        self.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x):
        context = _CONTEXT.get()
        if context is None:
            raise RuntimeError('Expert projections must be called through the full AES expert model')
        mask, records = context
        if x.ndim != 3 or x.shape[:2] != mask.shape:
            raise ValueError('Text mask/projection shape mismatch')
        valid = mask.to(device=x.device, dtype=x.dtype)
        pooled = (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        local = self.local_gate(x).float().softmax(-1)
        global_weights = self.global_gate(pooled).float().softmax(-1).unsqueeze(1)
        gates = .5 * (local + global_weights)
        records.append(gates)
        output = self.base(x)
        # Avoid stacking K full hidden tensors or promoting the residual to FP32.
        for index, (down, up) in enumerate(zip(self.down, self.up)):
            delta = up(down(x))
            output = output + delta * gates[..., index:index + 1].to(delta.dtype)
        return output


class AESLowRankExpertScorer(PAERV3ForEssayScoring):
    def __init__(self, base_model, *, use_paer=True, expert_rank=8, num_experts=3,
                 expert_layers=2, correction_scale=1., risk_bias_init=-2., max_token_evidence=.5):
        super().__init__(base_model, correction_scale=correction_scale,
                         risk_bias_init=risk_bias_init, max_token_evidence=max_token_evidence)
        reference = next(base_model.parameters())
        for head in (self.risk_head, self.token_evidence_head, self.attention_head):
            head.to(device=reference.device, dtype=reference.dtype)
        layers = self.base_model.deberta.encoder.layer
        if expert_layers <= 0 or expert_layers > len(layers):
            raise ValueError('expert_layers must fit the existing encoder')
        self.use_paer = bool(use_paer)
        self.expert_rank, self.num_experts, self.expert_layers = expert_rank, num_experts, expert_layers
        self.expert_paths = []
        for index in range(len(layers) - expert_layers, len(layers)):
            # Q is also used for relative-position embeddings in DeBERTa-v3.
            # V and attention-output projections only receive actual token states.
            for suffix in ('attention.self.value_proj', 'attention.output.dense'):
                parent_path, projection = suffix.rsplit('.', 1)
                parent = layers[index].get_submodule(parent_path)
                base = getattr(parent, projection)
                if not isinstance(base, nn.Linear):
                    raise TypeError('Expert v1 requires DeBERTa-v2/v3 V/O projections')
                setattr(parent, projection, LowRankExpertLinear(base, rank=expert_rank, experts=num_experts))
                self.expert_paths.append(f'deberta.encoder.layer.{index}.{suffix}')
        if not self.use_paer:
            for head in (self.risk_head, self.token_evidence_head, self.attention_head):
                head.requires_grad_(False)

    def adapters(self):
        return [self.base_model.get_submodule(path) for path in self.expert_paths]

    def expert_parameters(self):
        for adapter in self.adapters():
            for name, parameter in adapter.named_parameters():
                if not name.startswith('base.'):
                    yield parameter

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                token_type_ids=None, position_ids=None, output_hidden_states=False, return_dict=True, **kwargs):
        if getattr(self.base_model.deberta.encoder, 'gradient_checkpointing', False):
            raise RuntimeError('Expert v1 does not support encoder gradient checkpointing/context recomputation')
        source = input_ids if input_ids is not None else inputs_embeds
        if source is None:
            raise ValueError('input_ids or inputs_embeds required')
        if attention_mask is None:
            attention_mask = torch.ones(source.shape[:2], device=source.device, dtype=torch.long)
        records = []
        token = _CONTEXT.set((attention_mask, records))
        try:
            if self.use_paer:
                output = super().forward(input_ids=input_ids, attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, position_ids=position_ids,
                    output_hidden_states=output_hidden_states, return_dict=True)
                result = ExpertOutput(**dict(output), expert_gates=tuple(records))
            else:
                states, logits = self._encode_and_score(input_ids=input_ids, attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, position_ids=position_ids)
                result = ExpertOutput(logits=logits, base_logits=logits, global_logits=logits,
                    correction=torch.zeros(logits.shape[0], device=logits.device),
                    content_mask=self._content_mask(attention_mask), expert_gates=tuple(records),
                    hidden_states=(states,) if output_hidden_states else None)
        finally:
            _CONTEXT.reset(token)
        return result if return_dict else (result.logits, result.base_logits, result.correction)

    @classmethod
    def from_base_checkpoint(cls, checkpoint_path, *, dtype=torch.float32, **kwargs):
        if (Path(checkpoint_path) / 'paer_config.json').exists():
            raise ValueError('New expert training must start from shared B0, not a defended checkpoint')
        config = AutoConfig.from_pretrained(str(checkpoint_path), trust_remote_code=True)
        if config.model_type != 'deberta-v2':
            raise ValueError('Expert v1 supports DeBERTa-v2/v3 checkpoints only')
        config.num_labels = 1
        base = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_path), config=config,
                                                                  torch_dtype=dtype, trust_remote_code=True)
        return cls(base, **kwargs)

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save normal HF backbone keys, never silently discard wrapped projection weights.
        backbone = {}
        for name, value in self.base_model.state_dict().items():
            for path in self.expert_paths:
                if name.startswith(path + '.'):
                    suffix = name[len(path) + 1:]
                    name = path + '.' + suffix[5:] if suffix.startswith('base.') else None
                    break
            if name is not None:
                backbone[name] = value
        self.base_model.save_pretrained(output_dir, state_dict=backbone)
        state = {'adapters': [{k: v for k, v in module.state_dict().items() if not k.startswith('base.')}
                              for module in self.adapters()],
                 'heads': {name: getattr(self, name).state_dict() for name in
                           ('risk_head', 'token_evidence_head', 'attention_head')} if self.use_paer else {}}
        torch.save(state, output_dir / EXPERT_STATE)
        config = {'model_type': MODEL_TYPE, 'version': 1, 'use_paer': self.use_paer,
                  'expert_rank': self.expert_rank, 'num_experts': self.num_experts,
                  'expert_layers': self.expert_layers, 'expert_projections': ['attention.self.value_proj', 'attention.output.dense'],
                  'gate_mix': 'equal_local_and_masked_global', 'correction_scale': self.correction_scale,
                  'max_token_evidence': self.max_token_evidence, 'adaptation_not_full_RoME_reproduction': True}
        (output_dir / 'paer_config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')

    @classmethod
    def from_pretrained(cls, checkpoint_path, *, dtype=torch.float32):
        checkpoint_path = Path(checkpoint_path)
        config = json.loads((checkpoint_path / 'paer_config.json').read_text(encoding='utf-8'))
        if config.pop('model_type') != MODEL_TYPE or config.pop('version') != 1:
            raise ValueError('Not an expert v1 checkpoint')
        for name in ('expert_projections', 'gate_mix', 'adaptation_not_full_RoME_reproduction'):
            config.pop(name)
        base_config = AutoConfig.from_pretrained(str(checkpoint_path), trust_remote_code=True)
        base = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_path), config=base_config,
                                                                  torch_dtype=dtype, trust_remote_code=True)
        model = cls(base, **config)
        state = torch.load(checkpoint_path / EXPERT_STATE, map_location='cpu', weights_only=True)
        if len(state['adapters']) != len(model.expert_paths):
            raise ValueError('Expert adapter count mismatch')
        for module, saved in zip(model.adapters(), state['adapters']):
            incompat = module.load_state_dict(saved, strict=False)
            if set(incompat.missing_keys) != {'base.weight', 'base.bias'} or incompat.unexpected_keys:
                raise ValueError('Expert adapter state missing or unexpected keys')
        for name, saved in state['heads'].items():
            getattr(model, name).load_state_dict(saved)
        if set(state['heads']) != ({'risk_head', 'token_evidence_head', 'attention_head'} if model.use_paer else set()):
            raise ValueError('Expert PAER head state mismatch')
        return model
