"""Opt-in expert trainer reusing the existing sequential clean/attack backprop."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from paer.aes_rh_trainer import (RHTrainer, RHTrainingConfig, RHTraceCollator,
                                PairedEssayTrainingDataset, changed_token_uplift_targets)
from paer.rhi_experiment_utils import read_json, sha256, checkpoint_identity
from paer.expert_routing.model import AESLowRankExpertScorer
from paer.expert_routing.prototypes import RoutingPrototypeBank


@dataclass
class ExpertTrainingConfig(RHTrainingConfig):
    expert_use_paer: bool = True
    expert_rank: int = 8
    num_experts: int = 3
    expert_layers: int = 2
    expert_learning_rate: float = 1e-4
    expert_route_weight: float = .1
    prototype_momentum: float = .9
    prototype_temperature: float = .1
    routing_scope: str = 'edited'
    selected_train_row_indices: list[int] | None = None
    expert_smoke: bool = False

    def __post_init__(self):
        super().__post_init__()
        expected = 'paer_rh_v3' if self.expert_use_paer else 'mixed_at_rh'
        if self.training_mode != expected or self.routing_scope not in ('edited', 'all'):
            raise ValueError('Expert backend/scope mismatch')
        if (self.expert_rank <= 0 or self.expert_layers <= 0 or not 2 <= self.num_experts <= 8
                or not 0 <= self.prototype_momentum < 1
                or not all(math.isfinite(v) and v > 0 for v in
                           (self.expert_learning_rate, self.prototype_temperature))
                or not math.isfinite(self.expert_route_weight) or self.expert_route_weight < 0):
            raise ValueError('Invalid expert configuration')
        if self.selected_train_row_indices is not None and not self.expert_smoke:
            raise ValueError('Selected rows are for explicit smoke tests only')


class ExpertTraceCollator(RHTraceCollator):
    def __call__(self, examples):
        batch = super().__call__(examples)
        # Binary editing positions are independent of the signed cumulative score gain.
        edits = torch.zeros_like(batch['adversarial_attention_mask'], dtype=torch.float32)
        for i, original_index in enumerate(batch['adversarial_indices'].tolist()):
            n_a = int(batch['clean_attention_mask'][original_index].sum())
            n_b = int(batch['adversarial_attention_mask'][i].sum())
            targets = changed_token_uplift_targets(
                batch['clean_input_ids'][original_index, :n_a].tolist(),
                batch['adversarial_input_ids'][i, :n_b].tolist(),
                special_ids=self.special_ids, step_gain=1., gain_scale=1.)
            edits[i, :n_b] = torch.tensor(targets)
        families = {'rudimentary': 0, 'hotflip': 1, 'injection_external': 2, 'injection_self_dup': 2}
        batch['expert_edit_mask'] = edits * batch['localization_mask']
        batch['expert_families'] = torch.tensor([families[name] for name in batch['attacks']], dtype=torch.long)
        return batch


class ExpertTrainer(RHTrainer):
    def __init__(self, config):
        # Recheck recorded launcher inputs BEFORE loading a GPU model.
        inputs = read_json(Path(config.output_dir) / 'rhi_training_inputs.json')
        if (inputs['train_sha256'] != sha256(config.train_csv)
                or inputs['valid_sha256'] != sha256(config.valid_csv)
                or inputs['trace_sha256'] != sha256(config.trace_jsonl)
                or inputs['checkpoint'] != checkpoint_identity(config.checkpoint_path)
                or inputs['selected_train_row_indices'] != config.selected_train_row_indices):
            raise ValueError('Expert launcher inputs changed; refusing training')
        if any(digest != sha256(ROOT / name) for name, digest in inputs['source_sha256'].items()):
            raise ValueError('Expert implementation changed after launch; use a NEW experiment directory')
        def factory(cfg):
            return AESLowRankExpertScorer.from_base_checkpoint(cfg.checkpoint_path,
                use_paer=cfg.expert_use_paer, expert_rank=cfg.expert_rank,
                num_experts=cfg.num_experts, expert_layers=cfg.expert_layers,
                correction_scale=cfg.correction_scale, risk_bias_init=cfg.routing_risk_bias_init,
                max_token_evidence=cfg.max_token_evidence)
        super().__init__(config, model_factory=factory)
        self.prototype_bank = RoutingPrototypeBank(len(self.model.expert_paths), config.num_experts,
            momentum=config.prototype_momentum, temperature=config.prototype_temperature).to(self.device)
        self._route_metrics = {}
        report = {'total_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                  'expert_and_router_parameters': sum(p.numel() for p in self.model.expert_parameters()),
                  'expert_projection_paths': self.model.expert_paths,
                  'backbone_frozen': False, 'one_encoder_pass_per_branch': True,
                  'cross_microbatch_state_contains_encoder_graphs': False,
                  'gradient_checkpointing_supported': False,
                  'routing_scope': config.routing_scope, 'not_full_RoME_reproduction': True}
        Path(config.output_dir, 'expert_architecture.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('[EXPERT] ' + json.dumps(report), flush=True)

    def build_training_dataset(self):
        dataset = PairedEssayTrainingDataset(self.config.train_csv, self.trace_dataset,
                                             label_offset=self.config.label_offset)
        indices = self.config.selected_train_row_indices
        if indices is not None:
            if len(indices) != len(set(indices)) or any(type(i) is not int or i < 0 or i >= len(dataset) for i in indices):
                raise ValueError('Invalid smoke row indices')
            wanted = set(indices)
            dataset.items = [row for row in dataset.items if row['row_index'] in wanted]
            dataset.traces_by_row = {i: records for i, records in dataset.traces_by_row.items() if i in wanted}
        return dataset

    def build_trace_collator(self):
        return ExpertTraceCollator(self.tokenizer, self.config.max_length,
                                   self.config.label_offset, self.config.attribution_gain_scale)

    def optimizer_groups(self):
        experts = {id(p) for p in self.model.expert_parameters()}
        heads = {id(p) for head in (self.model.risk_head, self.model.token_evidence_head, self.model.attention_head)
                 for p in head.parameters()}
        buckets = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            role = 'expert' if id(parameter) in experts else 'head' if id(parameter) in heads else 'backbone'
            decay = parameter.ndim > 1 and not name.endswith('bias')
            buckets.setdefault((role, decay), []).append(parameter)
        lrs = {'expert': self.config.expert_learning_rate, 'head': self.config.paer_head_learning_rate,
               'backbone': self.config.learning_rate}
        return [{'params': parameters, 'lr': lrs[role],
                 'weight_decay': self.config.weight_decay if decay else 0.}
                for (role, decay), parameters in buckets.items()]

    def begin_extension_step(self):
        self._route_metrics = {'router_loss': 0., 'router_supervised_essays': 0,
                              'router_reference_families': int((self.prototype_bank.counts > 0).sum())}

    def additional_adversarial_objective(self, output, batch):
        mask = batch['expert_edit_mask'] if self.config.routing_scope == 'edited' else batch['localization_mask']
        loss, self._route_metrics = self.prototype_bank.loss_and_update(
            output.expert_gates, mask, batch['expert_families'])
        return self.config.expert_route_weight * loss

    def extension_metrics(self):
        metrics = dict(self._route_metrics)
        if self.device == 'cuda':
            metrics.update(cuda_allocated_mib=torch.cuda.memory_allocated() / 2**20,
                           cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                           cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        return metrics

    def save(self, tag):
        super().save(tag)
        torch.save(self.prototype_bank.state_dict(), Path(self.config.output_dir) / tag / 'paer_expert_training_state.pt')

    def train(self):
        result = super().train()
        report = self.extension_metrics()
        report.update(device=self.device, prototype_family_observations=self.prototype_bank.counts.cpu().tolist(),
                      smoke_only=self.config.expert_smoke)
        Path(self.config.output_dir, 'expert_resource_metrics.json').write_text(
            json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('[EXPERT RESOURCE] ' + json.dumps(report), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = ExpertTrainingConfig(**read_json(args.config))
    ExpertTrainer(config).train()


if __name__ == '__main__':
    main()
