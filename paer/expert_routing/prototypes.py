"""Detached EMA family prototypes: no retained encoder graphs across batches."""
import torch
from torch import nn
from torch.nn import functional as F


class RoutingPrototypeBank(nn.Module):
    def __init__(self, adapters, experts, momentum=.9, temperature=.1):
        super().__init__()
        if not 0 <= momentum < 1 or temperature <= 0:
            raise ValueError('Invalid prototype momentum or temperature')
        self.momentum, self.temperature = momentum, temperature
        self.register_buffer('prototypes', torch.zeros(adapters, 3, experts))
        self.register_buffer('counts', torch.zeros(3, dtype=torch.long))

    def loss_and_update(self, gates, edit_mask, families):
        if not gates or len(gates) != self.prototypes.shape[0]:
            raise ValueError('Missing expert gate outputs')
        if families.ndim != 1 or families.numel() != edit_mask.shape[0] or bool(((families < 0) | (families > 2)).any()):
            raise ValueError('Expected R/H/I family IDs, Injection subtypes share one ID')
        mass = edit_mask.float().sum(1)
        valid = mass > 0
        zero = sum(gate.sum() for gate in gates) * 0.
        if not bool(valid.any()):
            return zero, {'router_loss': 0., 'router_supervised_essays': 0,
                          'router_reference_families': int((self.counts > 0).sum())}
        distribution = edit_mask.float() / mass[:, None].clamp_min(1.)
        samples = torch.stack([(gate.float() * distribution.unsqueeze(-1)).sum(1)[valid] for gate in gates])
        labels = families[valid]
        refs = self.prototypes.detach().clone()
        available = self.counts > 0
        # Bootstrap unseen families from current detached observations.
        for family in labels.unique().tolist():
            if not bool(available[family]):
                refs[:, family] = samples[:, labels == family].detach().mean(1)
                available[family] = True
        ref_ids = available.nonzero(as_tuple=False).flatten()
        if len(ref_ids) >= 2:
            distances = (samples.unsqueeze(2) - refs[:, ref_ids].unsqueeze(1)).square().sum(-1)
            logits = -distances / self.temperature
            targets = (labels[:, None] == ref_ids[None, :]).long().argmax(1)
            loss = F.cross_entropy(logits.reshape(-1, len(ref_ids)), targets.repeat(len(gates)))
        else:
            loss = zero
        # refs must be cloned before mutation, including for autograd version checks.
        with torch.no_grad():
            for family in labels.unique().tolist():
                current = samples[:, labels == family].detach().mean(1)
                if self.counts[family] == 0:
                    self.prototypes[:, family].copy_(current)
                else:
                    self.prototypes[:, family].lerp_(current, 1. - self.momentum)
                self.counts[family] += int((labels == family).sum())
        return loss, {'router_loss': float(loss.detach()), 'router_supervised_essays': int(valid.sum()),
                      'router_reference_families': len(ref_ids),
                      'router_gate_usage': samples.detach().mean((0, 1)).cpu().tolist()}
