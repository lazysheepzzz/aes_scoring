"""Paired development-set statistics. No model or attack dependencies."""
from __future__ import annotations

import math

import numpy as np


def _resamples(n, count, seed):
    if n < 2 or count < 1:
        raise ValueError("Need at least two pairs and one bootstrap draw")
    rng = np.random.default_rng(seed)
    for start in range(0, count, 256):
        yield rng.integers(0, n, size=(min(256, count - start), n))


def _interval(values):
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None


def bootstrap_reduction(baseline, paer, *, n_bootstrap=10000, seed=42):
    """Compatibility helper for already-aligned Bernoulli success arrays."""
    baseline, paer = np.asarray(baseline, dtype=float), np.asarray(paer, dtype=float)
    if baseline.ndim != 1 or baseline.shape != paer.shape:
        raise ValueError("Unaligned paired shapes")
    if not all(np.isin(x, [0, 1]).all() for x in (baseline, paer)):
        raise ValueError("Expected Bernoulli success arrays")
    d = baseline - paer
    draws = np.concatenate([d[idx].mean(axis=1) for idx in _resamples(len(d), n_bootstrap, seed)])
    # Historical field: centered-bootstrap tail estimate, not an exact test.
    # The main analysis uses exact McNemar for individual attack comparisons.
    centered = np.abs(draws - d.mean())
    return {"baseline_asr": float(baseline.mean()), "paer_asr": float(paer.mean()),
            "absolute_asr_reduction": float(d.mean()), "ci95": _interval(draws),
            "bootstrap_probability_reduction_gt_0": float(np.mean(draws > 0)),
            "two_sided_bootstrap_p": (1 + int(np.sum(centered >= abs(d.mean())))) / (len(draws) + 1)}


def _qwk(labels, predictions):
    """Quadratic kappa with the fixed six-category scale (not rank compression)."""
    y, p = np.clip(np.rint(labels), 0, 5), np.clip(np.rint(predictions), 0, 5)
    expected = y.var(axis=-1) + p.var(axis=-1) + (y.mean(axis=-1) - p.mean(axis=-1)) ** 2
    observed = ((y - p) ** 2).mean(axis=-1)
    return 1 - np.divide(observed, expected, out=np.full_like(expected, np.nan), where=expected > 0)


def _clean(labels, predictions):
    errors = predictions - labels
    return {"qwk": _qwk(labels, predictions), "mae": np.abs(errors).mean(axis=-1),
            "rmse": (errors ** 2).mean(axis=-1) ** 0.5, "bias": errors.mean(axis=-1),
            "prediction_sd": predictions.std(axis=-1)}


def paired_statistics(labels, clean, deltas, *, threshold, n_bootstrap, seed):
    """Inputs aligned by verified essay ID; reuse every draw for all outcomes.

    deltas maps attack name to (Mixed, PAER); labels already use 0–5 space.
    The two Injection subtypes each have half a family weight.
    """
    labels = np.asarray(labels, dtype=float)
    clean = np.asarray(clean, dtype=float)
    names = ("rudimentary", "hotflip", "injection_external", "injection_self_dup")
    attack_arrays = {name: np.asarray(value, dtype=float) for name, value in deltas.items()}
    if clean.shape != (2, len(labels)) or any(a.shape != clean.shape for a in attack_arrays.values()):
        raise ValueError("All model/attack vectors must share one essay index")
    if not all(np.isfinite(a).all() for a in [labels, clean, *attack_arrays.values()]):
        raise ValueError("Nonfinite input to paired analysis")
    success = {k: a >= threshold for k, a in attack_arrays.items()}
    macro = lambda arrays: (arrays[names[0]] + arrays[names[1]] +
                             (arrays[names[2]] + arrays[names[3]]) / 2) / 3
    outcomes = {**{f"{k}.asr": a.astype(float) for k, a in success.items()},
                **{f"{k}.mean_delta": a for k, a in attack_arrays.items()},
                "rhi_macro.asr": macro({k: a.astype(float) for k, a in success.items()}),
                "rhi_macro.mean_delta": macro(attack_arrays)}
    point = {k: a.mean(axis=1) for k, a in outcomes.items()}
    point.update({f"clean.{k}": a for k, a in _clean(labels, clean).items()})
    samples = {k: [] for k in point}
    for idx in _resamples(len(labels), n_bootstrap, seed):
        for k, a in outcomes.items():
            samples[k].append(a[:, idx].mean(axis=-1))
        for k, a in _clean(labels[idx], clean[:, idx]).items():
            samples[f"clean.{k}"].append(a)
    comparisons = {}
    for k, value in point.items():
        draws = np.concatenate(samples[k], axis=1)
        difference = draws[1] - draws[0]
        comparisons[k] = {"mixed": float(value[0]) if np.isfinite(value[0]) else None,
                          "paer": float(value[1]) if np.isfinite(value[1]) else None,
                          "paer_minus_mixed": float(value[1] - value[0]) if np.isfinite(value).all() else None,
                          "difference_ci95": _interval(difference),
                          "valid_draws": int(np.isfinite(difference).sum())}
    tests = {}
    for name, (m, p) in success.items():
        b, c = int((m & ~p).sum()), int((~m & p).sum())
        n = b + c
        p_value = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (1 << n))
        tests[name] = {"mixed_only_success": b, "paer_only_success": c, "exact_p": p_value}
    adjusted = 0.0
    for i, name in enumerate(sorted(tests, key=lambda k: tests[k]["exact_p"])):
        adjusted = max(adjusted, min(1., (len(tests) - i) * tests[name]["exact_p"]))
        tests[name]["holm_p"] = adjusted
    strata = []
    for label in np.unique(labels):
        keep = labels == label
        strata.append({"true_score": int(label + 1), "n": int(keep.sum()),
                       "means": {k: a[:, keep].mean(axis=1).tolist() for k, a in outcomes.items()}})
    return {"n": len(labels), "seed": seed, "n_bootstrap": n_bootstrap,
            "unit": "essay; fixed development models and realized attacks",
            "contrasts": comparisons, "mcnemar": tests, "score_strata": strata}
