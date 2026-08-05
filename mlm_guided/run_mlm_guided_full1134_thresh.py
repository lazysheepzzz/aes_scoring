#!/usr/bin/env python3
"""Retired launcher for the invalid 1,134-essay MLM experiment.

The historical implementation shared integer token IDs between ModernBERT
and DeBERTa and therefore cannot be used as a formal result.  It is retained
only as a discoverable tombstone so an old command fails loudly instead of
silently recreating invalid output.
"""

raise SystemExit(
    "This historical MLM launcher is retired because it mixed ModernBERT "
    "and DeBERTa token IDs. Use mlm_guided/evaluate_aes_mlm_guided.py for "
    "formal evaluation. See mlm_guided/mlm_guided_summary.md."
)
