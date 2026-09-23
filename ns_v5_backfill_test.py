"""Backfill V5 across the three already-inspected annual windows."""

import sys
from app import (
    ns_instability_12m_selftest,
    ns_instability_v4_prior_12m_selftest,
    ns_instability_older_12m_selftest,
)


def summarize(label, result):
    print(label, {
        "period_start": result["date"],
        "v2": result["instability_v2"],
        "v3": result["v3_event_momentum_3h"],
        "v5": result["v5_adaptive_direction"],
        "momentum_gates": result["v3_momentum_gate_research"],
    })


def main():
    summarize("V5_CURRENT", ns_instability_12m_selftest())
    summarize("V5_PRIOR", ns_instability_v4_prior_12m_selftest())
    summarize("V5_OLDER", ns_instability_older_12m_selftest())
    print("NS_V5_BACKFILL: OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"NS_V5_BACKFILL FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
