"""Strict V5 validation on two untouched annual windows."""

import sys
from app import (
    ns_instability_older2_12m_selftest,
    ns_instability_oldest_12m_selftest,
)


def summarize(label, result):
    assert result["status"] == "ok"
    assert result["causality"]["lookahead"] is False
    print(label, {
        "period_start": result["date"],
        "v2": result["instability_v2"],
        "v3": result["v3_event_momentum_3h"],
        "v5": result["v5_adaptive_direction"],
        "momentum_gates": result["v3_momentum_gate_research"],
    })


def main():
    summarize("V5_HOLDOUT_2022", ns_instability_older2_12m_selftest())
    summarize("V5_HOLDOUT_2021", ns_instability_oldest_12m_selftest())
    print("NS_V5_HOLDOUT: OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"NS_V5_HOLDOUT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
