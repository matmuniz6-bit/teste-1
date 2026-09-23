"""Focused prior 12-month holdout validation for N-S V3/V4 research."""

import sys
from app import ns_instability_v4_prior_12m_selftest


def main():
    result = ns_instability_v4_prior_12m_selftest()
    assert result["status"] == "ok"
    assert result["causality"]["lookahead"] is False
    assert result["result"]["requested_days"] == 365
    assert result["result"]["complete_hours"] >= 8500
    assert result["instability_v2"]["evaluated_hours"] >= 8500

    print("NS_RESEARCH_12M_PRIOR: OK", {
        "period_start": result["date"],
        "result": result["result"],
        "v2": result["instability_v2"],
        "v3": result["v3_event_momentum_3h"],
        "momentum_gates": result["v3_momentum_gate_research"],
        "v4_detector": result["instability_v4_spatial"],
        "v4_raw": result["v4_event_momentum_3h"],
        "v4_confirmed": result["v4_confirmed_momentum_3h"],
        "monthly": result["monthly"],
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"NS_RESEARCH_12M_PRIOR FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
