"""Single-window N-S momentum research validation."""

import sys
from app import ns_instability_oldest_12m_selftest


def main():
    result = ns_instability_oldest_12m_selftest()
    assert result["status"] == "ok"
    assert result["causality"]["lookahead"] is False
    assert result["result"]["requested_days"] == 365
    assert result["result"]["complete_hours"] >= 8500

    print("NS_RESEARCH_12M_OLDEST: OK", {
        "period_start": result["date"],
        "result": result["result"],
        "v2": result["instability_v2"],
        "v3": result["v3_event_momentum_3h"],
        "v5": result["v5_adaptive_direction"],
        "v6": result["v6_horizon_research"],
        "momentum_gates": result["v3_momentum_gate_research"],
        "v4_detector": result["instability_v4_spatial"],
        "v4_raw": result["v4_event_momentum_3h"],
        "v4_confirmed": result["v4_confirmed_momentum_3h"],
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"NS_RESEARCH_12M_OLDEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
