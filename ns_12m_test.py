"""Focused 12-month validation for the N-S instability detector."""

import sys

from app import ns_instability_12m_selftest


def main():
    result = ns_instability_12m_selftest()
    assert result["status"] == "ok"
    assert result["causality"]["lookahead"] is False
    assert result["result"]["requested_days"] == 365
    assert result["result"]["complete_hours"] >= 8500
    assert result["instability_v2"]["evaluated_hours"] >= 8500

    print("NS_12M: OK", {
        "period_start": result["date"],
        "result": result["result"],
        "instability_v2": result["instability_v2"],
        "v2_momentum": result["stress_gated_direction_probes"]["momentum"],
        "v3_event_momentum_3h": result["v3_event_momentum_3h"],
        "monthly": result["monthly"],
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"NS_12M FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
