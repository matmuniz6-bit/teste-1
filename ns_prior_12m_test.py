"""Focused prior 12-month validation for the N-S V4 detector."""

import sys
from app import ns_instability_v4_prior_12m_selftest


def main():
    result = ns_instability_v4_prior_12m_selftest()
    assert result["status"] == "ok"
    assert result["causality"]["lookahead"] is False
    assert result["result"]["requested_days"] == 365
    assert result["result"]["complete_hours"] >= 8500
    assert result["instability_v4_spatial"]["evaluated_hours"] >= 8500

    print("NS_V4_12M_PRIOR: OK", {
        "period_start": result["date"],
        "result": result["result"],
        "v2": result["instability_v2"],
        "v3": result["v3_event_momentum_3h"],
        "v4_detector": result["instability_v4_spatial"],
        "v4_raw_strategy": result["v4_event_momentum_3h"],
        "v4_confirmed_strategy": result["v4_confirmed_momentum_3h"],
        "monthly": result["monthly"],
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"NS_V4_12M_PRIOR FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
