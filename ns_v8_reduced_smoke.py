"""Focused 31-day validation for the new reduced spatial V8."""
import sys
from app import ns_criticality_august_selftest

def main():
    result = ns_criticality_august_selftest()
    assert result["status"] == "ok"
    assert result["causality"]["lookahead"] is False
    v8 = result["v8_spatial_reduced_model"]
    print("NS_V8_REDUCED_SMOKE: OK", {
        "period_start": result["date"],
        "result": result["result"],
        "v2": result["instability_v2"],
        "v3": result["v3_event_momentum_3h"],
        "v8": v8,
    })

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"NS_V8_REDUCED_SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
