"""Fast 31-day smoke test for the primary spatial V8 strategy."""
import sys
from app import ns_spatial_v8, SpatialNSV8Request

def main():
    result = ns_spatial_v8(
        SpatialNSV8Request(
            date="2026-08-01",
            days=31,
            horizon_hours=3,
            spatial_bins=7,
            illustrative_cost_bps_per_turnover=5.0,
        )
    )
    assert result["status"] == "ok"
    assert result["causality"]["lookahead"] is False
    print("NS_V8_PRIMARY_SMOKE: OK", {
        "period": result["period"],
        "diagnostics": result["diagnostics"],
        "magnitude_tests": result["magnitude_tests"],
        "strategy_probes": result["strategy_probes_linear_short_accounting"],
        "spatial_field": result["spatial_field"],
        "structure": result["structure"],
    })

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"NS_V8_PRIMARY_SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
