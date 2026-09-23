"""Validate V8 reduced spatial Navier-Stokes strategy on prior 12m."""

import sys
from app import ns_spatial_v8_prior_12m_selftest


def main():
    result = ns_spatial_v8_prior_12m_selftest()
    assert result["status"] == "ok"
    assert result["causality"]["lookahead"] is False
    assert result["period"]["days"] == 365
    assert result["period"]["hours"] >= 8500

    print("NS_V8_12M_PRIOR: OK", {
        "period": result["period"],
        "diagnostics": result["diagnostics"],
        "magnitude_tests": result["magnitude_tests"],
        "strategy_probes": result["strategy_probes_linear_short_accounting"],
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"NS_V8_12M_PRIOR FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
