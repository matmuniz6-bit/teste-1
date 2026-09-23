"""Deployment-time integration smoke test.

Runs inside the same image/environment as the FastAPI service.
Never prints secret values.
"""
import json
import sys

from app import (
    NativeBacktestRequest,
    aave_selftest,
    demeter_selftest,
    health,
    long_reversal_selftest,
    native_backtest,
    trading_strategy_selftest,
)


def main():
    h = health()
    assert h["integration"]["all_engines_importable"], json.dumps(h["engines"], default=str)
    assert h["integration"]["trading_strategy_auth"] == "configured"
    print("SMOKE health: OK")
    print("SMOKE engines:", {
        name: {
            "importable": meta.get("importable"),
            "version": meta.get("version"),
        }
        for name, meta in h["engines"].items()
    })

    ts = trading_strategy_selftest()
    assert ts["status"] == "ok"
    assert ts["authenticated"] is True
    assert ts["rows"] > 0
    print("SMOKE trading_strategy: OK", {
        "pair": ts["pair"],
        "bucket": ts["bucket"],
        "rows": ts["rows"],
        "requested_range": ts["requested_range"],
    })

    result = native_backtest(NativeBacktestRequest())
    assert result["status"] == "ok"
    assert result["engine"] == "trade-executor"
    assert result["data_source"] == "Trading Strategy"
    assert result["no_lookahead"] is True
    print("SMOKE native_backtest: OK", {
        "resolved_pair": result["resolved_pair"],
        "dataset": result["dataset"],
        "result": result["result"],
    })

    demeter = demeter_selftest()
    assert demeter["status"] == "ok"
    assert demeter["source"] == "Trading Strategy CLMM"
    assert demeter["clmm_rows"] > 0
    assert demeter["demeter"]["loaded_rows"] > 0
    assert demeter["demeter"]["price_rows"] > 0
    print("SMOKE demeter_clmm: OK", {
        "pair": demeter["pair"],
        "request": demeter["request"],
        "clmm_rows": demeter["clmm_rows"],
        "demeter": demeter["demeter"],
    })

    lr = long_reversal_selftest()
    assert lr["status"] == "ok"
    assert lr["strategy"]["no_lookahead"] is True
    assert lr["coverage"]["complete_trading_dates"] >= 28
    assert set(lr["cost_scenarios"].keys()) == {"0", "1", "2"}
    assert lr["market"]["base"] == "WETH"
    assert lr["market"]["quote"] == "USDC"
    print("SMOKE long_reversal: OK", {
        "coverage": lr["coverage"],
        "benchmark": lr["benchmark"],
        "cost_scenarios": lr["cost_scenarios"],
    })

    aave = aave_selftest()
    assert aave["status"] == "ok"
    assert aave["engine"] == "Demeter AaveV3Market"
    assert aave["protocol"] == "Aave V3"
    required = {"SupplyAction", "BorrowAction", "RepayAction", "WithdrawAction"}
    assert required.issubset(set(aave["actions"]))
    print("SMOKE aave_v3: OK", {
        "period": aave["period"],
        "reserves": aave["reserves"],
        "actions": aave["actions"],
        "data_lineage": aave["data_lineage"],
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
