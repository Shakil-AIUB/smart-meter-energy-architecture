"""
Evaluation harness for the Building Energy Optimisation project.

Answers the research question from the guideline: "where should the
control decision run, and what's the trade-off between energy savings,
comfort, latency, and resilience under fault?"

Produces five measured results, each tied to a rubric line item:

1. ENERGY SAVINGS       — AggressiveSavingsStrategy vs a naive static
                           schedule baseline, on the same real sensor trace.
2. COMFORT COMPLIANCE   — % of ticks indoor temp stays within comfort band,
                           for each strategy.
3. CONTROL-LOOP LATENCY — edge-only decision time vs a simulated cloud
                           round-trip, measured directly in wall-clock time.
4. RESILIENCE           — comfort compliance & override count when the
                           cloud link is simulated as unavailable for a
                           window, comparing edge-fallback vs naive no-fallback.
5. SCALABILITY          — ingestion/processing throughput as the number of
                           simulated concurrent meters grows.

Usage:
    python evaluation.py --data real_building_energy.csv --out-dir results/
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from control_loop import (
    ReadingContext, EdgeGateway,
    ComfortFirstStrategy, AggressiveSavingsStrategy, OfflineFallbackStrategy,
)


# Nominal power draw per actuator action (kW) — used to ESTIMATE energy
# consumption under different strategies from the same real sensor trace,
# since we don't have a live building to actually re-run under each policy.
# Values are set relative to the real hvac/lighting readings' typical scale.
ACTION_POWER_KW = {
    "hold": 0.0,
    "heat_on": 3.0,
    "cool_on": 3.0,
    "emergency_heat": 4.5,
    "emergency_cool": 4.5,
    "no-op": 0.0,
}


class ThermalModel:
    """
    Minimal closed-loop thermal simulation. Without this, replaying the
    real dataset's indoor_temp_c column verbatim makes every strategy
    look identical -- the historical temperature already reflects
    whatever control existed when the data was recorded, not the
    strategy being tested. This model lets a chosen ACTION actually
    change the simulated indoor temperature going forward, so different
    strategies produce genuinely different comfort/energy outcomes.

    Real occupancy and outdoor temperature from the dataset still drive
    the simulation (exogenous inputs); only indoor temperature becomes
    a simulated, strategy-dependent state.
    """

    def __init__(self, start_temp: float, outdoor_fallback: float = 30.0):
        self.temp = start_temp
        self.outdoor_fallback = outdoor_fallback  # used only if outdoor_temp_c is missing

    def step(self, action: str, occupancy: float, outdoor_temp_c: float) -> float:
        outdoor = outdoor_temp_c if not np.isnan(outdoor_temp_c) else self.outdoor_fallback
        drift = 0.01 * (outdoor - self.temp)          # slow leak toward outdoor
        occ_heat = 0.15 * occupancy                     # occupants add heat
        action_effect = {
            "heat_on": 0.35, "emergency_heat": 0.6,
            "cool_on": -0.35, "emergency_cool": -0.6,
            "hold": 0.0, "no-op": 0.0,
        }.get(action, 0.0)
        noise = np.random.normal(0, 0.03)
        self.temp = self.temp + drift + occ_heat + action_effect + noise
        return self.temp


def run_strategy_closed_loop(contexts: list[ReadingContext], strategy, start_temp: float) -> pd.DataFrame:
    """
    Like run_strategy, but feeds the ThermalModel's simulated temperature
    back into the next tick's context instead of replaying the historical
    value -- this is what makes strategies actually diverge.
    """
    gateway = EdgeGateway(strategy=strategy)
    thermal = ThermalModel(start_temp=start_temp)
    logs = []
    sim_temp = start_temp
    for ctx in contexts:
        live_ctx = ReadingContext(
            timestamp=ctx.timestamp, occupancy=ctx.occupancy,
            indoor_temp_c=sim_temp, outdoor_temp_c=ctx.outdoor_temp_c,
        )
        result = gateway.tick(live_ctx)
        logs.append(result)
        sim_temp = thermal.step(result["action"], ctx.occupancy, ctx.outdoor_temp_c)
    return pd.DataFrame(logs)


def rows_to_contexts(df: pd.DataFrame) -> list[ReadingContext]:
    contexts = []
    for _, row in df.iterrows():
        contexts.append(ReadingContext(
            timestamp=str(row["timestamp"]),
            occupancy=float(row["occupancy"]),
            indoor_temp_c=float(row["indoor_temp_c"]),
            outdoor_temp_c=float(row.get("outdoor_temp_c", np.nan)) if not pd.isna(row.get("outdoor_temp_c", np.nan)) else 0.0,
        ))
    return contexts


def run_strategy(contexts: list[ReadingContext], strategy) -> pd.DataFrame:
    gateway = EdgeGateway(strategy=strategy)
    logs = [gateway.tick(ctx) for ctx in contexts]
    return pd.DataFrame(logs)


class StaticScheduleStrategy:
    """
    Naive baseline: always targets the exact centre of the comfort band,
    all day, regardless of occupancy. Represents "no smart scheduling at
    all" — the thing a smart system should beat.
    """
    name = "static_schedule"

    def decide_target(self, ctx: ReadingContext) -> float:
        return (ctx.comfort_low + ctx.comfort_high) / 2.0


def estimate_energy_kwh(log_df: pd.DataFrame, interval_min: int) -> float:
    power = log_df["action"].map(ACTION_POWER_KW).fillna(0.0)
    hours_per_tick = interval_min / 60.0
    return float((power * hours_per_tick).sum())


def comfort_compliance_pct(log_df: pd.DataFrame, comfort_low=24.0, comfort_high=26.0) -> float:
    in_band = log_df["indoor_temp_c"].between(comfort_low, comfort_high)
    return round(100 * in_band.mean(), 2)


def eval_energy_and_comfort(contexts: list[ReadingContext], interval_min: int, out_dir: str) -> pd.DataFrame:
    strategies = {
        "Static schedule (baseline)": StaticScheduleStrategy(),
        "Comfort-first": ComfortFirstStrategy(),
        "Aggressive savings": AggressiveSavingsStrategy(),
        "Offline fallback": OfflineFallbackStrategy(),
    }
    start_temp = contexts[0].indoor_temp_c

    rows = []
    for label, strat in strategies.items():
        log_df = run_strategy_closed_loop(contexts, strat, start_temp)
        energy_kwh = estimate_energy_kwh(log_df, interval_min)
        compliance = comfort_compliance_pct(log_df)
        overrides = int((log_df["state"] == "override").sum())
        rows.append({
            "strategy": label,
            "estimated_energy_kwh": round(energy_kwh, 1),
            "comfort_compliance_pct": compliance,
            "safety_overrides": overrides,
        })

    result = pd.DataFrame(rows)
    baseline_kwh = result.loc[result["strategy"] == "Static schedule (baseline)", "estimated_energy_kwh"].iloc[0]
    result["savings_vs_baseline_pct"] = round(100 * (baseline_kwh - result["estimated_energy_kwh"]) / baseline_kwh, 1)

    result.to_csv(os.path.join(out_dir, "energy_comfort_results.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(result["strategy"], result["estimated_energy_kwh"], color="#4C72B0")
    ax[0].set_ylabel("Estimated energy (kWh)")
    ax[0].set_title("Energy use by strategy")
    ax[0].tick_params(axis="x", rotation=30)

    ax[1].bar(result["strategy"], result["comfort_compliance_pct"], color="#55A868")
    ax[1].set_ylabel("Comfort compliance (%)")
    ax[1].set_title("Comfort compliance by strategy")
    ax[1].tick_params(axis="x", rotation=30)
    ax[1].set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "energy_comfort_chart.png"), dpi=150)
    plt.close(fig)

    return result


def eval_latency(contexts: list[ReadingContext], out_dir: str, simulated_cloud_delay_s: float = 0.05) -> pd.DataFrame:
    gateway = EdgeGateway(strategy=AggressiveSavingsStrategy())

    edge_times = []
    for ctx in contexts[:2000]:  # sample is enough for a latency measurement
        t0 = time.perf_counter()
        gateway.tick(ctx)
        edge_times.append(time.perf_counter() - t0)

    cloud_times = [t + simulated_cloud_delay_s for t in edge_times]

    result = pd.DataFrame({
        "path": ["Edge-only", "Cloud round-trip (simulated)"],
        "mean_latency_ms": [np.mean(edge_times) * 1000, np.mean(cloud_times) * 1000],
        "p95_latency_ms": [np.percentile(edge_times, 95) * 1000, np.percentile(cloud_times, 95) * 1000],
    }).round(3)
    result.to_csv(os.path.join(out_dir, "latency_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(result["path"], result["mean_latency_ms"], color=["#4C72B0", "#C44E52"])
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Control-loop latency: edge vs. cloud round-trip")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency_chart.png"), dpi=150)
    plt.close(fig)

    return result


def eval_resilience(contexts: list[ReadingContext], out_dir: str, outage_fraction: float = 0.15, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(contexts)
    outage_len = int(n * outage_fraction)
    outage_start = rng.integers(0, max(1, n - outage_len))
    outage_slice = set(range(outage_start, outage_start + outage_len))
    start_temp = contexts[0].indoor_temp_c

    # "With fallback" = our real design: OfflineFallbackStrategy kicks in during outage.
    gateway_fallback = EdgeGateway(strategy=AggressiveSavingsStrategy())
    thermal_fb = ThermalModel(start_temp=start_temp)
    fallback_logs = []
    sim_temp = start_temp
    for i, ctx in enumerate(contexts):
        gateway_fallback.set_strategy(OfflineFallbackStrategy() if i in outage_slice else AggressiveSavingsStrategy())
        live_ctx = ReadingContext(timestamp=ctx.timestamp, occupancy=ctx.occupancy,
                                   indoor_temp_c=sim_temp, outdoor_temp_c=ctx.outdoor_temp_c)
        result = gateway_fallback.tick(live_ctx)
        fallback_logs.append(result)
        sim_temp = thermal_fb.step(result["action"], ctx.occupancy, ctx.outdoor_temp_c)
    fallback_df = pd.DataFrame(fallback_logs)

    # "No fallback" = naive design: a cloud-dependent strategy that CANNOT compute a
    # target during the outage (returns None), so the gateway has nothing to act on
    # and freezes at "hold" -- simulating a design with no offline safeguard at all.
    gateway_naive = EdgeGateway(strategy=AggressiveSavingsStrategy())
    thermal_naive = ThermalModel(start_temp=start_temp)
    naive_logs = []
    sim_temp = start_temp
    for i, ctx in enumerate(contexts):
        live_ctx = ReadingContext(timestamp=ctx.timestamp, occupancy=ctx.occupancy,
                                   indoor_temp_c=sim_temp, outdoor_temp_c=ctx.outdoor_temp_c)
        if i in outage_slice:
            # no cloud, no fallback strategy configured -> gateway cannot safely
            # decide a new target, so nothing happens except the safety check.
            if not (live_ctx.safety_low < live_ctx.indoor_temp_c < live_ctx.safety_high):
                action = "emergency_heat" if live_ctx.indoor_temp_c <= live_ctx.safety_low else "emergency_cool"
                state = "override"
            else:
                action, state = "no-op", "frozen"
            result = {"timestamp": ctx.timestamp, "occupancy": ctx.occupancy,
                      "indoor_temp_c": live_ctx.indoor_temp_c, "target_temp_c": None,
                      "strategy": "none (cloud unreachable)", "state": state,
                      "action": action, "safety_override": state == "override"}
        else:
            result = gateway_naive.tick(live_ctx)
        naive_logs.append(result)
        sim_temp = thermal_naive.step(result["action"], ctx.occupancy, ctx.outdoor_temp_c)
    naive_df = pd.DataFrame(naive_logs)

    def compliance_during(df, mask_set):
        mask = df.index.isin(mask_set)
        return comfort_compliance_pct(df[mask]) if mask.any() else float("nan")

    result = pd.DataFrame([
        {
            "design": "With edge fallback (ADR-1)",
            "comfort_compliance_during_outage_pct": compliance_during(fallback_df, outage_slice),
            "overrides_during_outage": int(fallback_df.iloc[list(outage_slice)]["state"].eq("override").sum()),
        },
        {
            "design": "Naive (no fallback)",
            "comfort_compliance_during_outage_pct": compliance_during(naive_df, outage_slice),
            "overrides_during_outage": int(naive_df.iloc[list(outage_slice)]["state"].eq("override").sum()),
        },
    ])
    result.to_csv(os.path.join(out_dir, "resilience_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(result["design"], result["comfort_compliance_during_outage_pct"], color=["#55A868", "#C44E52"])
    ax.set_ylabel("Comfort compliance during outage (%)")
    ax.set_title(f"Resilience under a simulated {outage_fraction*100:.0f}% cloud outage")
    ax.tick_params(axis="x", rotation=10)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "resilience_chart.png"), dpi=150)
    plt.close(fig)

    return result


def eval_scalability(out_dir: str, device_counts=(10, 50, 100, 500, 1000)) -> pd.DataFrame:
    """
    Simulates ingesting N devices' worth of readings per tick and measures
    processing throughput. Uses synthetic replication of a single reading
    since we only have one real building's meters — this specifically
    tests the ingestion/processing path's scalability, not the content.
    """
    base_ctx = ReadingContext(timestamp="t", occupancy=0.5, indoor_temp_c=25.0, outdoor_temp_c=30.0)

    rows = []
    for n_devices in device_counts:
        gateway = EdgeGateway(strategy=AggressiveSavingsStrategy())
        t0 = time.perf_counter()
        for _ in range(n_devices):
            gateway.tick(base_ctx)
        elapsed = time.perf_counter() - t0
        rows.append({
            "n_devices": n_devices,
            "total_time_s": round(elapsed, 4),
            "throughput_ticks_per_s": round(n_devices / elapsed, 1) if elapsed > 0 else float("inf"),
        })

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(out_dir, "scalability_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(result["n_devices"], result["throughput_ticks_per_s"], marker="o", color="#4C72B0")
    ax.set_xlabel("Simulated device count")
    ax.set_ylabel("Throughput (ticks/s)")
    ax.set_title("Ingestion/processing throughput vs. device count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scalability_chart.png"), dpi=150)
    plt.close(fig)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run the full evaluation suite.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--interval-min", type=int, default=15)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.data)
    contexts = rows_to_contexts(df)

    print("=== 1. Energy & comfort ===")
    print(eval_energy_and_comfort(contexts, args.interval_min, args.out_dir).to_string(index=False))

    print("\n=== 2. Latency ===")
    print(eval_latency(contexts, args.out_dir).to_string(index=False))

    print("\n=== 3. Resilience ===")
    print(eval_resilience(contexts, args.out_dir).to_string(index=False))

    print("\n=== 4. Scalability ===")
    print(eval_scalability(args.out_dir).to_string(index=False))

    print(f"\nAll results and charts saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
