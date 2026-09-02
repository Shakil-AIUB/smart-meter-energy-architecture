"""
Edge gateway control loop for the Building Energy Optimisation project.

Implements two GoF design patterns, as required by the course:

1. STRATEGY (behavioural) — interchangeable control policies.
   ControlStrategy is the abstract interface; ComfortFirstStrategy,
   AggressiveSavingsStrategy, and OfflineFallbackStrategy are concrete
   strategies the EdgeGateway can swap at runtime without changing its
   own code (open/closed principle).

2. STATE (behavioural) — the HVAC/lighting operating mode.
   OperatingState is the abstract interface; IdleState, HeatingState,
   CoolingState, and OverrideState each know how to decide the next
   state given the current sensor reading. The EdgeGateway just asks
   "what state are we in" and "what should happen next" — it never
   contains a big if/elif chain of mode logic itself.

This script runs the edge control loop over the synthetic dataset
produced by data/generate_dataset.py, simulating what would normally
run continuously on a real edge device (e.g. a Raspberry Pi gateway).

Usage:
    python control_loop.py --data ../data/building_energy_sample.csv --out control_log.csv
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


# ---------------------------------------------------------------------------
# Shared context passed to strategies and states — one sensor reading's worth
# of information plus the comfort configuration.
# ---------------------------------------------------------------------------

@dataclass
class ReadingContext:
    timestamp: str
    occupancy: float
    indoor_temp_c: float
    outdoor_temp_c: float
    comfort_low: float = 24.0
    comfort_high: float = 26.0
    safety_low: float = 20.0   # hard safety bounds — never violated regardless of strategy
    safety_high: float = 30.0


# ---------------------------------------------------------------------------
# STRATEGY PATTERN — interchangeable control policies
# ---------------------------------------------------------------------------

class ControlStrategy(ABC):
    """Common interface every control policy must implement."""

    name: str = "base"

    @abstractmethod
    def decide_target(self, ctx: ReadingContext) -> float:
        """Return the target indoor temperature (deg C) this policy wants."""
        raise NotImplementedError


class ComfortFirstStrategy(ControlStrategy):
    """Keeps indoor temperature tightly centred in the comfort band."""

    name = "comfort_first"

    def decide_target(self, ctx: ReadingContext) -> float:
        return (ctx.comfort_low + ctx.comfort_high) / 2.0


class AggressiveSavingsStrategy(ControlStrategy):
    """
    Widens the acceptable band when the space is unoccupied to save energy,
    only tightening back to true comfort when people are present.
    """

    name = "aggressive_savings"

    def decide_target(self, ctx: ReadingContext) -> float:
        if ctx.occupancy < 0.1:
            # Nobody home — let it drift toward outdoor temp, saving energy,
            # but still nudge toward a wide setback band rather than doing nothing.
            setback_low, setback_high = ctx.comfort_low - 3.0, ctx.comfort_high + 3.0
            return min(max(ctx.indoor_temp_c, setback_low), setback_high)
        return (ctx.comfort_low + ctx.comfort_high) / 2.0


class OfflineFallbackStrategy(ControlStrategy):
    """
    Used when the cloud-computed schedule is stale or unavailable
    (see ADR-1). Conservative and simple by design — it must be safe
    to run indefinitely without any cloud input.
    """

    name = "offline_fallback"

    def decide_target(self, ctx: ReadingContext) -> float:
        return (ctx.comfort_low + ctx.comfort_high) / 2.0


# ---------------------------------------------------------------------------
# STATE PATTERN — HVAC/lighting operating mode
# ---------------------------------------------------------------------------

class OperatingState(ABC):
    """Common interface every operating mode must implement."""

    name: str = "base"

    @abstractmethod
    def next_state(self, ctx: ReadingContext, target_temp: float) -> "OperatingState":
        """Given the current reading and target, decide which state to be in next."""
        raise NotImplementedError

    def action(self, ctx: ReadingContext, target_temp: float) -> str:
        """Human-readable actuator action for logging."""
        return "no-op"


class IdleState(OperatingState):
    name = "idle"

    def next_state(self, ctx: ReadingContext, target_temp: float) -> OperatingState:
        if ctx.indoor_temp_c < target_temp - 0.5:
            return HEATING
        if ctx.indoor_temp_c > target_temp + 0.5:
            return COOLING
        return IDLE

    def action(self, ctx, target_temp) -> str:
        return "hold"


class HeatingState(OperatingState):
    name = "heating"

    def next_state(self, ctx: ReadingContext, target_temp: float) -> OperatingState:
        if ctx.indoor_temp_c >= target_temp:
            return IDLE
        return HEATING

    def action(self, ctx, target_temp) -> str:
        return "heat_on"


class CoolingState(OperatingState):
    name = "cooling"

    def next_state(self, ctx: ReadingContext, target_temp: float) -> OperatingState:
        if ctx.indoor_temp_c <= target_temp:
            return IDLE
        return COOLING

    def action(self, ctx, target_temp) -> str:
        return "cool_on"


class OverrideState(OperatingState):
    """
    Safety override — entered whenever the indoor temperature crosses a
    hard safety bound, regardless of which strategy is active. This is
    the piece that ADR-1 requires to run synchronously on the edge and
    never depend on the cloud.
    """

    name = "override"

    def next_state(self, ctx: ReadingContext, target_temp: float) -> OperatingState:
        if ctx.safety_low < ctx.indoor_temp_c < ctx.safety_high:
            return IDLE  # safety cleared, return to normal control
        return OVERRIDE

    def action(self, ctx, target_temp) -> str:
        return "emergency_heat" if ctx.indoor_temp_c <= ctx.safety_low else "emergency_cool"


# Singletons — states are stateless, so one instance of each is enough.
IDLE = IdleState()
HEATING = HeatingState()
COOLING = CoolingState()
OVERRIDE = OverrideState()


# ---------------------------------------------------------------------------
# Edge gateway — the "Context" for both patterns. Holds the current
# strategy and current state, and coordinates one control-loop tick.
# ---------------------------------------------------------------------------

class EdgeGateway:
    def __init__(self, strategy: ControlStrategy):
        self._strategy = strategy          # Strategy pattern: swappable at runtime
        self._state: OperatingState = IDLE  # State pattern: current operating mode

    def set_strategy(self, strategy: ControlStrategy) -> None:
        """Swap the active control policy without touching any other logic."""
        self._strategy = strategy

    def tick(self, ctx: ReadingContext) -> dict:
        """
        Run one control-loop iteration:
        1. Safety check first, always, synchronously (never skipped).
        2. Ask the active strategy what target temperature it wants.
        3. Ask the current state what the next state and action should be.
        """
        # 1. Safety override short-circuits everything else.
        if not (ctx.safety_low < ctx.indoor_temp_c < ctx.safety_high):
            self._state = OVERRIDE

        # 2. Strategy decides the target (ignored while in override).
        target_temp = self._strategy.decide_target(ctx)

        # 3. State decides the next state + actuator action.
        next_state = self._state.next_state(ctx, target_temp)
        action = self._state.action(ctx, target_temp)
        self._state = next_state

        return {
            "timestamp": ctx.timestamp,
            "occupancy": ctx.occupancy,
            "indoor_temp_c": ctx.indoor_temp_c,
            "target_temp_c": round(target_temp, 2),
            "strategy": self._strategy.name,
            "state": self._state.name,
            "action": action,
            "safety_override": self._state.name == "override",
        }


# ---------------------------------------------------------------------------
# Runner — reads the synthetic dataset and replays it through the gateway.
# Demonstrates swapping strategies mid-run (aggressive savings by default,
# comfort-first override could be triggered by a facility-manager setting).
# ---------------------------------------------------------------------------

def run(data_path: str, out_path: str) -> None:
    df = pd.read_csv(data_path)
    gateway = EdgeGateway(strategy=AggressiveSavingsStrategy())

    logs = []
    for _, row in df.iterrows():
        ctx = ReadingContext(
            timestamp=str(row["timestamp"]),
            occupancy=float(row["occupancy"]),
            indoor_temp_c=float(row["indoor_temp_c"]),
            outdoor_temp_c=float(row["outdoor_temp_c"]),
        )
        logs.append(gateway.tick(ctx))

    log_df = pd.DataFrame(logs)
    log_df.to_csv(out_path, index=False)

    n_override = int(log_df["safety_override"].sum())
    print(f"Processed {len(log_df)} ticks -> {out_path}")
    print(f"Safety overrides triggered: {n_override}")
    print(log_df["action"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the edge control loop over a dataset.")
    parser.add_argument("--data", type=str, default="../data/building_energy_sample.csv")
    parser.add_argument("--out", type=str, default="control_log.csv")
    args = parser.parse_args()
    run(args.data, args.out)
