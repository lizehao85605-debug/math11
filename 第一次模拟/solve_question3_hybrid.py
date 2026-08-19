"""Hybrid Question 3 solver combining compact proof and schedule quality.

The monthly layer uses a seven-rest-day path flow. The resulting schedule is
then certified against exact daily peak lower bounds and completed with the
surplus and named-worker fairness objectives used by the local solution.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from scipy.optimize import Bounds

from optimize_question3_results import assign_individual_low_hours, enhance_workbook
from solve_question2 import (
    IDX,
    append_constraint,
    build_base_model,
    load_arrivals,
    run_milp,
)
from solve_question3 import (
    MAX_CONSECUTIVE,
    WORK_DAYS_PER_WORKER,
    SparseConstraints,
    assign_workers_to_joint_shifts,
    derive_daily_minimums,
    lower_bounds,
    maximum_consecutive,
    solve_milp,
    solve_question3_daily_schedules,
    style_worksheet,
    write_joint_results,
)


DAYS = 30
REST_DAYS = DAYS - WORK_DAYS_PER_WORKER


@dataclass(frozen=True)
class PathFlowIndex:
    hired: int
    first_rest: dict[int, int]
    arcs: dict[tuple[int, int, int], int]
    max_surplus: int
    size: int


@dataclass
class HybridHiringSolution:
    hired: int
    max_surplus: int
    daily_work_counts: np.ndarray
    first_rest: dict[int, int]
    arcs: dict[tuple[int, int, int], int]
    variable_count: int


def make_path_indices(streak_limit: int = MAX_CONSECUTIVE) -> PathFlowIndex:
    offset = 0
    hired = offset
    offset += 1
    first_rest = {}
    for day in range(streak_limit + 1):
        first_rest[day] = offset
        offset += 1

    arcs = {}
    for layer in range(1, REST_DAYS):
        for left in range(DAYS):
            for right in range(left + 1, min(DAYS, left + streak_limit + 2)):
                arcs[layer, left, right] = offset
                offset += 1

    max_surplus = offset
    offset += 1
    return PathFlowIndex(hired, first_rest, arcs, max_surplus, offset)


def _add_coefficient(coefficients: dict[int, float], index: int, value: float) -> None:
    coefficients[index] = coefficients.get(index, 0.0) + value


def _incoming_arc_indices(idx: PathFlowIndex, day: int, layer: int | None = None) -> list[int]:
    return [
        index
        for (arc_layer, _left, right), index in idx.arcs.items()
        if right == day and (layer is None or arc_layer == layer)
    ]


def _outgoing_arc_indices(idx: PathFlowIndex, day: int, layer: int) -> list[int]:
    return [
        index
        for (arc_layer, left, _right), index in idx.arcs.items()
        if arc_layer == layer and left == day
    ]


def build_path_flow_model(
    daily_minimum: np.ndarray,
    streak_limit: int = MAX_CONSECUTIVE,
) -> tuple[PathFlowIndex, SparseConstraints, Bounds, np.ndarray]:
    if len(daily_minimum) != DAYS:
        raise ValueError("Question 3 requires exactly 30 daily minima.")
    idx = make_path_indices(streak_limit)
    constraints = SparseConstraints(idx.size)
    bounds_info = lower_bounds(daily_minimum)
    upper_hired = max(bounds_info["overall"] * 2, bounds_info["overall"] + 100)
    lower = np.zeros(idx.size)
    upper = np.full(idx.size, float(upper_hired))
    lower[idx.hired] = float(bounds_info["overall"])

    coefficients = {idx.hired: 1.0}
    for index in idx.first_rest.values():
        coefficients[index] = -1.0
    constraints.add(coefficients, 0.0, 0.0)

    # First rest to second rest.
    for day in range(DAYS):
        coefficients = {}
        for index in _outgoing_arc_indices(idx, day, 1):
            _add_coefficient(coefficients, index, 1.0)
        if day in idx.first_rest:
            _add_coefficient(coefficients, idx.first_rest[day], -1.0)
        constraints.add(coefficients, 0.0, 0.0)

    # Intermediate rest-day flow conservation.
    for layer in range(2, REST_DAYS):
        for day in range(DAYS):
            coefficients = {}
            for index in _outgoing_arc_indices(idx, day, layer):
                _add_coefficient(coefficients, index, 1.0)
            for index in _incoming_arc_indices(idx, day, layer - 1):
                _add_coefficient(coefficients, index, -1.0)
            constraints.add(coefficients, 0.0, 0.0)

    # The seventh rest must leave at most streak_limit workdays at month end.
    coefficients = {idx.hired: -1.0}
    final_rest_earliest = DAYS - streak_limit - 1
    for (layer, _left, right), index in idx.arcs.items():
        if layer == REST_DAYS - 1 and right >= final_rest_earliest:
            _add_coefficient(coefficients, index, 1.0)
    constraints.add(coefficients, 0.0, 0.0)

    for day, minimum in enumerate(daily_minimum):
        # staffing[day] = hired - workers resting on this day.
        staffing = {idx.hired: 1.0}
        if day in idx.first_rest:
            _add_coefficient(staffing, idx.first_rest[day], -1.0)
        for index in _incoming_arc_indices(idx, day):
            _add_coefficient(staffing, index, -1.0)
        constraints.add(staffing, float(minimum), np.inf)

        surplus = dict(staffing)
        _add_coefficient(surplus, idx.max_surplus, -1.0)
        constraints.add(surplus, -np.inf, float(minimum))

    return idx, constraints, Bounds(lower, upper), np.ones(idx.size, dtype=int)


def solve_path_flow_hiring(
    daily_minimum: np.ndarray,
    time_limit: float,
    streak_limit: int = MAX_CONSECUTIVE,
) -> HybridHiringSolution:
    idx, constraints, bounds, integrality = build_path_flow_model(daily_minimum, streak_limit)

    objective = np.zeros(idx.size)
    objective[idx.hired] = 1.0
    stage1 = solve_milp(
        objective, constraints, bounds, integrality, time_limit, "Hybrid hiring stage"
    )
    hired = int(round(float(stage1.x[idx.hired])))
    constraints.add({idx.hired: 1.0}, hired, hired)

    objective = np.zeros(idx.size)
    objective[idx.max_surplus] = 1.0
    stage2 = solve_milp(
        objective, constraints, bounds, integrality, time_limit, "Hybrid surplus stage"
    )
    max_surplus = int(round(float(stage2.x[idx.max_surplus])))
    constraints.add({idx.max_surplus: 1.0}, max_surplus, max_surplus)

    # Tie-break by reducing seven-day work blocks without changing N or surplus.
    objective = np.zeros(idx.size)
    if streak_limit in idx.first_rest:
        objective[idx.first_rest[streak_limit]] = 1.0
    final_rest_earliest = DAYS - streak_limit - 1
    for (layer, left, right), index in idx.arcs.items():
        if right - left == streak_limit + 1:
            objective[index] += 1.0
        if layer == REST_DAYS - 1 and right == final_rest_earliest:
            objective[index] += 1.0
    stage3 = solve_milp(
        objective, constraints, bounds, integrality, time_limit, "Hybrid streak tie-breaker"
    )

    values = np.rint(stage3.x).astype(int)
    first_rest = {day: int(values[index]) for day, index in idx.first_rest.items()}
    arcs = {key: int(values[index]) for key, index in idx.arcs.items()}
    daily_counts = []
    for day in range(DAYS):
        resting = first_rest.get(day, 0) + sum(
            count for (_layer, _left, right), count in arcs.items() if right == day
        )
        daily_counts.append(hired - resting)

    result = HybridHiringSolution(
        hired=hired,
        max_surplus=max_surplus,
        daily_work_counts=np.array(daily_counts, dtype=int),
        first_rest=first_rest,
        arcs=arcs,
        variable_count=idx.size,
    )
    if result.daily_work_counts.sum() != hired * WORK_DAYS_PER_WORKER:
        raise AssertionError("Path flow does not produce exactly 23 workdays per employee.")
    if np.any(result.daily_work_counts < daily_minimum):
        raise AssertionError("Path flow staffs a day below its minimum.")
    if int((result.daily_work_counts - daily_minimum).max()) != max_surplus:
        raise AssertionError("Path-flow maximum surplus is inconsistent.")
    return result


def decompose_rest_paths(
    hiring: HybridHiringSolution,
    streak_limit: int = MAX_CONSECUTIVE,
) -> tuple[list[list[int]], list[list[bool]], list[tuple[int, ...]]]:
    first_remaining = dict(hiring.first_rest)
    arc_remaining = dict(hiring.arcs)
    rest_paths: list[tuple[int, ...]] = []

    for start in sorted(first_remaining):
        for _ in range(first_remaining[start]):
            path = [start]
            current = start
            for layer in range(1, REST_DAYS):
                choices = sorted(
                    right
                    for right in range(current + 1, min(DAYS, current + streak_limit + 2))
                    if arc_remaining.get((layer, current, right), 0) > 0
                )
                if not choices:
                    raise AssertionError("Integer path flow could not be decomposed.")
                following = choices[0]
                arc_remaining[layer, current, following] -= 1
                path.append(following)
                current = following
            rest_paths.append(tuple(path))

    if len(rest_paths) != hiring.hired or any(value != 0 for value in arc_remaining.values()):
        raise AssertionError("Path-flow decomposition left unmatched flow.")

    work_calendar = [[False] * DAYS for _ in range(hiring.hired + 1)]
    daily_workers: list[list[int]] = [[] for _ in range(DAYS)]
    for worker, rest_path in enumerate(rest_paths, start=1):
        rest_set = set(rest_path)
        flags = [day not in rest_set for day in range(DAYS)]
        if sum(flags) != WORK_DAYS_PER_WORKER or maximum_consecutive(flags) > streak_limit:
            raise AssertionError(f"Worker {worker}: invalid decomposed rest path.")
        work_calendar[worker] = flags
        for day, worked in enumerate(flags):
            if worked:
                daily_workers[day].append(worker)

    if not np.array_equal(
        np.array([len(workers) for workers in daily_workers]), hiring.daily_work_counts
    ):
        raise AssertionError("Named path decomposition does not match daily staffing.")
    return daily_workers, work_calendar, rest_paths


def solve_absolute_daily_peak_bounds(
    arrivals_by_day: dict[int, np.ndarray],
    worker_cap: int,
    time_limit: float,
) -> dict[int, int]:
    bounds: dict[int, int] = {}
    for day in sorted(arrivals_by_day):
        rows, lower, upper, variable_bounds, integrality = build_base_model(arrivals_by_day[day])
        append_constraint(
            rows,
            lower,
            upper,
            {int(index): 1.0 for index in IDX.x},
            -np.inf,
            float(worker_cap),
        )
        objective = np.zeros(IDX.size)
        objective[IDX.peak] = 1.0
        result = run_milp(
            objective,
            rows,
            lower,
            upper,
            variable_bounds,
            integrality,
            time_limit,
            "absolute daily peak lower bound",
            day,
        )
        bounds[day] = int(round(float(result.fun)))
        print(f"peak lower bound day={day:02d} value={bounds[day]}")
    return bounds


def append_hybrid_certificate(
    output_path: Path,
    hiring: HybridHiringSolution,
    daily_peak_bounds: dict[int, int],
) -> None:
    workbook = load_workbook(output_path)
    summary = workbook.worksheets[0]
    summary.append(
        [
            "\u6708\u5ea6\u6392\u73ed\u5efa\u6a21",
            "7\u4e2a\u4f11\u606f\u65e5\u8def\u5f84\u6d41",
            f"{hiring.variable_count}\u4e2a\u6574\u6570\u53d8\u91cf\uff0c\u65e0\u9010\u4eba\u6807\u7b7e\u5bf9\u79f0",
        ]
    )
    summary.append(
        [
            "\u878d\u5408\u76ee\u6807\u5c42\u6b21",
            "N -> Pmax -> Smax -> sum(Pd) -> fairness",
            "\u4eba\u6570\u3001\u5cf0\u503c\u3001\u51a7\u4f59\u3001\u9010\u65e5\u5cf0\u503c\u3001\u4e2a\u4eba\u516c\u5e73\u6027",
        ]
    )
    summary.append(
        [
            "\u5168\u6708\u5cf0\u503c\u8ba4\u8bc1\u4e0b\u754c",
            max(daily_peak_bounds.values()),
            "\u5141\u8bb8\u6bcf\u65e5\u4efb\u610f\u4e0d\u8d85\u8fc7581\u4eba\u65f6\u4ecd\u4e0d\u53ef\u964d\u4f4e",
        ]
    )
    summary.append(
        [
            "\u9010\u65e5\u5cf0\u503c\u8ba4\u8bc1\u4e0b\u754c\u4e4b\u548c",
            sum(daily_peak_bounds.values()),
            "30\u5929\u5747\u5b9e\u73b0\u5404\u81ea\u4e0b\u754c",
        ]
    )
    style_worksheet(summary, {1: 31, 2: 34, 3: 52})
    workbook.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve the fused Question 3 model.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("\u7b2c\u4e00\u8f6eB\u9898\u9644\u4ef6.xlsx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("\u95ee\u9898\u4e09\u878d\u5408\u4f18\u5316\u7ed3\u679c.xlsx"),
    )
    parser.add_argument("--time-limit", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrivals_by_day = load_arrivals(args.input)
    demand, minimum_solutions = derive_daily_minimums(arrivals_by_day, args.time_limit)

    hiring = solve_path_flow_hiring(demand.daily_minimum, args.time_limit)
    daily_workers, work_calendar, _rest_paths = decompose_rest_paths(hiring)
    actual_max_streak = max(maximum_consecutive(flags) for flags in work_calendar[1:])

    daily_peak_bounds = solve_absolute_daily_peak_bounds(
        arrivals_by_day, hiring.hired, args.time_limit
    )
    daily_solutions = solve_question3_daily_schedules(
        arrivals_by_day,
        demand.days,
        hiring.daily_work_counts,
        args.time_limit,
    )
    actual_peaks = {day: daily_solutions[day].peak_workers for day in demand.days}
    if actual_peaks != daily_peak_bounds:
        differences = {
            day: (actual_peaks[day], daily_peak_bounds[day])
            for day in demand.days
            if actual_peaks[day] != daily_peak_bounds[day]
        }
        raise AssertionError(f"Fused staffing misses daily peak lower bounds: {differences}")

    calendar = assign_workers_to_joint_shifts(daily_workers, demand.days, daily_solutions)
    low_assignments = assign_individual_low_hours(calendar, demand.days, daily_solutions)
    write_joint_results(
        args.output,
        arrivals_by_day,
        demand,
        hiring,
        lower_bounds(demand.daily_minimum),
        daily_solutions,
        calendar,
    )
    enhance_workbook(
        args.output,
        "\u8def\u5f84\u6d41\u878d\u5408\u6548\u7387\u4f18\u5148",
        MAX_CONSECUTIVE,
        demand.daily_minimum,
        minimum_solutions,
        hiring,
        daily_solutions,
        calendar,
        low_assignments,
    )
    append_hybrid_certificate(args.output, hiring, daily_peak_bounds)

    metrics = {
        "hired": hiring.hired,
        "global_peak": max(actual_peaks.values()),
        "max_surplus": hiring.max_surplus,
        "daily_peak_sum": sum(actual_peaks.values()),
        "max_streak": actual_max_streak,
    }
    expected = {
        "hired": 581,
        "global_peak": 295,
        "max_surplus": 53,
        "daily_peak_sum": 6498,
        "max_streak": 7,
    }
    if metrics != expected:
        raise AssertionError(f"Unexpected hybrid metrics: {metrics}")
    print(f"Saved fused result: {args.output.resolve()} metrics={metrics}")


if __name__ == "__main__":
    main()
