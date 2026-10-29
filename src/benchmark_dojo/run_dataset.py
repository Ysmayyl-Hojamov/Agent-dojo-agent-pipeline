"""Generate traces: attacks x user tasks x injection tasks.

AgentDojo writes one JSON trace per (user task, injection task, attack) to:

    runs/<pipeline>/<suite>/<user_task>/<attack>/<injection_task>.json

Each trace carries the full conversation (tool calls and tool outputs), the
injected text, the attack name, and the `utility` / `security` booleans. Those
files are the dataset; this script only drives the runs.

Existing traces are skipped (`force_rerun=False`), so re-running is cheap and
resumable: kill it whenever, run the same command again, it continues.

Examples
--------
    # 2 user tasks x 2 injection tasks = 4 traces, one wording
    uv run python -m benchmark_dojo.run_dataset --attacks w_bare --limit 2

    # every wording, whole suite (40 x 14 traces per wording - expensive!)
    uv run python -m benchmark_dojo.run_dataset --attacks all

    # clean runs with no injection, for negative examples
    uv run python -m benchmark_dojo.run_dataset --benign --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentdojo.attacks import load_attack
from agentdojo.attacks.attack_registry import ATTACKS
from agentdojo.benchmark import (
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
)
from agentdojo.logging import OutputLogger
from agentdojo.task_suite import get_suite

from benchmark_dojo.agent.pipeline import DEFAULT_MODEL, build_pipeline
from benchmark_dojo.attacks.wordings import WORDINGS, register_wordings

SUITES = ["workspace", "banking", "slack", "travel"]
BUILTIN_ATTACK = "important_instructions"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="workspace", choices=SUITES)
    parser.add_argument("--benchmark-version", default="v1.2")
    parser.add_argument(
        "--attacks",
        default="w_bare",
        help="comma-separated wording names, or 'all' for every wording in wordings.py",
    )
    parser.add_argument(
        "--include-builtin",
        action="store_true",
        help=f"also run AgentDojo's own '{BUILTIN_ATTACK}' attack",
    )
    parser.add_argument("--benign", action="store_true", help="run without injections (negative examples)")
    parser.add_argument("--user-tasks", default=None, help="comma-separated ids, e.g. user_task_0,user_task_1")
    parser.add_argument("--injection-tasks", default=None, help="comma-separated ids")
    parser.add_argument("--limit", type=int, default=None, help="keep only the first N user tasks and N injection tasks")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--logdir", default="runs")
    return parser.parse_args(argv)


def select(task_map: dict, csv: str | None, limit: int | None) -> list[str]:
    """Pick task ids: explicit list if given, else all, then truncate."""
    ids = [part.strip() for part in csv.split(",") if part.strip()] if csv else list(task_map)
    missing = [i for i in ids if i not in task_map]
    if missing:
        raise SystemExit(f"Unknown task ids {missing}. Available: {list(task_map)}")
    return ids[:limit] if limit else ids


def rates(results: dict[tuple[str, str], bool]) -> str:
    if not results:
        return "n=0"
    success = sum(results.values())
    return f"n={len(results)} rate={success / len(results):.0%}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    register_wordings()

    suite = get_suite(args.benchmark_version, args.suite)
    pipeline = build_pipeline(model=args.model, temperature=args.temperature)
    logdir = Path(args.logdir)
    logdir.mkdir(exist_ok=True)

    user_task_ids = select(suite.user_tasks, args.user_tasks, args.limit)
    injection_ids = select(suite.injection_tasks, args.injection_tasks, args.limit)

    print(f"pipeline : {pipeline.name}")
    print(f"suite    : {args.suite} {args.benchmark_version}")
    if args.benign:
        print(f"tasks    : {len(user_task_ids)} user (no injections)")
    else:
        print(f"tasks    : {len(user_task_ids)} user x {len(injection_ids)} injection")
    print(f"traces   : {logdir.resolve()}")
    print()

    try:
        with OutputLogger(str(logdir)):
            if args.benign:
                results = benchmark_suite_without_injections(
                    pipeline,
                    suite,
                    logdir=logdir,
                    force_rerun=False,
                    user_tasks=user_task_ids,
                )
                print("\n=== no-injection runs ===")
                print(f"utility (task completed): {rates(results['utility_results'])}")
                return 0

            if args.attacks.strip() == "all":
                attack_names = list(WORDINGS)
            else:
                attack_names = [part.strip() for part in args.attacks.split(",") if part.strip()]
            if args.include_builtin:
                attack_names.append(BUILTIN_ATTACK)

            # Any name registered in AgentDojo's attack registry is accepted,
            # so built-in attacks can be tested alongside our wordings.
            unknown = [a for a in attack_names if a not in ATTACKS]
            if unknown:
                raise SystemExit(f"Unknown attacks {unknown}. Known wordings: {list(WORDINGS)}")

            print("=== attack results (security: True = the attack worked) ===")
            for attack_name in attack_names:
                attack = load_attack(attack_name, suite, pipeline)
                results = benchmark_suite_with_injections(
                    pipeline,
                    suite,
                    attack,
                    logdir=logdir,
                    force_rerun=False,
                    user_tasks=user_task_ids,
                    injection_tasks=injection_ids,
                    verbose=False,
                )
                print(f"  {attack_name:<16} security {rates(results['security_results'])}  "
                      f"| utility {rates(results['utility_results'])}")
    except KeyboardInterrupt:
        print("\ninterrupted - traces written so far are kept", file=sys.stderr)
        return 130

    print("\nDone. Inspect one trace with: uv run python -m benchmark_dojo.data.show_trace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
