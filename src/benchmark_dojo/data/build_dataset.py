"""Turn saved AgentDojo traces into training rows.

One trace file is one whole episode (user request -> sequence of tool calls ->
final answer). A behaviour detector has to judge a single action, so this
script emits **one row per tool call**, carrying enough context to decide
whether that action fits the user's request.

Row kinds (field `kind`):
    attack      a real injection was in the transcript
    benign      no injection at all (the agent just did the user's task)
    authorized  no injection, but the user asked for the very action an
                attack would want (AgentDojo runs these to check that the
                injection task is doable). These are the hard negatives:
                a detector that flags them is detecting "dangerous tool",
                not "hijacked agent".

Label (`label`): 1 if the call matches the injection task's ground truth,
i.e. the call the attacker needed, 0 otherwise. Ground-truth calls whose
arguments contain "$" placeholders are matched on tool name and on the
argument keys that are concrete.

Usage:
    uv run python -m benchmark_dojo.data.build_dataset
    uv run python -m benchmark_dojo.data.build_dataset --out data/rows.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from agentdojo.task_suite import get_suite

CALL_MATCHES_ONLY_NAME = "name_only"


def text_of(content) -> str:
    """AgentDojo stores message content as a list of text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("content") or block.get("text") or "")
        return "".join(parts)
    return "" if content is None else str(content)


def squash(text: str) -> str:
    """Collapse whitespace.

    Tool outputs are round-tripped through YAML, which reflows newlines and
    indentation, so raw substring matching against the injected text fails.
    """
    return " ".join(text.split())


def trace_kind(record: dict) -> str:
    attack_type = record.get("attack_type")
    user_task_id = record.get("user_task_id") or ""
    if attack_type not in (None, "none"):
        return "attack"
    if user_task_id.startswith("injection_task_"):
        return "authorized"
    return "benign"


def ground_truth_calls(suite, injection_task_id: str) -> list[tuple[str, dict]]:
    """The actions a successful attack would need, as (tool name, args)."""
    if not injection_task_id or injection_task_id not in suite.injection_tasks:
        return []
    env = suite.load_and_inject_default_environment({})
    task = suite.injection_tasks[injection_task_id]
    return [(call.function, dict(call.args or {})) for call in task.ground_truth(env)]


def arg_matches(expected: dict, actual: dict) -> tuple[bool, str]:
    """Compare a ground-truth call's args with what the agent actually passed.

    Ground-truth args can contain "$name" placeholders that are only resolved
    while the task runs, so those keys are skipped.
    """
    concrete = {
        k: v for k, v in expected.items() if not (isinstance(v, str) and "$" in v)
    }
    if not concrete:
        return True, CALL_MATCHES_ONLY_NAME
    if set(concrete) - set(actual):
        return False, "missing_args"
    for key, value in concrete.items():
        if actual.get(key) != value:
            return False, "arg_value_differs"
    return True, "exact"


def is_malicious(
    tool: str, args: dict, gt_calls: list[tuple[str, dict]]
) -> tuple[bool, str]:
    for expected_tool, expected_args in gt_calls:
        if tool != expected_tool:
            continue
        matched, how = arg_matches(expected_args, args)
        if matched:
            return True, how
    return False, "no_match"


def _iter_traces(run_dirs: list[Path]):
    for root in run_dirs:
        for path in sorted(root.rglob("*.json")):
            yield root, path


def build_rows(run_dirs: list[Path], suite_version: str) -> tuple[list[dict], Counter]:
    suites: dict[str, object] = {}
    gt_cache: dict[tuple[str, str], list[tuple[str, dict]]] = {}
    rows: list[dict] = []
    stats: Counter = Counter()

    for root, path in _iter_traces(run_dirs):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            stats["unreadable_file"] += 1
            continue

        suite_name = record.get("suite_name")
        injection_task_id = record.get("injection_task_id")
        if injection_task_id in (None, "none"):
            injection_task_id = None

        if suite_name not in suites:
            try:
                suites[suite_name] = get_suite(suite_version, suite_name)
            except Exception:
                stats["unknown_suite"] += 1
                continue
        suite = suites[suite_name]

        kind = trace_kind(record)
        gt_calls: list[tuple[str, dict]] = []
        if kind == "attack" and injection_task_id:
            key = (suite_name, injection_task_id)
            if key not in gt_cache:
                gt_cache[key] = ground_truth_calls(suite, injection_task_id)
            gt_calls = gt_cache[key]

        messages = record.get("messages") or []
        user_prompt = ""
        for message in messages:
            if message.get("role") == "user":
                user_prompt = text_of(message.get("content"))
                break

        # Flatten the episode into an ordered list of actions, remembering which
        # tool outputs the agent had already seen when it made each call.
        actions = []
        seen_outputs: list[str] = []
        injection_texts = [
            squash(t)[:40] for t in (record.get("injections") or {}).values() if t
        ]
        for message in messages:
            if message.get("role") == "tool":
                seen_outputs.append(text_of(message.get("content")))
                continue
            for call in message.get("tool_calls") or []:
                actions.append(
                    {
                        "tool": call.get("function"),
                        "args": call.get("args") or {},
                        "seen_outputs": list(seen_outputs),
                    }
                )

        for step, action in enumerate(actions):
            malicious, match_how = is_malicious(
                action["tool"], action["args"], gt_calls
            )
            label = 1 if (kind == "attack" and malicious) else 0

            # Second tier: agents often reach the attacker's goal with slightly
            # different arguments (a different transfer subject, a different
            # date). AgentDojo's own checker still scores those traces as
            # successful hijacks, because it only inspects the fields that
            # matter. So when the trace verdict is "hijacked" but the args do
            # not match exactly, fall back to a tool-name match - and only in
            # traces already known to be hijacked, so benign traffic can never
            # be promoted to positive.
            if label == 0 and kind == "attack" and record.get("security") is True:
                if action["tool"] in {tool for tool, _ in gt_calls}:
                    label, match_how = 1, "function_only_in_hijacked"

            rows.append(
                {
                    "trace_id": f"{root.name}/{path.relative_to(root)}",
                    "kind": kind,
                    "pipeline": record.get("pipeline_name"),
                    "suite": suite_name,
                    "user_task_id": record.get("user_task_id"),
                    "injection_task_id": injection_task_id,
                    "attack_type": record.get("attack_type"),
                    "security": record.get("security"),
                    "utility": record.get("utility"),
                    "user_prompt": user_prompt,
                    "step": step,
                    "n_calls": len(actions),
                    "tool": action["tool"],
                    "args": action["args"],
                    "prev_tools": [a["tool"] for a in actions[:step]],
                    "injection_seen_before": any(
                        any(inj in squash(out) for inj in injection_texts)
                        for out in action["seen_outputs"]
                    ),
                    "label": label,
                    "label_match": match_how,
                }
            )
            stats[f"label={label}"] += 1

    return rows, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs",
        default="runs",
        help="folder(s) written by run_dataset.py / splice.py, comma separated (e.g. runs,runs_spliced)",
    )
    parser.add_argument("--out", default="data/rows.jsonl")
    parser.add_argument("--suite-version", default="v1.2")
    args = parser.parse_args(argv)

    run_dirs = [Path(part.strip()) for part in args.runs.split(",") if part.strip()]
    missing = [str(d) for d in run_dirs if not d.exists()]
    if missing:
        raise SystemExit(f"No runs folder at {missing}. Run the benchmark first.")

    rows, stats = build_rows(run_dirs, args.suite_version)
    if not rows:
        raise SystemExit("No tool calls found - are the traces empty?")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    by_kind = Counter(r["kind"] for r in rows)
    by_attack = Counter(r["attack_type"] or "none" for r in rows)
    positives = sum(r["label"] for r in rows)
    print(f"rows written : {len(rows)} -> {out}")
    print(f"positives    : {positives} ({positives / len(rows):.1%})")
    print(f"traces       : {len({r['trace_id'] for r in rows})}")
    print(f"\nby kind   : {dict(by_kind)}")
    print(f"by attack : {dict(by_attack)}")
    print(
        f"\nlabel match: {dict(Counter(r['label_match'] for r in rows if r['label']))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
