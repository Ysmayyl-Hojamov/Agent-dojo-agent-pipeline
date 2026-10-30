"""Splice hijacked actions into clean traces - no model needed.

Why this exists
---------------
A detector needs examples of a *hijacked* agent. Getting them by running the
agent and hoping it obeys the injection does not work: on the models we can
afford, attacks succeeded 0/20 and 2/5 times. That is too few positives to
train on, and re-running until one works is slow and expensive.

So we manufacture them. AgentDojo knows, for every attack objective, exactly
which tool call would accomplish it (`injection_task.ground_truth()`). Take a
clean trace, put the injected text into the tool output the agent reads, then
append the attacker's call after it. The result is a transcript that looks like
a successful hijack, with labels that are correct by construction.

What is real and what is not
----------------------------
Real: the clean trace (genuine model behaviour), the injected text, the
attacker's tool call and its arguments.
Not real: the *causal link*. In a spliced trace the model never actually
decided to obey; we wrote the decision in. So a detector can learn to spot
splicing artefacts rather than hijacking. Report it as a limitation, and keep
a few genuine hijacks separate as a sanity check.

Where the pieces are placed
---------------------------
AgentDojo's injection vectors are `{}` placeholders in a YAML template, so
their text is not findable by searching the trace. Instead we reuse AgentDojo's
own trick: inject a unique canary per vector, run the ground-truth pipeline,
and note which tool call's output contained the canary. That gives
vector -> tool function, which is enough to find the right tool output in a
real trace.

Output goes to `runs_spliced/` with `attack_type = "spliced_<wording>"`, so
`build_dataset.py` can read it exactly like a normal run.

Usage:
    uv run python -m benchmark_dojo.data.splice --attacks all
    uv run python -m benchmark_dojo.data.splice --attacks w_bare --injection-tasks injection_task_0
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.attacks import load_attack
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite import get_suite
from agentdojo.types import get_text_content_as_str

from benchmark_dojo.agent.pipeline import DEFAULT_MODEL, build_pipeline
from benchmark_dojo.attacks.wordings import WORDINGS, register_wordings

CANARY = "<CANARY---{}--->"


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("content") or b.get("text") or ""
            for b in content
            if isinstance(b, dict)
        )
    return "" if content is None else str(content)


def function_of(tool_message: dict) -> str | None:
    """The tool a message is describing, whether it is a dict or a model."""
    call = tool_message.get("tool_call")
    if call is None:
        return None
    if isinstance(call, dict):
        return call.get("function")
    return getattr(call, "function", None)


def vector_to_tool(suite, user_task) -> dict[str, str]:
    """Map each injection vector that this user task actually reads to the tool
    whose output carries it.

    Same machinery as `BaseAttack.get_injection_candidates`: put a unique canary
    in every vector, replay the ground-truth solution, and see which tool output
    contains which canary.
    """
    canaries = {
        vector: CANARY.format(vector)
        for vector in suite.get_injection_vector_defaults()
    }
    environment = suite.load_and_inject_default_environment(canaries)
    runtime = FunctionsRuntime(suite.tools)
    _, _, _, responses, _ = GroundTruthPipeline(user_task).query(
        user_task.PROMPT, runtime, environment
    )

    mapping: dict[str, str] = {}
    for response in responses:
        content = text_of(response.get("content"))
        for vector, canary in canaries.items():
            if canary in content:
                function = function_of(response)  # type: ignore
                if function:
                    mapping.setdefault(vector, function)
    return mapping


def find_clean_traces(runs_dir: Path, pipeline_filter: str | None) -> dict[str, dict]:
    """Pick a base clean trace per user task (attack_type is 'none')."""
    traces: dict[str, dict] = {}
    for path in sorted(runs_dir.rglob("none/none.json")):
        if pipeline_filter and pipeline_filter not in str(path):
            continue
        record = json.loads(path.read_text())
        user_task_id = record.get("user_task_id") or ""
        if not user_task_id.startswith("user_task_"):
            continue  # these are the "authorized" traces, not clean user tasks
        traces.setdefault(user_task_id, {**record, "_path": str(path)})
    return traces


def splice(
    clean: dict,
    suite,
    user_task,
    injection_task,
    injections: dict[str, str],
    vectors: dict[str, str],
) -> tuple[dict, int] | None:
    """Return a hijacked copy of `clean`, plus the number of injection vectors placed."""
    messages = copy.deepcopy(clean["messages"])

    # 1. Drop the injected text into the tool output the agent reads.
    poisoned: list[int] = []
    for vector, payload in injections.items():
        function = vectors.get(vector)
        if function is None:
            continue
        for index in reversed(range(len(messages))):
            if (
                messages[index].get("role") == "tool"
                and function_of(messages[index]) == function
            ):
                content = messages[index].get("content")
                text = text_of(content) + "\n" + payload
                messages[index]["content"] = [{"type": "text", "content": text}]
                poisoned.append(index)
                break

    if not poisoned:
        return None

    # 2. Append the attacker's calls after the last poisoned output.
    calls = [
        {
            "function": call.function,
            "args": dict(call.args or {}),
            "id": f"splice-{i}",
            "placeholder_args": call.placeholder_args,
        }
        for i, call in enumerate(
            injection_task.ground_truth(suite.load_and_inject_default_environment({}))
        )
    ]
    if not calls:
        return None

    after = max(poisoned)
    target = next(
        (
            i
            for i in range(after + 1, len(messages))
            if messages[i].get("role") == "assistant" and messages[i].get("tool_calls")
        ),
        None,
    )
    if target is not None:
        messages[target]["tool_calls"] = (
            list(messages[target].get("tool_calls") or []) + calls
        )
        result_at = target + 1
    else:
        messages.insert(
            after + 1,
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "content": "Sure, taking care of that now."}
                ],
                "tool_calls": calls,
            },
        )
        result_at = after + 2

    # 3. Give each call a plausible tool response so the transcript stays coherent.
    for offset, call in enumerate(calls):
        messages.insert(
            result_at + offset,
            {
                "role": "tool",
                "content": [{"type": "text", "content": "Done."}],
                "tool_call_id": call["id"],
                "tool_call": call,
                "error": None,
            },
        )

    spliced = {k: v for k, v in clean.items() if k != "_path"}
    spliced["messages"] = messages
    spliced["spliced"] = True
    spliced["splice_source"] = clean.get("_path")
    spliced["injections"] = injections
    # The clean trace has no injection task; without this the labeller cannot
    # find the ground-truth call to match against.
    spliced["injection_task_id"] = injection_task.ID
    spliced["security"] = True  # the attacker's calls are present by construction
    spliced["duration"] = 0.0
    return spliced, len(poisoned)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", default="runs", help="where the clean traces live")
    parser.add_argument("--out", default="runs_spliced")
    parser.add_argument("--suite", default="workspace")
    parser.add_argument("--benchmark-version", default="v1.2")
    parser.add_argument("--attacks", default="all")
    parser.add_argument("--injection-tasks", default=None)
    parser.add_argument(
        "--pipeline",
        default=None,
        help="only use clean traces from pipelines containing this string",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="only used to resolve the prose model name for templates",
    )
    args = parser.parse_args(argv)

    register_wordings()
    suite = get_suite(args.benchmark_version, args.suite)
    pipeline = build_pipeline(model=args.model, temperature=0.0)

    clean_traces = find_clean_traces(Path(args.runs), args.pipeline)
    if not clean_traces:
        raise SystemExit(
            f"No clean traces in {Path(args.runs).resolve()}. "
            "Generate some first: uv run python -m benchmark_dojo.run_dataset --benign"
        )

    injection_ids = (
        [i.strip() for i in args.injection_tasks.split(",") if i.strip()]
        if args.injection_tasks
        else list(suite.injection_tasks)
    )
    attack_names = (
        list(WORDINGS)
        if args.attacks == "all"
        else [a.strip() for a in args.attacks.split(",") if a.strip()]
    )

    out_root = Path(args.out)
    made = skipped = 0
    print(f"clean traces: {len(clean_traces)}  ({', '.join(sorted(clean_traces))})")

    for attack_name in attack_names:
        attack = load_attack(attack_name, suite, pipeline)
        for user_task_id, clean in sorted(clean_traces.items()):
            user_task = suite.user_tasks[user_task_id]
            vectors = vector_to_tool(suite, user_task)
            for injection_task_id in injection_ids:
                injection_task = suite.injection_tasks[injection_task_id]
                injections = attack.attack(user_task, injection_task)
                result = splice(
                    clean, suite, user_task, injection_task, injections, vectors
                )
                if result is None:
                    skipped += 1
                    continue
                record, placed = result
                record["attack_type"] = f"spliced_{attack_name}"

                directory = (
                    out_root
                    / clean["pipeline_name"]
                    / suite.name
                    / user_task_id
                    / f"spliced_{attack_name}"
                )
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"{injection_task_id}.json").write_text(
                    json.dumps(record, indent=4, default=str)
                )
                made += 1

    print(f"spliced traces written: {made} -> {out_root.resolve()}")
    if skipped:
        print(f"skipped (no locatable injection vector for that task): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
