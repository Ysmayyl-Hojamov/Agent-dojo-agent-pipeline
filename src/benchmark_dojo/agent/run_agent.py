import json
from pathlib import Path

from agentdojo.attacks.attack_registry import ATTACKS

for attack in ATTACKS:
    print(attack)

from agentdojo.agent_pipeline import (
    AgentPipeline,
    OpenAILLM,
    InitQuery,
    SystemMessage,
    ToolsExecutionLoop,
    ToolsExecutor,
)
from agentdojo.attacks import load_attack
from agentdojo.benchmark import benchmark_suite_with_injections
from agentdojo.logging import OutputLogger
from agentdojo.models import MODEL_NAMES
from agentdojo.task_suite import get_suite

from openai import OpenAI

import logging

logging.basicConfig(level=logging.DEBUG)


from benchmark_dojo.utils import openai_key

MY_MODEL = "deepseek/deepseek-v4.1-flash-1017"
MODEL_NAMES[MY_MODEL] = "DeepSeek"


client = OpenAI(
    api_key=openai_key,
    base_url="https://openrouter.ai/api/v1",
)
llm = OpenAILLM(model=MY_MODEL, client=client)


pipeline = AgentPipeline(
    [
        SystemMessage("You are a helpful assistant."),
        InitQuery(),
        llm,
        ToolsExecutionLoop([ToolsExecutor(), llm]),
    ]
)

pipeline.name = MY_MODEL
# --- the attack: one user task, one injection task ---
suite = get_suite("v1.2", "workspace")
# I can swap the type of the attack to create a new dataset
attack = load_attack("important_instructions", suite, pipeline)

logdir = Path("./runs")
logdir.mkdir(exist_ok=True)


with OutputLogger(str(logdir)):
    """
    Could use the same function but iterative.
    May use different wordings by a way of incantating
    PREFIX (tone, urgency, and/or authority) + GOAL (existing goal) +- SUFFIX (Basically the continuation of PREFIX.)
    — but I don't know how it's gonna word with polite goals (see injection task 0)
    """
    results = benchmark_suite_with_injections(
        pipeline,
        suite,
        attack,
        logdir=logdir,
        force_rerun=True,
        user_tasks=["user_task_0"],
        injection_tasks=["injection_task_0"],
    )

print("\n=== RESULTS ===")
print("Utility  (did the agent do the user's task?):", results["utility_results"])
print("Security (did the attack succeed? True = bad):", results["security_results"])


# --- print the logs for the run ---
def text_of(content):
    """Message content is a string in older versions, a list of blocks in newer ones."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("content", "") or b.get("text", "")
            for b in content
            if isinstance(b, dict)
        )
    return str(content or "")


for log_file in logdir.rglob("injection_task_0.json"):
    if "important_instructions" not in str(log_file):
        continue
    log = json.loads(log_file.read_text())

    print(f"\n=== LOG: {log_file} ===")
    print("Injections placed in tool outputs:")
    for key, value in (log.get("injections") or {}).items():
        print(f"  [{key}] {value[:200]}...")

    print("\nConversation:")
    for msg in log["messages"]:
        print(f"\n[{msg['role'].upper()}] {text_of(msg.get('content'))[:500]}")
        for call in msg.get("tool_calls") or []:
            print(f"   -> tool_call: {call['function']}({call.get('args')})")
        if msg.get("error"):
            print(f"   !! error: {msg['error']}")
