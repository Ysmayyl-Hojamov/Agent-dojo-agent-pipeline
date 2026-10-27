from agentdojo.task_suite import get_suite

suite = get_suite("v1.2", "workspace")

types = []


tasks = []
for task_id, t in suite.user_tasks.items():
    tasks.append({"id": task_id, "prompt": t.PROMPT})


goal = []
for task_id, t in suite.injection_tasks.items():
    goal.append({"id": task_id, "goal": t.GOAL})


# The format:
# task | [array of differen ways to word the sentence]
different_wordings = {}

print(f"{len(tasks)} user tasks, {len(goal)} injection tasks\n")

print("=== USER TASKS ===")
for t in tasks:
    print(f"{t['id']}: {t['prompt']}")

print("\n=== GOALS (injection tasks) ===")
for g in goal:
    print(f"{g['id']}: {g['goal']}")
