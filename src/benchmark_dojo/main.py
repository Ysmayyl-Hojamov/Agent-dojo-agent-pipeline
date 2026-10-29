from agentdojo.task_suite.load_suites import get_suite


def main():
    suite = get_suite("v1.2", "banking")
    print(
        f"USER TASKS: {len(suite.user_tasks)} \n PROMPT INJECTION ATTACKS: {len(suite.injection_tasks)}"
    )


if __name__ == "__main__":
    main()
