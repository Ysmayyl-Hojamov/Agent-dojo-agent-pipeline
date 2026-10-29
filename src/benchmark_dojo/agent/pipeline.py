"""The agent pipeline that gets attacked, defined in one place.

Both `run_agent.py` (a single hand-run experiment) and `run_dataset.py`
(the sweep that produces training data) build the pipeline here, so the
model, temperature and system message are only ever changed in one spot.
"""

from __future__ import annotations

import os

from agentdojo.agent_pipeline import (
    AgentPipeline,
    InitQuery,
    OpenAILLM,
    SystemMessage,
    ToolsExecutionLoop,
    ToolsExecutor,
)
from agentdojo.models import MODEL_NAMES
from openai import OpenAI

from benchmark_dojo.utils import openai_key

DEFAULT_MODEL = os.getenv("BENCH_MODEL", "deepseek/deepseek-v4.1-flash-1017")
BASE_URL = "https://openrouter.ai/api/v1"
SYSTEM_MESSAGE = "You are a helpful assistant."


def build_pipeline(model: str = DEFAULT_MODEL, temperature: float = 0.7) -> AgentPipeline:
    """Build the tool-calling agent under attack.

    `pipeline.name` is what AgentDojo uses to name the trace directory, so we
    append the temperature when it is non-zero. Without that, runs at
    different temperatures would overwrite each other's traces.
    """
    # AgentDojo fills the prose name of the model into attack templates
    # (`{model}`), and raises if the model is missing from this registry.
    MODEL_NAMES.setdefault(model, "DeepSeek")

    client = OpenAI(api_key=openai_key, base_url=BASE_URL)
    llm = OpenAILLM(client=client, model=model, temperature=temperature)

    pipeline = AgentPipeline(
        [
            SystemMessage(SYSTEM_MESSAGE),
            InitQuery(),
            llm,
            ToolsExecutionLoop([ToolsExecutor(), llm]),
        ]
    )
    pipeline.name = model if temperature == 0 else f"{model}@t{temperature}"
    return pipeline
