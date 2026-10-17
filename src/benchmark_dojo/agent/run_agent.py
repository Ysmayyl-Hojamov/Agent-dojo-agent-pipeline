from agentdojo import agent_pipeline
from google import genai

from benchmark_dojo.utils import api_key

client = genai.Client(api_key=api_key)

model_gemini = agent_pipeline.GoogleLLM(model="gemini-3.8-flash", client=client)

response = client.models.generate_content(
    model="gemini-3.8-flash", contents="Hello, Gemini!"
)
