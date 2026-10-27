from dotenv import load_dotenv
import os

load_dotenv()

api_key_google = os.getenv("GEMINI_API")

# Openrouter is compatible with openai keys
openai_key = os.getenv("OPENROUTER_API")
