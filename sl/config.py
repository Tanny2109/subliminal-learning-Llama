import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model-specific endpoint mapping for fine-tuned models
# Format: "model_name": "http://host:port"
OLLAMA_MODEL_ENDPOINTS = {
    # Fine-tuned models from your setup
    "Toxic100_0": "http://127.0.0.1:11434",
    "Toxic100_1": "http://127.0.0.1:11435", 
    "Toxic100_2": "http://127.0.0.1:11436",
    "Toxic100_3": "http://127.0.0.1:11437",
    # Add more fine-tuned models as needed
}
