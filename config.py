MODEL_NAME = "llama-3.1-8b-instant"
INPUT_NAME = "hypo-l.csv"
MAX_SAMPLES = 30
RESULTS_DIR = "results"
RESUME = True
CHECKPOINT_INTERVAL = 5
API_RETRIES = 8
API_RETRY_BASE_SECONDS = 10
API_CALL_DELAY_SECONDS = 0
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Do not hardcode secrets in source code. Set GROQ_API_KEY in .env.local
GROQ_API_KEY = ""
