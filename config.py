MODEL_NAME = "deepseek-chat"
INPUT_NAME = "hypo-l.csv"
MAX_SAMPLES = 30
RESULTS_DIR = "results"
RESUME = True
CHECKPOINT_INTERVAL = 5
API_RETRIES = 5
API_RETRY_BASE_SECONDS = 5
API_CALL_DELAY_SECONDS = 0
GROQ_API_URL = "https://api.deepseek.com/v1/chat/completions"
# API Key loaded from .env.local
GROQ_API_KEY = ""
