import os
from pathlib import Path
from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().parents[1] / ".env.run"
load_dotenv(ENV_PATH)
