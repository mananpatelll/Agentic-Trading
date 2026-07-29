import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# The agents build a ChatOpenAI at import time, which raises without a key, so
# even a test that never calls the API needs one present to import `main`.
# setdefault runs before load_dotenv(), and load_dotenv does not override an
# existing value - so this also keeps the real key out of the test process.
os.environ.setdefault("OPENAI_API_KEY", "test-key-never-used")
