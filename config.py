"""Configuration module.

Loads environment variables and exposes constants used across the
classification pipeline. Keeping this small and explicit avoids
surprise implicit defaults sprinkled throughout the codebase.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file if present (safe for dev/local usage).
load_dotenv()

# --- OpenAI / LLM settings ---
# API key is required by nodes that invoke the vision / classification model.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Central place to switch model version if upgraded.
OPENAI_MODEL = "gpt-4o"

# --- Domain taxonomy ---
# Explicit allowed categories ("reject" is produced dynamically when needed).
CATEGORIES = ["garbage", "potholes", "deforestation"]

# NOTE: Add future tunables (timeouts, temperature overrides, etc.) here to
# avoid scattering magic numbers or environment lookups in logic modules.
