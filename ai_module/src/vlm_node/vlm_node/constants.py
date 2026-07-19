import os

# ---- VLM API keys ----
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ---- Provider selection ----
# Explicit via VLM_PROVIDER=openai|gemini|qwen, or auto-detected from which key is
# set (openai > gemini > qwen). SysNav fires many vision calls per run, so a paid
# OpenAI key (high rate limits, cheap gpt-4o-mini vision) avoids Gemini free-tier
# 429s.
VLM_PROVIDER = os.environ.get("VLM_PROVIDER", "").lower()
if VLM_PROVIDER not in ("openai", "gemini", "qwen"):
    if OPENAI_API_KEY:
        VLM_PROVIDER = "openai"
    elif GEMINI_API_KEY:
        VLM_PROVIDER = "gemini"
    elif DASHSCOPE_API_KEY:
        VLM_PROVIDER = "qwen"
    else:
        VLM_PROVIDER = "gemini"  # will fail at connect time; surfaces missing-key error clearly

if VLM_PROVIDER == "openai":
    VLM_API_KEY = OPENAI_API_KEY
    VLM_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o")
    MODEL_NAME_LITE = os.environ.get("OPENAI_MODEL_LITE", "gpt-4o-mini")
elif VLM_PROVIDER == "qwen":
    VLM_API_KEY = DASHSCOPE_API_KEY
    VLM_BASE_URL = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME = os.environ.get("QWEN_MODEL", "qwen3.6-plus")
    MODEL_NAME_LITE = os.environ.get("QWEN_MODEL_LITE", "qwen3.6-flash")
else:
    VLM_API_KEY = GEMINI_API_KEY
    VLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    MODEL_NAME_LITE = os.environ.get("GEMINI_MODEL_LITE", "gemini-flash-lite-latest")

# Target object to search for in the scene
target_object = ""
room_condition     = ""
spatial_condition  = ""
anchor_object      = ""
attribute_condition = ""
