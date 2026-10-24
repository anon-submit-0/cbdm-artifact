"""
Copy to code/providers.py (or point $CBDM_PROVIDERS at your copy) and fill in keys.
Any OpenAI-compatible chat-completions endpoint works.
"""
import os
from types import SimpleNamespace as P

PROVIDERS = {
    "deepseek_v4_flash": P(model="deepseek-v4-flash", base_url="https://api.deepseek.com/v1",
                           api_key=os.environ.get("DEEPSEEK_API_KEY", "")),
    "claude_opus_4_6": P(model="claude-opus-4-6", base_url=os.environ.get("OPUS_BASE_URL", ""),
                         api_key=os.environ.get("OPUS_API_KEY", "")),
}
