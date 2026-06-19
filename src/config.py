from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Loads environment variables and returns a LabConfig."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    
    # Load .env file if it exists at the root
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # Fallback to default load

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"

    # Sensible defaults for compact memory
    compact_threshold = int(os.environ.get("COMPACT_THRESHOLD_TOKENS", "1000"))
    compact_keep = int(os.environ.get("COMPACT_KEEP_MESSAGES", "4"))

    # Main agent LLM configuration
    provider_name = os.environ.get("LLM_PROVIDER", "openai")
    model_name = os.environ.get("LLM_MODEL", "gpt-4o")
    temp = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
    
    # Retrieve standard keys based on provider
    api_key = os.environ.get(f"{provider_name.upper()}_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get(f"{provider_name.upper()}_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

    model_config = ProviderConfig(
        provider=provider_name,
        model_name=model_name,
        temperature=temp,
        api_key=api_key,
        base_url=base_url
    )

    # Judge LLM configuration
    judge_provider = os.environ.get("JUDGE_PROVIDER", "openai")
    judge_model_name = os.environ.get("JUDGE_MODEL", "gpt-4o")
    judge_temp = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))
    judge_api_key = os.environ.get(f"{judge_provider.upper()}_API_KEY") or os.environ.get("OPENAI_API_KEY")
    judge_base_url = os.environ.get(f"{judge_provider.upper()}_BASE_URL")

    judge_config = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=judge_temp,
        api_key=judge_api_key,
        base_url=judge_base_url
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_config,
        judge_model=judge_config
    )

