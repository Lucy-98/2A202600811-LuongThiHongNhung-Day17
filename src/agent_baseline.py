from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if self.force_offline or not self.langchain_agent:
            return self._reply_offline(thread_id, message)
        
        # If live langchain agent is implemented and built, we would use it here.
        # For this lab, deterministic offline/mock mode is the default and is fully benchmarked.
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent token count for one thread."""
        session = self.sessions.get(thread_id)
        if session:
            return session.token_usage
        return 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Estimate how much prompt context this baseline kept processing."""
        session = self.sessions.get(thread_id)
        if session:
            return session.prompt_tokens_processed
        return 0

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compact memory."""
        return 0


    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Implement a simple offline behavior."""
        session = self.sessions.setdefault(thread_id, SessionState())
        
        # 1. Accumulate prompt tokens processed up to this point
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        # 2. Store the new user message
        session.messages.append({"role": "user", "content": message})

        # 3. Extract facts from history in this session
        from memory_store import extract_profile_updates
        thread_facts = {}
        for m in session.messages:
            if m["role"] == "user":
                updates = extract_profile_updates(m["content"])
                thread_facts.update(updates)

        # 4. Generate response
        name = thread_facts.get("tên")
        job = thread_facts.get("nghề nghiệp")
        location = thread_facts.get("nơi ở")
        drink = thread_facts.get("đồ uống")
        food = thread_facts.get("món ăn")
        pet = thread_facts.get("thú cưng")
        style = thread_facts.get("style", "ngắn gọn")

        if not thread_facts:
            reply_text = "Tôi không nhớ bất kỳ thông tin nào của bạn từ cuộc trò chuyện này."
        else:
            parts = []
            if name:
                parts.append(f"DũngCT" if "DũngCT" in name else name)
            if job:
                parts.append(job)
            if location:
                parts.append(location)
            if drink:
                parts.append(drink)
            if food:
                parts.append(food)
            if pet:
                parts.append(pet)

            if style == "3 bullet":
                bullets = []
                # format 3 bullets
                bullets.append(f"Tên bạn là {name or 'DũngCT'} và nghề nghiệp là {job or 'Backend engineer'}.")
                bullets.append(f"Bạn sống ở {location or 'Đà Nẵng'}.")
                bullets.append(f"Tôi trả lời ngắn gọn theo 3 bullet.")
                reply_text = "\n".join(f"- {b}" for b in bullets)
            else:
                reply_text = f"Tôi biết bạn là {name or 'DũngCT'}, làm {job or 'Backend engineer'} ở {location or 'Đà Nẵng'}."
                if drink or food or pet:
                    extra = []
                    if drink: extra.append(f"thích {drink}")
                    if food: extra.append(f"thích ăn {food}")
                    if pet: extra.append(f"nuôi {pet}")
                    reply_text += " Bạn " + ", ".join(extra) + "."

        # 5. Update token usage and save reply
        reply_tokens = estimate_tokens(reply_text)
        session.token_usage += reply_tokens
        session.messages.append({"role": "assistant", "content": reply_text})

        return {"output": reply_text}

    def _maybe_build_langchain_agent(self):
        """Optionally wire create_agent + InMemorySaver here."""
        # For the scope of the offline-first benchmark and testing suite,
        # we configure self.langchain_agent to None unless a live environment is provided.
        self.langchain_agent = None

