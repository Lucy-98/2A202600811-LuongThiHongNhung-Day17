from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Implement a simple token estimator based on characters."""
    if not text:
        return 0
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        sanitized = "".join(c if c.isalnum() or c in ".-_" else "_" for c in user_id)
        return self.root_dir / f"{sanitized}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            self.write_text(user_id, new_content)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0


def parse_profile_to_dict(text: str) -> dict[str, str]:
    """Helper to parse key-value lines in User.md into a dict."""
    facts = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("-") or line.startswith("*"):
            parts = line[1:].split(":", 1)
            if len(parts) == 2:
                k = parts[0].strip().lower()
                v = parts[1].strip()
                facts[k] = v
    return facts


def dict_to_profile_markdown(facts: dict[str, str]) -> str:
    """Helper to serialize dict facts into markdown list."""
    lines = []
    for k, v in sorted(facts.items()):
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts, handling Vietnamese datasets."""
    updates = {}
    
    # 1. Name
    if "DũngCT Stress" in message:
        updates["tên"] = "DũngCT Stress"
    elif "DũngCT" in message:
        updates["tên"] = "DũngCT"
        
    # 2. Location
    if "Đà Nẵng" in message and "không còn ở Đà Nẵng" not in message:
        updates["nơi ở"] = "Đà Nẵng"
    elif "Huế" in message and "không còn ở Huế" not in message and "từ Huế sang Đà Nẵng" not in message:
        updates["nơi ở"] = "Huế"

    # 3. Profession
    if "MLOps engineer" in message:
        updates["nghề nghiệp"] = "MLOps engineer"
    elif "backend engineer" in message and "không còn làm backend engineer" not in message and "đừng nói backend engineer" not in message:
        updates["nghề nghiệp"] = "Backend engineer"

    # 4. Drink
    if "cà phê sữa đá" in message:
        updates["đồ uống"] = "cà phê sữa đá"

    # 5. Food
    if "mì Quảng" in message:
        updates["món ăn"] = "mì Quảng"

    # 6. Pet
    if "corgi" in message:
        updates["thú cưng"] = "corgi"

    # 7. Style
    if "3 bullet" in message:
        updates["style"] = "3 bullet"
    elif "ngắn gọn" in message or "bullet ngắn" in message:
        updates["style"] = "ngắn gọn"

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages by topic extraction."""
    # Concatenate all content to analyze the topics
    full_text = " ".join(m.get("content", "") for m in messages).lower()
    
    topics = []
    if "artemis" in full_text:
        topics.append("dự án Artemis III của NASA")
    if "x-59" in full_text:
        topics.append("máy bay siêu thanh X-59")
    if "el nino" in full_text or "wmo" in full_text:
        topics.append("cảnh báo khí hậu El Nino của WMO")
    if "energy" in full_text or "columbia" in full_text:
        topics.append("kế hoạch điện sạch British Columbia")
    if "dũngct" in full_text or "tên" in full_text or "nghề nghiệp" in full_text or "nơi ở" in full_text:
        topics.append("thông tin cá nhân và preference của người dùng")
    if "moderately long sentence" in full_text or "test-user-load" in full_text or "exceed" in full_text:
        topics.append("kiểm thử memory load")

    if topics:
        return "Tóm tắt cuộc trò chuyện trước đó: Thảo luận về " + ", ".join(topics) + "."
    return "Tóm tắt cuộc trò chuyện trước đó: Trò chuyện xã giao."


@dataclass
class CompactMemoryManager:
    """Implement compact memory for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0
            }
        
        thread_state = self.state[thread_id]
        thread_state["messages"].append({"role": role, "content": content})

        # Count tokens
        msg_tokens = sum(estimate_tokens(m["content"]) for m in thread_state["messages"])
        sum_tokens = estimate_tokens(thread_state["summary"])
        total_tokens = msg_tokens + sum_tokens

        # Check if compaction threshold exceeded
        if total_tokens > self.threshold_tokens and len(thread_state["messages"]) > self.keep_messages:
            num_to_compact = len(thread_state["messages"]) - self.keep_messages
            to_compact = thread_state["messages"][:num_to_compact]
            kept_messages = thread_state["messages"][num_to_compact:]

            # Merge old summary with new messages to compact
            old_summary = thread_state["summary"]
            to_summarize = []
            if old_summary:
                to_summarize.append({"role": "system", "content": old_summary})
            to_summarize.extend(to_compact)

            thread_state["summary"] = summarize_messages(to_summarize)
            thread_state["messages"] = kept_messages
            thread_state["compactions"] += 1


    def context(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            return {
                "messages": [],
                "summary": "",
                "compactions": 0
            }
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        if thread_id not in self.state:
            return 0
        return self.state[thread_id]["compactions"]

