from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config, LabConfig



from model_provider import ProviderConfig
from memory_store import UserProfileStore


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests."""
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=50,  # low threshold so compaction triggers easily
        compact_keep_messages=2,
        model=ProviderConfig("openai", "gpt-4o", 0.0),
        judge_model=ProviderConfig("openai", "gpt-4o", 0.0)
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test-user"

    # Initial state
    assert store.read_text(user_id) == ""
    assert store.file_size(user_id) == 0

    # Write profile
    content = "- tên: DũngCT\n- nghề nghiệp: Backend engineer\n"
    path = store.write_text(user_id, content)
    assert path.exists()
    assert store.read_text(user_id) == content
    assert store.file_size(user_id) > 0

    # Edit profile
    changed = store.edit_text(user_id, "Backend engineer", "MLOps engineer")
    assert changed is True
    updated_content = store.read_text(user_id)
    assert "MLOps engineer" in updated_content
    assert "Backend engineer" not in updated_content

    # Edit failure case
    changed_fail = store.edit_text(user_id, "non-existent", "value")
    assert changed_fail is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    user_id = "user1"
    thread_id = "thread1"

    # Appending messages to exceed 50 tokens
    # Long message: ~200 chars -> 50 tokens
    msg = "This is a very long message designed to trigger compaction by exceeding the threshold tokens in the compact memory manager quickly. " * 2

    # First turn
    agent.reply(user_id, thread_id, msg)
    assert agent.compaction_count(thread_id) == 0

    # Second turn: appends another long user msg.
    # Total history contains (User msg + Assistant reply + User msg), which will exceed 50 tokens.
    agent.reply(user_id, thread_id, msg)
    assert agent.compaction_count(thread_id) > 0

    ctx = agent.compact_memory.context(thread_id)
    assert len(ctx["messages"]) == config.compact_keep_messages
    assert ctx["summary"] != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "test-user-recall"

    # Turn 1: introduce facts
    t1 = "thread-1"
    baseline.reply(user_id, t1, "Chào bạn, mình tên là DũngCT.")
    advanced.reply(user_id, t1, "Chào bạn, mình tên là DũngCT.")

    # Turn 2: fresh thread, ask name
    t2 = "thread-2"
    res_base = baseline.reply(user_id, t2, "Mình tên là gì?")
    res_adv = advanced.reply(user_id, t2, "Mình tên là gì?")

    assert "DũngCT" not in res_base["output"]
    assert "DũngCT" in res_adv["output"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "test-user-load"
    thread_id = "thread-long"

    msg = "This is a moderately long sentence to construct a conversation history. " * 3

    for _ in range(5):
        baseline.reply(user_id, thread_id, msg)
        advanced.reply(user_id, thread_id, msg)

    base_load = baseline.prompt_token_usage(thread_id)
    adv_load = advanced.prompt_token_usage(thread_id)

    # Advanced agent's prompt load should be strictly less than baseline's load because of compaction
    assert adv_load < base_load

