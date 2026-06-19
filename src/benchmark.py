from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


import json
import shutil
from tabulate import tabulate


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear."""
    if not expected:
        return 1.0
    matched = sum(1 for exp in expected if exp.lower() in answer.lower())
    if matched == len(expected):
        return 1.0
    elif matched > 0:
        return 0.5
    else:
        return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Add a lightweight quality score for offline mode."""
    if not answer:
        return 0.0
    return recall_points(answer, expected)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations."""
    total_tokens_only = 0
    total_prompt_tokens = 0
    total_recall = 0.0
    total_quality = 0.0
    total_compactions = 0
    total_growth = 0
    num_questions = 0

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        turns = conv["turns"]
        recall_questions = conv["recall_questions"]

        thread_id = f"thread-{conv_id}"

        # Track profile file start size
        start_size = 0
        if hasattr(agent, "profile_store"):
            start_size = agent.profile_store.file_size(user_id)

        # Run conversations
        for turn in turns:
            agent.reply(user_id, thread_id, turn)

        # Get token usages
        tokens_only = agent.token_usage(thread_id)
        prompt_tokens = agent.prompt_token_usage(thread_id)
        compactions = agent.compaction_count(thread_id)

        total_tokens_only += tokens_only
        total_prompt_tokens += prompt_tokens
        total_compactions += compactions

        # Run recall check in a fresh thread
        recall_thread_id = f"thread-{conv_id}-recall"
        for q_item in recall_questions:
            q = q_item["question"]
            expected = q_item["expected_contains"]

            reply_dict = agent.reply(user_id, recall_thread_id, q)
            ans = reply_dict.get("output", "")

            r_score = recall_points(ans, expected)
            q_score = heuristic_quality(ans, expected)

            total_recall += r_score
            total_quality += q_score
            num_questions += 1

        # Track profile file end size
        if hasattr(agent, "profile_store"):
            end_size = agent.profile_store.file_size(user_id)
            total_growth += max(0, end_size - start_size)

    avg_recall = (total_recall / num_questions) if num_questions > 0 else 1.0
    avg_quality = (total_quality / num_questions) if num_questions > 0 else 1.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_tokens_only,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=total_growth,
        compactions=total_compactions
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a markdown table or tabulated output."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions"
    ]
    data = []
    for r in rows:
        data.append([
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions
        ])
    return tabulate(data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both benchmark suites."""
    config = load_config(Path(__file__).resolve().parent.parent)

    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    standard_convs = load_conversations(standard_path)
    stress_convs = load_conversations(stress_path)

    print("=== RUNNING STANDARD BENCHMARK ===")
    
    # 1. Evaluate Baseline Agent on Standard
    # Clear state dir to avoid interference
    profiles_dir = config.state_dir / "profiles"
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)
    baseline_agent_std = BaselineAgent(config, force_offline=True)
    baseline_row_std = run_agent_benchmark("Baseline Agent", baseline_agent_std, standard_convs, config)

    # 2. Evaluate Advanced Agent on Standard
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)
    advanced_agent_std = AdvancedAgent(config, force_offline=True)
    advanced_row_std = run_agent_benchmark("Advanced Agent", advanced_agent_std, standard_convs, config)

    print(format_rows([baseline_row_std, advanced_row_std]))
    print()

    print("=== RUNNING LONG-CONTEXT STRESS BENCHMARK ===")

    # 3. Evaluate Baseline Agent on Stress
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)
    baseline_agent_stress = BaselineAgent(config, force_offline=True)
    baseline_row_stress = run_agent_benchmark("Baseline Agent", baseline_agent_stress, stress_convs, config)

    # 4. Evaluate Advanced Agent on Stress
    # Let's adjust threshold for stress test if needed, or use config values.
    # The config values are 1000 and 4.
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)
    advanced_agent_stress = AdvancedAgent(config, force_offline=True)
    advanced_row_stress = run_agent_benchmark("Advanced Agent", advanced_agent_stress, stress_convs, config)

    print(format_rows([baseline_row_stress, advanced_row_stress]))


if __name__ == "__main__":
    main()


