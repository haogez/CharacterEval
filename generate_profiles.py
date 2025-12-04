"""
使用 CharacterLLM 将 CharacterEval 的角色档案转写为模型生成的全量角色档案。
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from openai_client import CharacterLLM


def summarize_profile(name: str, profile: Dict[str, Any]) -> str:
    segments = [f"角色名：{name}"]
    for key, value in profile.items():
        if isinstance(value, (list, dict)):
            serialized = json.dumps(value, ensure_ascii=False)
        else:
            serialized = str(value)
        segments.append(f"{key}：{serialized}")
    return "；".join(segments)


async def build_profiles(
    input_path: Path,
    output_path: Path,
    timeline_mode: str,
) -> None:
    llm = CharacterLLM()
    with input_path.open("r", encoding="utf-8") as f:
        source_profiles = json.load(f)

    generated_profiles: Dict[str, Any] = {}
    for name, profile in source_profiles.items():
        description = summarize_profile(name, profile)
        print(f"\n==== 生成角色：{name} ====")
        result = await llm.generate_character(description, timeline_mode=timeline_mode)
        generated_profiles[name] = result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(generated_profiles, f, ensure_ascii=False, indent=2)
    print(f"\n已保存生成的角色档案：{output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量生成角色档案")
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("data/character_profiles.json"),
        help="原始角色档案路径",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("results/generated_character_profiles.json"),
        help="保存生成档案的输出路径",
    )
    parser.add_argument(
        "--timeline_mode",
        choices=["strict", "relaxed"],
        default="strict",
        help="是否使用严格的逐年时间线",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(build_profiles(args.input_path, args.output_path, args.timeline_mode))
