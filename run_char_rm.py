import argparse
import json
import sys
from pathlib import Path

import torch
from BaichuanCharRM.modeling_baichuan import BaichuanCharRM
from BaichuanCharRM.tokenization_baichuan import BaichuanTokenizer

max_seq_length = 4096


def format_input(example, character_profile):
    input_text = (
        "<RoleInfo>\n\n"
        + str(character_profile[example["role"]])
        + "\n\n<Context>\n\n"
        + example["context"]
        + "\n\n<Response>\n\n"
        + example["model_output"]
        + "\n\n<Dimension>\n\n"
        + example["metric_zh"]
    )
    return input_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 CharacterRM 评估")
    parser.add_argument(
        "--profile_path",
        type=Path,
        default=Path("results/generated_character_profiles.json"),
        help="角色档案路径",
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("results/generation_trans.jsonl"),
        help="转换后的模型回复路径",
    )
    parser.add_argument(
        "--reward_model_path",
        type=Path,
        default=Path("BaichuanCharRM/"),
        help="Reward Model 本地路径",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("results/evaluation.jsonl"),
        help="评估结果保存路径",
    )
    return parser.parse_args()


def load_files(profile_path: Path, input_path: Path):
    with profile_path.open("r", encoding="utf-8") as f:
        character_profile = json.load(f)
    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    return character_profile, records


def main() -> None:
    args = parse_args()
    profile_path = args.profile_path
    if not profile_path.exists():
        print(f"未找到生成的角色档案 {profile_path} ，回退到 data/character_profiles.json")
        profile_path = Path("data/character_profiles.json")

    character_profile, records = load_files(profile_path, args.input_path)

    tokenizer = BaichuanTokenizer.from_pretrained(args.reward_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = BaichuanCharRM.from_pretrained(args.reward_model_path, torch_dtype=torch.bfloat16).cuda()

    import tqdm

    for record in tqdm.tqdm(records):
        input_text = format_input(record, character_profile)
        input_ids = tokenizer.encode(text=input_text, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[-max_seq_length:]
        input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
        with torch.no_grad():
            score = base_model(input_ids=input_ids)[1].item() * 4 + 1
            record[record["metric_en"]] = score

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(records, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
