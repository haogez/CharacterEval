import argparse
import json
from pathlib import Path
from typing import Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM


def concat_messages(conversations, role, system):
    history = []
    first_query = system
    if conversations[0]["from"] == role:
        first_response = f"好的！现在我来扮演{role}。" + "我首先发话：" + conversations[0]["value"]
    else:
        first_response = f"好的！现在我来扮演{role}。"

    history.append({"role": "user", "content": first_query})
    history.append({"role": "assistant", "content": first_response})

    for i in range(len(conversations)):
        if conversations[i]["from"] == role:
            if i == 0:
                continue
            assert conversations[i - 1]["from"] != role
            query = f"{conversations[i - 1]['from']}：" + conversations[i - 1]["value"]
            response = f"{conversations[i]['from']}：" + conversations[i]["value"]
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
    assert conversations[-1]["from"] != role

    query = f"{conversations[-1]['from']}：" + conversations[-1]["value"]
    return history, query


def make_inputs(context):
    dialogues = context.split("\n")
    inputs = []
    for dial in dialogues:
        role = dial.split("：")[0]
        dial = "：".join(dial.split("：")[1:])
        inputs.append({"from": role, "value": dial})
    return inputs


def format_role_information(role_information: Any) -> str:
    if isinstance(role_information, dict):
        return json.dumps(role_information, ensure_ascii=False, indent=2)
    return str(role_information)


def to_chatglm_history(messages: Any):
    """Convert message list to chatglm3 history tuples."""

    history = []
    pending_user = None
    for msg in messages:
        if msg["role"] == "user":
            pending_user = msg["content"]
        elif msg["role"] == "assistant" and pending_user is not None:
            history.append((pending_user, msg["content"]))
            pending_user = None
    return history


def get_response_chatglm(data: Dict[str, Any], role_informations: Dict[str, Any], model, tokenizer):
    context = data["context"]
    role = data["role"]

    role_information = role_informations[role]
    role_system = f"""{format_role_information(role_information)}
现在请你扮演一个角色扮演专家。请你根据上述信息扮演{role}进行对话。
"""

    messages, query = concat_messages(make_inputs(context), role, role_system)
    history = to_chatglm_history(messages)
    response, _ = model.chat(tokenizer, query, history=history)

    data["model_output"] = response

    return data


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto"
    ).eval()
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用角色扮演模型生成回复")
    parser.add_argument("--model_path", required=True, help="对话模型的本地路径")
    parser.add_argument("--profile_path", type=Path, default=Path("results/generated_character_profiles.json"), help="角色档案路径")
    parser.add_argument("--test_path", type=Path, default=Path("data/test_data.jsonl"), help="测试对话数据路径")
    parser.add_argument("--output_path", type=Path, default=Path("results/generation.jsonl"), help="生成结果输出路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.profile_path.exists():
        print(f"未找到生成的角色档案 {args.profile_path} ，回退到原始档案 data/character_profiles.json")
        profile_path = Path("data/character_profiles.json")
    else:
        profile_path = args.profile_path

    with args.test_path.open("r", encoding="utf-8") as f:
        datas = json.load(f)
    with profile_path.open("r", encoding="utf-8") as f:
        role_informations = json.load(f)

    model, tokenizer = load_model(args.model_path)

    results = []
    for data in datas:
        results.append(get_response_chatglm(data, role_informations, model, tokenizer))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
