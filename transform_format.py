import argparse
import copy
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="格式转换以适配奖励模型")
    parser.add_argument("--id2metric_path", type=Path, default=Path("data/id2metric.jsonl"))
    parser.add_argument("--generation_path", type=Path, default=Path("results/generation.jsonl"))
    parser.add_argument("--output_path", type=Path, default=Path("results/generation_trans.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.id2metric_path.open("r", encoding="utf-8") as f:
        id_metric = json.load(f)

    with args.generation_path.open("r", encoding="utf-8") as f:
        datas = json.load(f)

    results = []

    for data in datas:
        if data.get("model_output") is not None and data["model_output"] != "ERROR":
            model_output = data["model_output"].split("\n")[0]
            data["model_output"] = model_output
            if str(data["id"]) in id_metric:
                for x in id_metric[str(data["id"])]:
                    data["metric_en"] = x[0]
                    data["metric_zh"] = x[1]
                    tmp = copy.deepcopy(data)
                    results.append(tmp)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
