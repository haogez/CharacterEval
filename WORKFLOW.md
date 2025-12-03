# 使用 CharacterLLM 自动生成角色档案并运行 CharacterEval 全流程

下述流程说明如何在具备 GPU 的远程服务器（如 VS Code Remote Containers）上，将 `data/character_profiles.json` 的角色资料输入 `CharacterLLM.generate_character`，用生成档案替换基准档案，再按 CharacterEval 评价链路完成测试。

## 0. 环境准备（GPU 必备）
- **硬件**：NVIDIA GPU（建议显存 ≥24GB，跑 BaichuanCharRM 时需要 bfloat16 支持）。
- **驱动 / CUDA**：确保宿主机已安装匹配的 NVIDIA 驱动与 CUDA runtime，`nvidia-smi` 正常。
- **容器**：启动容器时启用 GPU，例如：
  ```bash
  docker run --gpus all -it -v /path/to/CharacterEval:/workspace/CharacterEval --workdir /workspace/CharacterEval nvcr.io/nvidia/pytorch:23.10-py3 /bin/bash
  ```
  若使用 VS Code Remote Containers，请在 `.devcontainer.json` 中加入：
  ```json
  "runArgs": ["--gpus", "all"]
  ```

## 1. 安装依赖
```bash
pip install -r requirements.txt
```
- 需要额外下载奖励模型权重：`git lfs install` 后，`git clone https://huggingface.co/morecry/BaichuanCharRM` 到仓库根目录。

## 2. 配置 OpenAI 接口
- 设置环境变量：
  ```bash
  export OPENAI_API_KEY="<你的key>"
  # 若使用自建或代理：
  export OPENAI_BASE_URL="https://your.proxy.endpoint/v1"
  export OPENAI_MODEL="gpt-4.1-mini"  # 可按需覆盖
  ```

## 3. 生成新的角色档案
- 将基准档案逐条汇总后交给 `CharacterLLM.generate_character`，输出到 `results/generated_character_profiles.json`：
  ```bash
  python generate_profiles.py \
    --input_path data/character_profiles.json \
    --output_path results/generated_character_profiles.json \
    --timeline_mode strict
  ```
- 生成脚本默认强制 `relationship_to_protagonist` 为“本人”。如需放宽时间线，可将 `--timeline_mode` 改为 `relaxed`。

## 4. 生成对话回复
- 使用你的对话模型（例：ChatGLM）并加载新档案：
  ```bash
  CUDA_VISIBLE_DEVICES=0 python get_response.py \
    --model_path /path/to/your/chatglm \
    --profile_path results/generated_character_profiles.json \
    --test_path data/test_data.jsonl \
    --output_path results/generation.jsonl
  ```
- 若未提供生成档案，脚本会自动回退到 `data/character_profiles.json`。

## 5. 转换格式以适配奖励模型
```bash
python transform_format.py \
  --id2metric_path data/id2metric.jsonl \
  --generation_path results/generation.jsonl \
  --output_path results/generation_trans.jsonl
```

## 6. 运行 CharacterRM 评估（需 GPU）
```bash
CUDA_VISIBLE_DEVICES=0 python run_char_rm.py \
  --profile_path results/generated_character_profiles.json \
  --input_path results/generation_trans.jsonl \
  --reward_model_path BaichuanCharRM/ \
  --output_path results/evaluation.jsonl
```
- 若未找到生成档案，将回退到原始基准档案。

## 7. 计算平均得分
```bash
python compute_score.py
```

## 8. 常见提示
- 若 OpenAI 端点或权限受限，可先少量测试（将 `generate_profiles.py` 改为只跑部分角色）。
- 所有输出文件默认写入 `results/`，可在对应脚本的 `--output_path` 中调整。

完成以上步骤后，`results/evaluation.jsonl` 和 `compute_score.py` 的输出即为使用新角色档案的完整评估结果。
