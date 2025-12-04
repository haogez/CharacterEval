#!/bin/bash

set -e

echo "============================================="
echo "   Fast Downloader for BaichuanCharRM"
echo "   HuggingFace Mirror + HF Transfer"
echo "============================================="

cd /workspace/CharacterEval

# 1. 设置镜像加速
export HF_ENDPOINT="https://hf-mirror.com"
echo "[OK] Using HF mirror: $HF_ENDPOINT"

# 2. 启用 hf_transfer 多线程加速
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "[OK] Enabled HF Transfer acceleration"

# 3. 检查 huggingface_hub 是否安装
python3 - << 'PYEOF'
try:
    import huggingface_hub  # noqa: F401
    print("[OK] huggingface_hub already installed")
except ImportError:
    print("[INFO] huggingface_hub not found, installing...")
    import os
    os.system('pip install -U "huggingface_hub<1.0,>=0.19.3" hf_transfer')
PYEOF

# 4. 写入 download_rm.py
cat > download_rm.py << 'PYEOF'
from huggingface_hub import snapshot_download

print("=========================================================")
print("  Downloading BaichuanCharRM with optimized settings...")
print("  This may take a while, but it will resume if interrupted.")
print("=========================================================")

snapshot_download(
    repo_id="morecry/BaichuanCharRM",
    local_dir="BaichuanCharRM",
)

print("=========================================================")
print("  Download complete! Files saved to BaichuanCharRM/")
print("=========================================================")
PYEOF

echo "[OK] Created download_rm.py"

# 5. 运行下载（自动断点续传）
echo "Starting fast download with snapshot_download..."
python3 download_rm.py