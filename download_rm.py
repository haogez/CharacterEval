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
