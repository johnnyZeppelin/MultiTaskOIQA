from huggingface_hub import snapshot_download

snapshot_download(
    repo_id = "JohnnyZeppelin/CVIQ",
    repo_type = "dataset",
    local_dir="./data/CVIQ"
)
