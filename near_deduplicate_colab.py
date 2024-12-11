import json
import os
import time
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from argparse import ArgumentParser

from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from minhash_deduplication import deduplicate_dataset


def parse_args():
    parser = ArgumentParser(description='Near deduplication')
    parser.add_argument(
        "--dataset_name",
        default="JetBrains/KStack",
        type=str,
        help="Dataset to deduplicate, path to HF repo or local path",
    )
    parser.add_argument(
        "--text_column",
        default="content",
        type=str,
        help="Column name of the text to deduplicate",
    )
    parser.add_argument(
        "--jaccard_threshold",
        default=0.85,
        type=float,
        help="Jaccard similarity threshold",
    )
    parser.add_argument(
        "--repo_name",
        default="JB_KStack_near_dedup",
        type=str,
        help="HF repo where deduplicated dataset will be pushed",
    )
    parser.add_argument(
        "--out_path",
        default="./data/data-near-dedup",
        type=str,
        help="Local directory where repo data will be saved",
    )
    parser.add_argument(
        "--org",
        default="SergGN",
        type=str,
        help="HF org/username where the data will be pushed",
    )
    parser.add_argument(
        "--shard_size",
        default=1000 << 20,
        type=int,
        help="Size of the dataset shards",
    )
    parser.add_argument(
        "--test_run",
        default=False,
        type=bool,
        help="Run a test subset if True",
    )
    return parser.parse_args()


def save_shard(shard_tuple):
    """Save shard"""
    filename, shard = shard_tuple
    shard.to_parquet(filename)


def setup_repo(repo_name, org, out_path):
    """Setup HF repo and local directory."""
    repo_url = f"{org}/{repo_name}"
    create_repo(repo_url, repo_type="dataset", exist_ok=True)
    upload_folder(folder_path=out_path, repo_id=repo_url, repo_type="dataset")


def main():
    args = parse_args()

    print("Setting up the local directory")
    output_dir = Path(args.out_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)

    print("Loading the dataset")
    t_start = time.time()
    ds = load_dataset(args.dataset_name, split="train" )
    #ds = ds.select(range(2000000)) # limit for testing
    
    print("Loading the dataset", len(ds))
    if args.test_run:
        ds = ds.select([i for i in range(7000)])
    init_size = len(ds)
    print(f"Dataset loaded in {time.time() - t_start:.2f} seconds.")

    print("Deduplicating dataset")
    t_start = time.time()
    ds, duplicate_clusters = deduplicate_dataset(ds, args.jaccard_threshold)
    new_size = len(ds)
    print(f"Deduplication completed in {time.time() - t_start:.2f} seconds.")
    print(f"Original size: {init_size}, deduplicated size: {new_size}")

    with open(output_dir / "size_info.json", "w") as f:
        json.dump([init_size, new_size, (init_size - new_size) * 100 / init_size], f)

    with open(output_dir / "duplicate_clusters.json", "w") as f:
        json.dump(duplicate_clusters, f)

    dataset_nbytes = ds.data.nbytes if ds._indices is None else ds.data.nbytes * len(ds._indices) / len(ds.data)
    num_shards = int(dataset_nbytes / args.shard_size) + 1

    print("Saving dataset shards")
    t_start = time.time()
    shards = (ds.shard(num_shards=num_shards, index=i, contiguous=True) for i in range(num_shards))
    filenames = (f"{args.out_path}/data/train-{index:05d}-of-{num_shards:05d}.parquet" for index in range(num_shards))

    with Pool(16) as p:
        list(tqdm(p.imap_unordered(save_shard, zip(filenames, shards), chunksize=4), total=num_shards))
    print(f"Dataset shards saved in {time.time() - t_start:.2f} seconds.")

    print("Uploading to Hugging Face Hub")
    setup_repo(args.repo_name, args.org, args.out_path)
    print("Upload completed.")


if __name__ == "__main__":
    main()
