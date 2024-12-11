import json
import multiprocessing as mp
import re
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Set, Tuple, Type

from prefect import task, flow
from datasets import Dataset, load_dataset
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from dpu_utils.utils.iterators import ThreadedIterator

# Constants
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
MIN_NUM_TOKENS = 10
NUM_PERM = 256
PATH_COLUMN = "path"
CONTENT = "content"

# Dataset Configuration Constants
HF_SOURCE_DATASET = "JetBrains/KStack"  # Hugging Face source dataset
DATASET_SIZE_LIMIT = 100000  # Limit dataset size for testing/resource management
JACCARD_THRESHOLD = 0.85  # Jaccard similarity threshold
OUTPUT_DATASET_PATH = "./data/data-near-dedup"  # Local output path
HF_REPO_NAME = "JB_KStack_near_dedup"  # Repository name for upload
HF_ORG = "SergGN"  # Organization or username for HF upload
TEST_SAMPLE_SIZE = 7000  # Size for test runs

@task(name="prepare_dataset", retries=2)
def prepare_dataset(
    dataset_name: str,
    size_limit: int = DATASET_SIZE_LIMIT,
    test_run: bool = False
) -> Dataset:
    """
    Load and prepare dataset with size limitations
    
    Args:
        dataset_name: Name of the dataset in Hugging Face hub
        size_limit: Maximum number of samples to process
        test_run: If True, uses a very small subset for testing
    """
    ds = load_dataset(dataset_name, split="train")
    
    if test_run:
        ds = ds.select(range(min(TEST_SAMPLE_SIZE, len(ds))))
    elif size_limit:
        ds = ds.select(range(min(size_limit, len(ds))))
    
    print(f"Prepared dataset size: {len(ds)}")
    return ds

@task(name="make_duplicate_clusters")
def make_duplicate_clusters(dataset: Dataset, jaccard_threshold: float) -> List[List[Dict]]:
    """Find duplicate clusters in the dataset"""
    print("Starting duplicate cluster creation")
    duplication_index = DuplicationIndex(jaccard_threshold)
    
    # Process each item in the dataset
    for idx in tqdm(range(len(dataset)), desc="Processing items"):
        try:
            content = dataset[idx][CONTENT]
            path = dataset[idx][PATH_COLUMN]
            
            # Create MinHash
            tokens = [t for t in NON_ALPHA.split(content) if len(t.strip()) > 0]
            if len(tokens) < MIN_NUM_TOKENS:
                continue
                
            minhash = MinHash(num_perm=NUM_PERM)
            for token in tokens:
                minhash.update(token.encode('utf-8'))
            
            duplication_index.add((idx, path), minhash)
            
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            continue
    
    return duplication_index.get_duplicate_clusters()

class DuplicationIndex:
    """Wrapper class for MinHashLSH operations"""
    def __init__(self, jaccard_threshold: float = JACCARD_THRESHOLD):
        self._duplication_jaccard_threshold = jaccard_threshold
        self._num_perm = NUM_PERM
        self._index = MinHashLSH(
            threshold=self._duplication_jaccard_threshold,
            num_perm=self._num_perm
        )
        self._duplicate_clusters = defaultdict(set)

    def add(self, code_key: Tuple, min_hash: MinHash) -> None:
        close_duplicates = self._index.query(min_hash)
        if code_key in self._index.keys:
            return

        self._index.insert(code_key, min_hash)
        if close_duplicates:
            for base_duplicate in close_duplicates:
                if base_duplicate in self._duplicate_clusters:
                    self._duplicate_clusters[base_duplicate].add(code_key)
                    break
            else:
                self._duplicate_clusters[close_duplicates[0]].add(code_key)

    def get_duplicate_clusters(self) -> List[List[Dict]]:
        duplicate_clusters = []
        for base, duplicates in self._duplicate_clusters.items():
            cluster = [base] + list(duplicates)
            cluster = [{"base_index": el[0], "path": el[1]} for el in cluster]
            duplicate_clusters.append(cluster)
        return duplicate_clusters

@task(name="process_extremes")
def process_extremes(
    duplicate_clusters: List[List[Dict]],
    dataset: Dataset,
    jaccard_threshold: float
) -> List[List[Dict]]:
    """Process clusters to find extremes"""
    extremes_list = []
    
    for cluster in tqdm(duplicate_clusters, desc="Processing clusters"):
        extremes = find_cluster_extremes(cluster, dataset, jaccard_threshold)
        extremes_list.append(extremes)
    
    return extremes_list

def find_cluster_extremes(
    cluster: List[Dict],
    dataset: Dataset,
    jaccard_threshold: float
) -> List[Dict]:
    """Find reduced cluster representation"""
    extremes = []
    for element1 in cluster:
        code1 = dataset[element1["base_index"]][CONTENT]
        for element2 in extremes:
            code2 = dataset[element2["base_index"]][CONTENT]
            tokens1 = set([t for t in NON_ALPHA.split(code1) if len(t.strip()) > 0])
            tokens2 = set([t for t in NON_ALPHA.split(code2) if len(t.strip()) > 0])
            similarity = len(tokens1 & tokens2) / len(tokens1 | tokens2)
            if similarity >= jaccard_threshold:
                element2["copies"] += 1
                break
        else:
            element1["copies"] = 1
            extremes.append(element1)
    return extremes

@flow(name="Near-Deduplication")
def deduplicate_dataset_flow(
    dataset_name: str = HF_SOURCE_DATASET,
    size_limit: int = DATASET_SIZE_LIMIT,
    jaccard_threshold: float = JACCARD_THRESHOLD,
    test_run: bool = False
) -> Tuple[Dataset, List[List[Dict]]]:
    """
    Main flow for dataset deduplication
    
    Args:
        dataset_name: Name of the dataset in Hugging Face hub
        size_limit: Maximum number of samples to process
        jaccard_threshold: Threshold for Jaccard similarity
        test_run: If True, runs on a small test subset
    """
    # Prepare dataset
    dataset = prepare_dataset(
        dataset_name=dataset_name,
        size_limit=size_limit,
        test_run=test_run
    )
    
    # Find duplicate clusters
    duplicate_clusters = make_duplicate_clusters(dataset, jaccard_threshold)
    
    # Get all duplicate indices
    duplicate_indices = set(
        x["base_index"] for cluster in duplicate_clusters for x in cluster
    )
    
    # Process extremes
    extremes_clusters = process_extremes(duplicate_clusters, dataset, jaccard_threshold)
    
    # Build extreme dictionary
    extreme_dict = {}
    for extremes in extremes_clusters:
        for element in extremes:
            extreme_dict[element["base_index"]] = element
    
    # Filter dataset
    remove_indices = duplicate_indices - set(extreme_dict.keys())
    ds_filter = dataset.filter(
        lambda x, idx: idx not in remove_indices,
        with_indices=True
    )
    
    # Update duplicate clusters with extreme information
    for cluster in duplicate_clusters:
        for element in cluster:
            element["is_extreme"] = element["base_index"] in extreme_dict
            if element["is_extreme"]:
                element["copies"] = extreme_dict[element["base_index"]]["copies"]
    
    return ds_filter, duplicate_clusters


if __name__ == "__main__":
    # Example usage with different configurations
    
    # Full run with size limit
    dedup_ds, clusters = deduplicate_dataset_flow(
        dataset_name=HF_SOURCE_DATASET,
        size_limit=DATASET_SIZE_LIMIT
    )
    
    #Test run
    #dedup_ds, clusters = deduplicate_dataset_flow(
    #    dataset_name=HF_SOURCE_DATASET,
    #    test_run=True
    #) 