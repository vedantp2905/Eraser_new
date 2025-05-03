import argparse
import json
import os

import pandas as pd


def load_dataset(dataset_file):
    """
    Read the data from the dataset_file.

    Parameters
    ----------
    dataset_file : str
        The path to the filtering dataset file.

    Returns
    -------
    all_line_idx : list
        The list of the line idx.

    all_position_idx : list
        The list of the position idx.

    all_token: list
        The list of the token.

    all_embedding : list
        The list of the embedding.
    """
    all_line_idx = []
    all_position_idx = []
    all_token = []
    all_embedding = []

    with open(dataset_file) as f:
        dataset = json.load(f)

    for data in dataset:
        info = data[0].split("|||")
        embedding = data[1]
        
        # Print first few entries to debug
        if len(all_token) < 5:
            print(f"Dataset entry:")
            print(f"  Full info: {info}")
            print(f"  Token: {info[0]}")
            print(f"  Line idx: {info[2]}")
            print(f"  Position idx: {info[3]}")
        
        all_token.append(info[0])
        all_line_idx.append(info[2])
        all_position_idx.append(info[3])
        all_embedding.append(embedding)

    print(f"\nDataset Statistics:")
    print(f"Total tokens: {len(all_token)}")
    print(f"Unique line_idx values: {len(set(all_line_idx))}")
    print(f"Unique position_idx values: {len(set(all_position_idx))}")
    return all_line_idx, all_position_idx, all_token, all_embedding


def load_clusters(cluster_file):
    """
    Read the data from the cluster_file.

    Parameters
    ----------
    cluster_file : str
        The path to the cluster file.

    Returns
    -------
    clusterMap : dict
        The dict of the cluster that save the cluster id for each token.
    """
    clusterMap = {}
    with open(cluster_file, 'r') as f:
        clusters = f.readlines()
        
        # Print first few entries to debug
        if len(clusters) > 0:
            print(f"First cluster entry: {clusters[0]}")
            parts = clusters[0].rstrip().split('|||')
            print(f"Split parts: {parts}")

        for line in clusters:
            parts = line.rstrip().split('|||')
            # Format: token|||line_idx|||sentence_idx|||position_idx|||cluster_idx
            token = parts[0]
            line_idx = parts[1]
            position_idx = parts[3]
            cluster_idx = parts[4]
            
            # Debug first few entries
            if len(clusterMap) < 5:
                print(f"Creating mapping for: {token}")
                print(f"line_idx: {line_idx}, position_idx: {position_idx}, cluster: {cluster_idx}")
            
            token_pos = f"{line_idx}_{position_idx}"
            clusterMap[token_pos] = cluster_idx

    print(f"Loaded {len(clusterMap)} cluster mappings")
    # Debug: show some sample mappings
    sample_keys = list(clusterMap.keys())[:5]
    print("Sample mappings:")
    for key in sample_keys:
        print(f"{key}: {clusterMap[key]}")
    return clusterMap


def mapping_tokens_with_cluster_idx(all_line_idx, all_position_idx, clusterMap):
    """Map tokens with cluster indices."""
    cluster_idx = []
    missing_mappings = 0
    first_misses = []
    
    for i in range(len(all_line_idx)):
        # Convert 0-based to 1-based indexing for line_idx
        line_idx = str(int(all_line_idx[i]) + 1)  # Convert to 1-based indexing
        position_idx = all_position_idx[i]
        token_pos = f"{line_idx}_{position_idx}"
        
        # Debug first few entries
        if i < 5:
            print(f"Original line_idx: {all_line_idx[i]}, Converted line_idx: {line_idx}")
            print(f"Looking up token_pos: {token_pos}")
            print(f"Available keys sample: {list(clusterMap.keys())[:5]}")
            
        if token_pos in clusterMap:
            cluster_idx.append(clusterMap[token_pos])
        else:
            missing_mappings += 1
            if len(first_misses) < 5:
                first_misses.append((token_pos, all_line_idx[i], position_idx))
            cluster_idx.append("-1")
            
    if missing_mappings > 0:
        print(f"Warning: {missing_mappings} tokens had no cluster mapping")
        print(f"First few missing mappings (token_pos, original_line, position):")
        for miss in first_misses:
            print(f"{miss[0]}, original_line: {miss[1]}, position: {miss[2]}")
    
    print(f"Mapped {len(cluster_idx)} tokens to clusters")
    return cluster_idx


def generate_csv_file(all_line_idx, all_position_idx, all_token, all_embedding, cluster_idx, output_file):
    """
    Generate the csv file.

    Parameters
    ----------
    all_line_idx : list
        The list of the line idx.

    all_position_idx : list
        The list of the position idx.

    all_token: list
        The list of the token.

    all_embedding : list
        The list of the embedding.

    cluster_idx : list
        The list of the cluster idx.

    output_file : str
        The path to the output file.
    """
    df = pd.DataFrame({
        'token': all_token,
        'line_idx': all_line_idx,
        'position_idx': all_position_idx,
        'embedding': all_embedding,
        'cluster_idx': cluster_idx
    })
    
    print(f"Generated DataFrame with {len(df)} rows")
    print("Sample of DataFrame:")
    print(df.head())
    
    df.to_csv(output_file, index=False, sep='\t')
    print(f"Saved CSV to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str, help='The path to the filtering dataset file.')
    parser.add_argument('--cluster_file', type=str, help='The path to the cluster file.')
    parser.add_argument('--output_file', type=str, help='The path to the output file.')

    args = parser.parse_args()

    print(f"\nLoading dataset from {args.dataset_file}")
    all_line_idx, all_position_idx, all_token, all_embedding = load_dataset(args.dataset_file)
    
    print(f"\nLoading clusters from {args.cluster_file}")
    clusterMap = load_clusters(args.cluster_file)
    
    print("\nMapping tokens to clusters")
    cluster_idx = mapping_tokens_with_cluster_idx(all_line_idx, all_position_idx, clusterMap)
    
    print(f"\nGenerating CSV file {args.output_file}")
    generate_csv_file(all_line_idx, all_position_idx, all_token, all_embedding, cluster_idx, args.output_file)


if __name__ == "__main__":
    main()

