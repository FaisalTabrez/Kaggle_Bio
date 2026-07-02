"""
Submission CSV generator for the Kaggle competition.

Output format:
  id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
  
  Node rows: id,dataset,node,node_id,t,z,y,x,-1,-1
  Edge rows: id,dataset,edge,-1,-1,-1,-1,-1,source_id,target_id
"""

import pandas as pd
from typing import List, Dict, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.coords import Detection


def generate_submission(
    tracks_by_dataset: Dict[str, Tuple[List[Detection], List[Tuple[int, int]]]],
    output_path: str = "submission.csv",
) -> pd.DataFrame:
    """
    Generate the submission CSV from track predictions.

    Args:
        tracks_by_dataset: Dict mapping dataset_name → (node_list, edge_list).
            node_list: List of Detection objects.
            edge_list: List of (source_id, target_id) tuples.
        output_path: Path to write the CSV.

    Returns:
        DataFrame with the submission.
    """
    rows = []
    row_id = 0

    for dataset_name in sorted(tracks_by_dataset.keys()):
        nodes, edges = tracks_by_dataset[dataset_name]

        # Node rows
        for node in nodes:
            rows.append({
                "id": row_id,
                "dataset": dataset_name,
                "row_type": "node",
                "node_id": int(node.node_id),
                "t": int(node.t),
                "z": int(round(node.z)),
                "y": int(round(node.y)),
                "x": int(round(node.x)),
                "source_id": -1,
                "target_id": -1,
            })
            row_id += 1

        # Edge rows
        for src_id, dst_id in edges:
            rows.append({
                "id": row_id,
                "dataset": dataset_name,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": int(src_id),
                "target_id": int(dst_id),
            })
            row_id += 1

    df = pd.DataFrame(rows)

    # Ensure column order matches sample_submission.csv
    columns = [
        "id", "dataset", "row_type", "node_id",
        "t", "z", "y", "x", "source_id", "target_id",
    ]
    df = df[columns]

    # Write
    df.to_csv(output_path, index=False)
    print(f"Submission written: {output_path} ({len(df)} rows)")

    return df


def validate_submission(
    df: pd.DataFrame,
    expected_datasets: List[str] = None,
) -> List[str]:
    """
    Validate submission format before export.

    Returns:
        List of error messages (empty = valid).
    """
    errors = []

    # Check columns
    expected_cols = {
        "id", "dataset", "row_type", "node_id",
        "t", "z", "y", "x", "source_id", "target_id",
    }
    if set(df.columns) != expected_cols:
        errors.append(
            f"Column mismatch. Expected {expected_cols}, got {set(df.columns)}"
        )

    # Check row types
    valid_types = {"node", "edge"}
    bad_types = set(df["row_type"].unique()) - valid_types
    if bad_types:
        errors.append(f"Invalid row_types: {bad_types}")

    # Check for duplicate IDs
    if df["id"].duplicated().any():
        errors.append("Duplicate row IDs found")

    # Per-dataset validation
    for dataset in df["dataset"].unique():
        ds_df = df[df["dataset"] == dataset]
        node_df = ds_df[ds_df["row_type"] == "node"]
        edge_df = ds_df[ds_df["row_type"] == "edge"]

        node_ids = set(node_df["node_id"].values)

        # Check edge references
        for _, row in edge_df.iterrows():
            if row["source_id"] not in node_ids:
                errors.append(
                    f"{dataset}: Edge references unknown source {row['source_id']}"
                )
                break  # Don't flood with errors
            if row["target_id"] not in node_ids:
                errors.append(
                    f"{dataset}: Edge references unknown target {row['target_id']}"
                )
                break

        # Check for self-loops
        self_loops = edge_df[edge_df["source_id"] == edge_df["target_id"]]
        if len(self_loops) > 0:
            errors.append(f"{dataset}: {len(self_loops)} self-loop edges found")

    # Check expected datasets
    if expected_datasets:
        missing = set(expected_datasets) - set(df["dataset"].unique())
        if missing:
            errors.append(f"Missing datasets: {missing}")

    return errors
