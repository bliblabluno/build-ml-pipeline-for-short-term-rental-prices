#!/usr/bin/env python
"""Download a W&B artifact, perform basic cleaning, and log the result."""

import argparse
import logging

import tempfile

import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # Using context managers makes run.finish() redundant
    with wandb.init(job_type="basic_cleaning") as run:
        run.config.update(args)

        local_path = run.use_artifact(args.input_artifact).file()
        df = pd.read_csv(local_path)

        # Actual data cleaning
        df = df[df['price'].between(10, 350)].copy()
        df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
        df['last_review'] = pd.to_datetime(df['last_review'])

        idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
        df = df[idx].copy()

        # Use temporary file as context manager
        with tempfile.NamedTemporaryFile() as fp:
            df.to_csv(fp.name, index=False)

            # Create artifact to be able to use description from args
            artifact = wandb.Artifact(
                name=args.output_artifact,
                type=args.output_type,
                description=args.output_description
            )
            artifact.add_file(fp.name)

            run.log_artifact(artifact)
            # Wait for artifact to be uploaded before exiting context manager
            artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help='Name of input artifact.',
        required=True,
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help='Name of output artifact.',
        required=True,
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help='Type of output artifact.',
        required=True,
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help='Description of output artifact.',
        required=True,
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help='Lower bound for price column.',
        required=True,
    )
    parser.add_argument(
        "--max_price",
        type=float,
        help='Upper bound for price column.',
        required=True,
    )

    args = parser.parse_args()
    go(args)
