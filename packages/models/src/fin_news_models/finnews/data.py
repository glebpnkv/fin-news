import logging
import os
from collections.abc import Iterable

import numpy as np
import pandas as pd
import tensorstore as ts

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s.%(msecs)03d - %(name)s.%(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_embeddings_ts_dataset(
    df_row_iter: Iterable,
    n_total: int,
    output_dir: str,
    d_embeddings: int = 768,
    write_events: int = 131_072,
    read_events: int = 4096
):
    """
    Writes news embedding vectors and timestamps data to TensorStore files, and generates a metadata file.
    This function processes data from an iterator of pandas DataFrames, splits the data into chunks, and writes them
    efficiently using TensorStore.

    Parameters
    ----------
    df_row_iter : Iterable
        An iterable of pandas DataFrames where each DataFrame represents a batch of data.
        Each DataFrame should contain the columns 'timestamp', 'company_id', and 'embedding_vector'.
    n_total : int
        The total number of records expected across all batches.
    output_dir : str
        The directory where TensorStore files and metadata should be written.
    d_embeddings : int, optional
        The dimensionality of the embedding vectors. Defaults to 768.
    write_events : int, optional
        The number of rows to write in each chunk to TensorStore. Defaults to 131_072.
    read_events : int, optional
        The chunk size for reading and compressing the data when using TensorStore. Defaults to 4,096.

    Returns
    -------
    None
        This function does not return a value. Instead, it writes data to TensorStore files and generates a
        metadata parquet file.
    """
    ctx = {
        "cache_pool": {"total_bytes_limit": 8_000_000_000},
        "gcs_request_concurrency": {"limit": 128},
    }

    # Tensorstore spec for timestamps
    ts_timestamp_spec = {
        "driver": "zarr3",
        "kvstore": {
            "driver": "file",
            "path": os.path.join(output_dir, "timestamp/")
        },
        "metadata": {
            "shape": [n_total, 1],
            "data_type": "float64",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [write_events, 1]}
            },
            "codecs": [
                {"name": "sharding_indexed",
                 "configuration": {
                     "chunk_shape": [read_events, 1],
                     "codecs": [{"name": "zstd", "configuration": {"level": 1}}]
                 }}
            ]
        },
        "create": True,
        "open": True,
        "context": ctx,
        "recheck_cached_data": "open"
    }

    ts_embedding_spec = {
        "driver": "zarr3",
        "kvstore": {
            "driver": "file",
            "path": os.path.join(output_dir, "embeddings/")
        },
        "metadata": {
            "shape": [n_total, d_embeddings],
            "data_type": "float32",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [write_events, d_embeddings]}
            },
            "codecs": [
                {"name": "sharding_indexed",
                 "configuration": {
                     "chunk_shape": [read_events, d_embeddings],
                     "codecs": [{"name": "zstd", "configuration": {"level": 1}}]
                 }}
            ]
        },
        "create": True,
        "open": True,
        "context": ctx,
        "recheck_cached_data": "open"
    }

    # TensorStore for Time
    ts_timestamp = ts.open(ts_timestamp_spec).result()
    # TensorStore for Embeddings
    ts_embeddings = ts.open(ts_embedding_spec).result()

    # Defining metadata dataframe
    df_metadata = pd.DataFrame()
    idx_offset = 0

    # Iterating through dataframes and writing to TensorStore
    logger.info("Writing news embeddings to TensorStore files.")
    for cur_idx, df in enumerate(df_row_iter):
        # Getting the metadata of the current batch
        df_meta_cur = df[["timestamp", "company_id"]]
        df_metadata = pd.concat([df_metadata, df_meta_cur], ignore_index=True)

        df["timestamp_secs"] = df["timestamp"].astype("int64") / 1e6

        cur_start = idx_offset
        cur_end = cur_start + df.shape[0]

        # Writing
        ts_embeddings[cur_start:cur_end].write(
            np.stack(df["embedding_vector"].values, axis=0).astype("float32")
        ).result()
        ts_timestamp[cur_start:cur_end].write(df[["timestamp_secs"]].values).result()

        idx_offset += df.shape[0]
        # Writing after every iteration
        df_metadata.to_parquet(os.path.join(output_dir, "df_metadata.parquet"))

        logger.info(f"Wrote batch {cur_idx + 1} to TensorStore files.")


def write_returns_ts_dataset(
    df_row_iter: Iterable,
    n_total: int,
    output_dir: str,
    d_returns: int = 1,
    write_events: int = 131_072,
    read_events: int = 4096
):
    """
    Writes time-series data of returns and timestamps to TensorStore files.

    This function writes structured financial data into TensorStore files for efficient
    retrieval and storage. The input data is divided into batches, and each batch is processed
    and written to separate TensorStore specifications for timestamps and returns. Metadata
    is also extracted and stored separately for reference.

    Parameters
    ----------
    df_row_iter : Iterable
        An iterable of pandas DataFrames where each DataFrame represents a batch of data
        containing daily returns and timestamps.
    n_total : int
        The total number of records expected across all batches.
    output_dir : str
        The directory where TensorStore files and metadata should be written.
    d_returns : int, optional
        The dimension of the returns data, default is 1.
    write_events : int, optional
        The number of rows to write in each chunk to TensorStore. Defaults to 131_072.
    read_events : int, optional
        The chunk size for reading and compressing the data when using TensorStore. Defaults to 4,096.

    Returns
    -------
    None
        This function does not return a value. Instead, it writes data to TensorStore files and generates a
        metadata parquet file.
    """
    ctx = {
        "cache_pool": {"total_bytes_limit": 8_000_000_000},
        "gcs_request_concurrency": {"limit": 128},
    }

    # Tensorstore spec for timestamps
    ts_timestamp_spec = {
        "driver": "zarr3",
        "kvstore": {
            "driver": "file",
            "path": os.path.join(output_dir, "timestamp/")
        },
        "metadata": {
            "shape": [n_total, 1],
            "data_type": "float64",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [write_events, 1]}
            },
            "codecs": [
                {"name": "sharding_indexed",
                 "configuration": {
                     "chunk_shape": [read_events, 1],
                     "codecs": [{"name": "zstd", "configuration": {"level": 1}}]
                 }}
            ]
        },
        "create": True,
        "open": True,
        "context": ctx,
        "recheck_cached_data": "open"
    }

    ts_returns_spec = {
        "driver": "zarr3",
        "kvstore": {
            "driver": "file",
            "path": os.path.join(output_dir, "returns/")
        },
        "metadata": {
            "shape": [n_total, d_returns],
            "data_type": "float32",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [write_events, d_returns]}
            },
            "codecs": [
                {"name": "sharding_indexed",
                 "configuration": {
                     "chunk_shape": [read_events, d_returns],
                     "codecs": [{"name": "zstd", "configuration": {"level": 1}}]
                 }}
            ]
        },
        "create": True,
        "open": True,
        "context": ctx,
        "recheck_cached_data": "open"
    }

    # TensorStore for Time
    ts_timestamp = ts.open(ts_timestamp_spec).result()
    # TensorStore for Returns
    ts_returns = ts.open(ts_returns_spec).result()

    # Defining metadata dataframe
    df_metadata = pd.DataFrame()
    idx_offset = 0

    # Iterating through dataframes and writing to TensorStore
    logger.info("Writing daily returns to TensorStore files.")
    for cur_idx, df in enumerate(df_row_iter):
        # Getting the metadata of the current batch
        df_meta_cur = df[["timestamp", "company_id", "ticker"]]
        df_metadata = pd.concat([df_metadata, df_meta_cur], ignore_index=True)

        df["timestamp_secs"] = df["timestamp"].astype("int64") / 1e6

        cur_start = idx_offset
        cur_end = cur_start + df.shape[0]

        # Writing
        ts_returns[cur_start:cur_end].write(
            df["returns"].values.reshape(-1, 1).astype("float32")
        ).result()
        ts_timestamp[cur_start:cur_end].write(df[["timestamp_secs"]].values).result()

        idx_offset += df.shape[0]
        # Writing after every iteration
        df_metadata.to_parquet(os.path.join(output_dir, "df_metadata.parquet"))

        logger.info(f"Wrote batch {cur_idx + 1} to TensorStore files.")
