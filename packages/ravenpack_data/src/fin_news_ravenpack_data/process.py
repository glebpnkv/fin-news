import uuid
import polars as pl


def process_ravenpack_data(df: pl.DataFrame) -> pl.DataFrame:
    # Converting timestamp
    df = df.with_columns(
        timestamp=df["timestamp"].str.to_datetime().dt.replace_time_zone(None)
    )

    # Extracting "company_id" from the file path
    df = (
        df
        .with_columns(
            pl.col("file_path")
            .str.replace(r".*/", "")
            .str.replace(r"\.jsonl$", "")
            .alias("company_id")
        )
    )

    # Dropping the "cluster" column which contains
    # related entries
    df = df.drop([
        "file_path",
        "cluster",
        "reporting_period",
        "document_type",
        "reporting_entities",
        "url"
    ])

    # Renaming "sentiment" column
    df = df.rename({
        "sentiment": "document_sentiment"
    })

    # Unpacking the "source" column
    df = df.unnest("source")
    df = df.rename({
        "name": "source_name",
        "key": "source_id",
        "rank": "source_rank",
        "id": "document_id"
    })

    # Unpacking chunks
    df = df.explode("chunks")
    df = df.unnest("chunks")

    # Getting the lengths of text
    df = df.with_columns(
        pl.col("text")
        .str.len_chars()
        .alias("text_length")
    )

    # Creating a unique UUID5 identifier per chunk
    df = df.with_columns(
        pl.concat_str(
            [
                pl.col("company_id"),
                pl.col("document_id"),
                pl.col("chunk")
            ],
            separator="-"
        ).alias("id_str")
    )

    df = df.with_columns(
        pl.col("id_str").map_elements(
            lambda x: str(uuid.uuid5(uuid.NAMESPACE_DNS, x)),
            return_dtype=pl.String
        ).alias("id")
    )

    df = df.drop([
        "section_metadata",
        "speaker",
        "entities",
        "sentences",
        "id_str"
    ])

    return df
