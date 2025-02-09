import os

import pandas as pd
import requests
import zipfile
from pathlib import Path

import polars as pl


class FNSPIDController:
    # Having to hard-code whilst their HuggingFace download is broken
    articles_url: str = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"
    price_url: str = " https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip"

    def __init__(self):
        pass

    @property
    def raw_articles_name(self) -> str:
        return Path(self.articles_url).name

    def download_raw_data(self, output_dir: str):
        """
        Downloads raw FNSPID data into `output_dir`.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Download the news article data
        articles_file_path = os.path.join(output_dir, Path(self.articles_url).name)
        with requests.get(self.articles_url, stream=True) as response:
            response.raise_for_status()
            with open(articles_file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        # Download the price data archive
        price_zip_path = os.path.join(output_dir, Path(self.price_url).name)
        with requests.get(self.price_url, stream=True) as response:
            response.raise_for_status()
            with open(price_zip_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        # Extract the zip file to `full_history` directory
        full_history_dir = os.path.join(output_dir, "full_history")
        os.makedirs(full_history_dir, exist_ok=True)
        with zipfile.ZipFile(price_zip_path, 'r') as zip_ref:
            zip_ref.extractall(full_history_dir)

        # Delete the downloaded zip file
        os.remove(price_zip_path)

    def get_articles(self, input_dir: str) -> pl.LazyFrame | None:
        article_path = os.path.join(input_dir, self.raw_articles_name)
        if not os.path.exists(article_path):
            return None

        dl = pl.scan_csv(
            article_path
        )

        dl = (
            dl
            .drop([
                "Unnamed: 0",
                "Url",
                "Author",
                "Lsa_summary",
                "Luhn_summary",
                "Textrank_summary",
                "Lexrank_summary",
            ])
        )

        dl = (
            dl
            .with_columns(
                pl.col("Date")
                .str.strptime(
                    pl.Datetime(time_zone="UTC"),
                    "%Y-%m-%d %H:%M:%S %Z",
                    strict=True
                )
                .dt.replace_time_zone(time_zone=None)
                .alias("date")
            )
            .with_columns(
                pl.format(
                    "Publisher: {}; Title: {}; Article: {}",
                    pl.col("Publisher").fill_null("None"),
                    pl.col("Article_title").fill_null("None"),
                    pl.col("Article").fill_null("None")
                )
                .alias("article")
            )
            .drop(
                ["Date"]
            )
        )
        
        return dl

    def get_prices(self, input_dir: str, symbols: str | list[str] | None = None) -> pd.DataFrame | None:
        price_path = os.path.join(input_dir, "full_history")
        if not os.path.exists(price_path):
            return None

        if symbols is None:
            symbols = [f.stem for f in Path(price_path).iterdir() if f.suffix == ".csv"]
        elif isinstance(symbols, str):
            symbols = [symbols]

        df_merged = pd.DataFrame()  # Initialize an empty DataFrame to collect the data

        for symbol in symbols:
            file_path = os.path.join(price_path, f"{symbol}.csv")  # Build the full path to the CSV file
            if os.path.exists(file_path):
                try:
                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(
                        file_path,
                        usecols=["date", "open", "close"]
                    )
                    # Add a new column to identify the source file
                    df['symbol'] = symbol
                    # Append the data to the merged DataFrame
                    df_merged = pd.concat([df_merged, df], ignore_index=True)
                except Exception as e:
                    print(f"Error loading file for symbol '{symbol}': {e}")
            else:
                print(f"File not found for symbol '{symbol}' at path: {file_path}")

        # Applying additional formatting
        df_merged["date"] = pd.to_datetime(df_merged["date"])
        df_merged = df_merged.sort_values(
            by=["symbol", "date"],
            ignore_index=True
        )

        return df_merged
