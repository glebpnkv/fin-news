import os
import pandas as pd


def load_and_merge_csv_files(
    directory: str,
    symbols: str | list[str]
) -> pd.DataFrame:
    """
    Load and merge CSV files from a directory based on a list of stock symbols, with the symbol name added as a column.

    Args:
        directory (str): The path to the directory containing CSV files.
        symbols (Union[str, List[str]]): A single symbol (str) or a list of symbols (file names without extension).

    Returns:
        pd.DataFrame: A merged dataframe with a 'symbol' column to distinguish each file's data.
    """
    # Ensure symbols is a list
    if isinstance(symbols, str):
        symbols = [symbols]

    df_merged = pd.DataFrame()  # Initialize an empty DataFrame to collect the data

    for symbol in symbols:
        file_path = os.path.join(directory, f"{symbol}.csv")  # Build the full path to the CSV file
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
