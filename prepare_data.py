import pandas as pd
from pathlib import Path
import os

# Project directories
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Output files
TEMP_OUTPUT = PROCESSED_DIR / "flight_data_new.csv"
FINAL_OUTPUT = PROCESSED_DIR / "flight_data.csv"

# Process large files in smaller chunks
CHUNK_SIZE = 100_000

# Required input columns
REQUIRED_COLUMNS = [
    "FL_DATE",
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_DELAY",
    "CANCELLED",
    "DIVERTED",
]

def process_chunk(chunk):
    chunk.columns = chunk.columns.str.strip()

    missing = [col for col in REQUIRED_COLUMNS if col not in chunk.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only required source columns
    chunk = chunk[REQUIRED_COLUMNS].copy()

    # Convert numeric columns
    numeric_columns = [
        "CRS_DEP_TIME",
        "DEP_DELAY",
        "CANCELLED",
        "DIVERTED",
    ]

    for col in numeric_columns:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    # Parse flight date
    chunk["FL_DATE"] = pd.to_datetime(
        chunk["FL_DATE"], errors="coerce"
    )

    # Remove cancelled and diverted flights
    chunk = chunk[
        (chunk["CANCELLED"] == 0) &
        (chunk["DIVERTED"] == 0)
    ].copy()

    # Remove records with missing required values
    chunk = chunk.dropna(subset=REQUIRED_COLUMNS)

    # Departure delay classification:
    # 1 = delayed more than 15 minutes
    # 0 = not delayed more than 15 minutes
    chunk["target"] = (chunk["DEP_DELAY"] > 15).astype(int)

    # Scheduled departure hour
    chunk["CRS_DEP_TIME"] = (
        chunk["CRS_DEP_TIME"].clip(0, 2359).astype(int)
    )

    chunk["DEP_HOUR"] = (
        chunk["CRS_DEP_TIME"] // 100
    ).clip(0, 23)

    # Calendar features
    chunk["DAY_OF_WEEK"] = chunk["FL_DATE"].dt.dayofweek
    chunk["MONTH"] = chunk["FL_DATE"].dt.month
    chunk["YEAR"] = chunk["FL_DATE"].dt.year

    # Season
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }

    chunk["SEASON"] = chunk["MONTH"].map(season_map)

    # Route feature
    chunk["ROUTE"] = (
        chunk["ORIGIN"].astype(str).str.strip()
        + "_"
        + chunk["DEST"].astype(str).str.strip()
    )

    # Clean categorical values
    for col in ["OP_CARRIER", "ORIGIN", "DEST"]:
        chunk[col] = chunk[col].astype(str).str.strip()

    # Final model-ready dataset
    output_columns = [
        "FL_DATE",
        "YEAR",
        "OP_CARRIER",
        "ORIGIN",
        "DEST",
        "ROUTE",
        "CRS_DEP_TIME",
        "DEP_HOUR",
        "DAY_OF_WEEK",
        "MONTH",
        "SEASON",
        "target",
    ]

    return chunk[output_columns]


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if TEMP_OUTPUT.exists():
        TEMP_OUTPUT.unlink()

    total_rows = 0
    first_write = True

    raw_files = sorted(RAW_DIR.glob("*.csv"))

    if not raw_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}"
        )

    print("=" * 60)
    print("SKYPREDICT - FLIGHT DATA PREPROCESSING")
    print("=" * 60)

    for file_path in raw_files:
        print(f"\nProcessing: {file_path.name}")

        file_rows = 0

        reader = pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        )

        for chunk_number, chunk in enumerate(reader, start=1):
            processed = process_chunk(chunk)

            if not processed.empty:
                processed.to_csv(
                    TEMP_OUTPUT,
                    mode="w" if first_write else "a",
                    header=first_write,
                    index=False
                )

                first_write = False
                file_rows += len(processed)
                total_rows += len(processed)

            print(
                f"  Chunk {chunk_number}: "
                f"{len(processed):,} usable rows"
            )

        print(f"  Total from {file_path.name}: {file_rows:,}")

    if first_write or not TEMP_OUTPUT.exists():
        raise ValueError("No valid records were produced.")

    print("\nValidating new dataset...")

    sample = pd.read_csv(TEMP_OUTPUT, nrows=10)

    expected_columns = [
        "FL_DATE", "YEAR", "OP_CARRIER", "ORIGIN",
        "DEST", "ROUTE", "CRS_DEP_TIME", "DEP_HOUR",
        "DAY_OF_WEEK", "MONTH", "SEASON", "target"
    ]

    if sample.columns.tolist() != expected_columns:
        raise ValueError(
            f"Unexpected output columns: {sample.columns.tolist()}"
        )

    # Replace the old dataset only after successful processing
    os.replace(TEMP_OUTPUT, FINAL_OUTPUT)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Total usable records: {total_rows:,}")
    print(f"Total features: {len(expected_columns) - 1}")
    print(f"Saved to: {FINAL_OUTPUT}")


if __name__ == "__main__":
    main()