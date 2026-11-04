from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

OUTPUT = PROCESSED / "flight_data.csv"
MAX_ROWS_PER_FILE = 150000
CHUNK_SIZE = 100000

if OUTPUT.exists():
    OUTPUT.unlink()

files = sorted(RAW.glob("*.csv"))

if not files:
    raise FileNotFoundError("No CSV files found in data/raw")

total = 0

for file in files:
    print(f"\nProcessing {file.name}")
    kept = 0

    try:
        reader = pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            low_memory=False
        )

        for chunk in reader:
            chunk.columns = chunk.columns.str.strip()

            aliases = {
                "UniqueCarrier": "OP_CARRIER",
                "Origin": "ORIGIN",
                "Dest": "DEST",
                "CRSDepTime": "CRS_DEP_TIME",
                "DEP_DELAY": "DepDelay",
            }

            chunk.rename(columns=aliases, inplace=True)

            required = [
                "OP_CARRIER",
                "ORIGIN",
                "DEST",
                "CRS_DEP_TIME"
            ]

            if "DEP_DEL15" in chunk.columns:
                chunk["target"] = pd.to_numeric(
                    chunk["DEP_DEL15"], errors="coerce"
                )
            elif "DepDelay" in chunk.columns:
                delay = pd.to_numeric(
                    chunk["DepDelay"], errors="coerce"
                )
                chunk["target"] = (delay > 15).astype(float)
                chunk.loc[delay.isna(), "target"] = float("nan")
            elif "ARR_DEL15" in chunk.columns:
                chunk["target"] = pd.to_numeric(
                    chunk["ARR_DEL15"], errors="coerce"
                )
            elif "ArrDelay" in chunk.columns:
                delay = pd.to_numeric(
                    chunk["ArrDelay"], errors="coerce"
                )
                chunk["target"] = (delay > 15).astype(float)
                chunk.loc[delay.isna(), "target"] = float("nan")
            else:
                raise ValueError(
                    f"No delay target column found in {file.name}"
                )

            if not all(c in chunk.columns for c in required):
                raise ValueError(
                    f"Required columns missing in {file.name}: "
                    f"{chunk.columns.tolist()}"
                )

            chunk["CRS_DEP_TIME"] = pd.to_numeric(
                chunk["CRS_DEP_TIME"], errors="coerce"
            )

            chunk["target"] = pd.to_numeric(
                chunk["target"], errors="coerce"
            )

            chunk = chunk[
                required + ["target"]
            ].dropna()

            chunk = chunk[
                chunk["target"].isin([0, 1])
            ]

            chunk["CRS_DEP_TIME"] = (
                chunk["CRS_DEP_TIME"].astype(float)
            )

            for col in ["OP_CARRIER", "ORIGIN", "DEST"]:
                chunk[col] = chunk[col].astype(str).str.strip()

            remaining = MAX_ROWS_PER_FILE - kept

            if remaining <= 0:
                break

            chunk = chunk.head(remaining)

            if not chunk.empty:
                chunk.to_csv(
                    OUTPUT,
                    mode="a",
                    header=not OUTPUT.exists(),
                    index=False
                )

                kept += len(chunk)
                total += len(chunk)

            print(f"Usable rows collected: {kept:,}")

            if kept >= MAX_ROWS_PER_FILE:
                break

    except Exception as exc:
        print(f"ERROR processing {file.name}: {exc}")
        raise

print("\nDATA PREPARATION COMPLETE")
print(f"Total usable records: {total:,}")
print(f"Saved to: {OUTPUT}")
