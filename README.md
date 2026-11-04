# SKYPREDICT — Flight Delay Prediction

Local Streamlit project using historical aviation data and Scikit-learn.

## Structure
- `app.py`: Streamlit dashboard and prediction interface.
- `common.py`: shared model-loading and prediction functions.
- `prepare_data.py`: reads raw CSVs and creates a sampled training file.
- `train_model.py`: trains and evaluates a Logistic Regression pipeline.
- `index.html`: additional HTML section embedded in the dashboard.
- `data/raw/`: put original flight CSV files here.
- `data/processed/`: generated training sample.
- `models/`: generated model and metrics.

## Expected dataset columns
Features: `OP_CARRIER`, `ORIGIN`, `DEST`, `CRS_DEP_TIME`.
Target: `DEP_DEL15` or `DepDelay` (where delay > 15 minutes is class 1). If neither exists, it can fall back to `ARR_DEL15`.

The preparation script also supports the older 2009–2018 Kaggle column names: `UniqueCarrier`, `Origin`, `Dest`, `CRSDepTime`, and `DepDelay`.

## Windows setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Prepare and train
```powershell
python prepare_data.py
python train_model.py
```

The preparation script samples up to 150,000 usable rows per CSV file to keep training manageable. Adjust `MAX_ROWS_PER_FILE` in `prepare_data.py` as needed.

## Run
```powershell
python -m streamlit run app.py
```

No GitHub download is used. The app reads `models/flight_delay_model.joblib` locally.

## Notes
This starter implementation trains Logistic Regression only. It does not reuse or claim previous Random Forest scores. Verify target meaning and class encoding before presenting results. Predictions are estimates, not live flight-status information.
