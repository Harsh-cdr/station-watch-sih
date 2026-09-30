# Station Watch — AWS Sensor Anomaly Monitor

A demo application that detects anomalous AWS sensor readings with an Isolation Forest and explains anomaly flags with SHAP.

## Project structure

```text
station-watch-sih/
├── backend_api.py
├── dashboard.html
├── shap_explainer_demo.py
├── isolation_forest_edge_model.pkl
├── requirements.txt
├── render.yaml
├── .python-version
├── .gitignore
└── run_demo.bat
```

## What was fixed

- The API now loads the committed model artifact instead of retraining on every startup.
- If the model artifact cannot be loaded because of a local dependency/version mismatch, the API falls back to the deterministic training routine.
- The dashboard is served by FastAPI, so the online demo can use one public URL.
- The dashboard still works when opened directly from `dashboard.html` on the same computer.
- Synthetic training data is deterministic, so repeated runs use the same reference distribution.
- The training script no longer prints thousands of sensor values to the terminal.
- The project includes a `.gitignore` so `tf_env` and Python cache files are not uploaded.

## Run locally on Windows

From the project root:

```powershell
py -3.11 -m venv tf_env
.	f_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn backend_api:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

There is also a convenience script:

```text
run_demo.bat
```

## Test the API

Health check:

```text
GET http://127.0.0.1:8000/health
```

Demo stream:

```text
GET http://127.0.0.1:8000/stream
```

Custom reading:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"temperature\":63,\"pressure\":1011,\"humidity\":59}"
```

## Important note about the model

The committed `.pkl` is a Python/Scikit-learn model artifact. The repository pins the Scikit-learn version used by that artifact (`1.9.1`). Only load this `.pkl` from a trusted source.

The displayed "confidence" is a relative score mapped to 0–100 for the demo; it is not a calibrated probability.

## GitHub

The GitHub repository is the source-code submission. Do **not** upload `tf_env/`.

For SIH, provide:
- the GitHub repository URL for the code;
- the live demo URL after deployment;
- the SIH Idea Presentation PDF in the SIH portal where requested.

## Deployment

This repo includes `render.yaml` for a simple FastAPI web-service deployment.

Build command:

```text
pip install -r requirements.txt
```

Start command:

```text
uvicorn backend_api:app --host 0.0.0.0 --port $PORT
```
