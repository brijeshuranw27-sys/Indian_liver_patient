# Liver Patient Prediction

A compact, professional Flask prototype that demonstrates a reproducible scikit-learn training pipeline, an HTTP prediction API, a browser UI, and a simple report generator. This repository is intended for educational and exploratory use only — not for clinical diagnosis.

Prerequisites
- Python 3.10+ (recommended). The code uses modern typing and idioms; please ensure a recent Python runtime is used.
- On Windows, use PowerShell to run the following commands.

Quick start (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python train_model.py   # trains a model and saves liver_patient_model.pkl
python app.py           # starts the prediction service on http://127.0.0.1:5000
```

Open http://127.0.0.1:5000/ in a browser. API endpoints:
- GET  /api/health  — service health check
- POST /api/predict — accepts JSON with the features listed on the homepage and returns prediction and probability

Generating the report
- Run `python generate_report.py` to create `prediction_report.docx` and the `report_images/` charts.

Project map
- `liver_patient.csv`: example training data included for demonstration.
- `train_model.py`: trains and persists `liver_patient_model.pkl`.
- `model.py`: feature contract and prediction helper (used by the API).
- `app.py`: Flask web server and JSON API, serving the UI from `frontend/`.
- `frontend/`: HTML/Jinja template for the prediction interface.
- `generate_report.py`: creates a simple Word report and charts.
- `untitled.ipynb`: notebook walkthrough starter.

Notes
- The dataset and model are only for demonstration. Do not use the predictions for medical decisions.
- If `python train_model.py` fails due to missing columns or unexpected values, inspect `liver_patient.csv` and ensure the column headers match the expected features listed in `model.py`.

Development & CI

This repository includes helper files to make development, testing, and CI straightforward:

- `requirements-dev.txt` — development dependencies (pytest, flake8, black).
- `tests/` — unit tests for the model helper and the Flask app.
- `.github/workflows/ci.yml` — a GitHub Actions workflow that installs dependencies and runs pytest on push/pull requests.
- `Dockerfile` and `.dockerignore` — build a container image that installs dependencies and (optionally) trains the model during image build.
- `Makefile` — convenience targets: `make install`, `make train`, `make test`, `make report`, `make run`.

Run tests locally
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q
```

Docker

Build and run locally (Linux/macOS or Docker Desktop on Windows):
```bash
docker build -t liver-patient-pred .
docker run -p 5000:5000 --rm liver-patient-pred
```

CI

A GitHub Actions workflow is provided at `.github/workflows/ci.yml` and will run tests on pushes and pull requests to `main`/`master`.

If you want, next steps can include adding more unit coverage, a Docker Compose file for multi-service setups, a basic UI test, or a small dataset validation script.