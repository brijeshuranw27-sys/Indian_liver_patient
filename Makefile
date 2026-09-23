PY=python
PIP=pip
VENV=.venv

.PHONY: venv install train run test report docker-build docker-up clean

venv:
	$(PY) -m venv $(VENV)
	@echo "Activate with: .\\$(VENV)\\Scripts\\Activate.ps1 (PowerShell)"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

train:
	$(PY) train_model.py

run:
	$(PY) app.py

test:
	pytest -q

report:
	$(PY) generate_report.py

# Docker helpers
docker-build:
	docker build -t liver-patient-pred .

docker-up:
	docker-compose up --build

clean:
	rm -rf __pycache__ $(VENV) report_images prediction_report.docx liver_patient_model.pkl
