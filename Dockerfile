# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set a working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Train the model at build time so the image contains the trained artifact
# (This step will fail if the CSV or dependencies are missing.)
RUN python train_model.py || true

# Expose the port the Flask app runs on
EXPOSE 5000

# Run with Gunicorn for production readiness
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app", "--workers", "2"]
