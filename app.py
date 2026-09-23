from flask import Flask, jsonify, render_template, request

from model import FEATURES, predict_patient

app = Flask(__name__, template_folder="frontend")


@app.get("/")
def index():
    return render_template("indian-liver-patient-prediction.html", features=FEATURES)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "liver-patient-prediction"})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    missing = [feature for feature in FEATURES if feature not in payload or payload[feature] == ""]
    if missing:
        return jsonify({"error": "Missing required fields", "fields": missing}), 400
    try:
        result = predict_patient(payload)
    except (TypeError, ValueError, KeyError) as error:
        return jsonify({"error": f"Invalid patient data: {error}"}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
    return jsonify(result)


if __name__ == "__main__":
    # Run in non-debug mode by default for a more production-like behaviour.
    app.run(host="127.0.0.1", port=5000, debug=False)
