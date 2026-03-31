from flask import Flask, request, jsonify
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os
import json
import gc

# =====================================
# Flask App
# =====================================
app = Flask(__name__)

# =====================================
# Firestore Init
# =====================================
firebase_key = os.environ.get("FIREBASE_KEY")
firebase_dict = json.loads(firebase_key)

cred = credentials.Certificate(firebase_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# =====================================
# Feature Config
# =====================================
SECTOR_FEATURES = {

    "banking": [
        "NIM (%)","Net Interest Income (₹ Cr)","ROA (%)","ROE (%)",
        "Gross NPA (%)","Net NPA (%)","Provision Coverage Ratio (%)",
        "CAR (%)","CASA (%)","P/B Ratio","Cost-to-Income Ratio (%)","Dividend Yield (%)"
    ],

    "it": [
        "EPS","P/E Ratio","ROE","ROCE","Debt-to-Equity Ratio","Current Ratio",
        "Revenue Growth Rate (%)","Operating Margin (%)","FCF Yield (%)",
        "Attrition Rate (%)","Price-to-Sales Ratio","Dividend Yield (%)"
    ],

    "auto": [
        "Volume Growth (%)","EPS","Operating Margin (%)","ROCE (%)","ROE (%)",
        "Asset Turnover Ratio","Inventory Turnover","Debt-to-Equity Ratio",
        "P/E Ratio","Net Profit Margin (%)"
    ],

    "power": [
        "Plant Load Factor (PLF %)","EBITDA Margin (%)","Revenue_per_MW (₹ Cr)",
        "ROE (%)","Debt-to-Equity Ratio","Interest Coverage Ratio",
        "Operating Cash Flow (₹ Cr)","Dividend Yield (%)","P/B Ratio"
    ],

    "real_estate": [
        "Net Debt-to-Equity Ratio","Interest Coverage Ratio","EBITDA Margin (%)",
        "Project Completion Ratio (%)","Sales/Booking Growth (%)",
        "Operating Cash Flow (₹ Cr)","P/B Ratio","ROCE (%)","ROE (%)"
    ],

    "telecom": [
        "ARPU (₹)","Subscriber Growth (%)","EBITDA Margin (%)","Debt-to-Equity Ratio",
        "Capex-to-Sales Ratio (%)","Churn Rate (%)","Operating Margin (%)",
        "ROE (%)","Net Profit Margin (%)"
    ],

    "energy": [
        "EBITDA Margin (%)","Net Profit Margin (%)","ROE (%)","Debt-to-Equity Ratio",
        "Reserves_to_Production Ratio","Dividend Yield (%)","Operational Efficiency Index",
        "P/E Ratio","ROCE (%)"
    ],

    "metals": [
        "EBITDA_per_Ton","Operating Margin (%)","ROCE (%)","Volume Growth (%)",
        "Debt-to-Equity Ratio","Reserve Life Index","Dividend Payout Ratio (%)",
        "Net Profit Margin (%)","EPS"
    ]
}

# =====================================
# SAFE FEATURE HANDLING
# =====================================
def filter_and_order_features(sector, incoming_features):
    ordered = {}

    for feature in SECTOR_FEATURES[sector]:
        value = incoming_features.get(feature, 0)
        try:
            ordered[feature] = float(value)
        except:
            ordered[feature] = 0.0

    return ordered

# =====================================
# LOAD MODEL PER REQUEST (MEMORY SAFE)
# =====================================
def load_model_for_sector(sector):
    return {
        "reg": joblib.load(f"models/{sector}/regression.pkl"),
        "clf": joblib.load(f"models/{sector}/classifier.pkl")
    }

# =====================================
# Confidence Score
# =====================================
def compute_confidence(reg_change, clf_proba):
    magnitude_score = min(abs(reg_change) / 15, 1.0)
    confidence = 0.6 * clf_proba + 0.4 * magnitude_score
    return round(float(confidence), 3)

# =====================================
# Core Prediction Logic
# =====================================
def run_prediction(sector_name, company_name, features, current_price):

    # Load only required model
    models = load_model_for_sector(sector_name)

    model_reg = models["reg"]
    model_clf = models["clf"]

    f = filter_and_order_features(sector_name, features)
    X = pd.DataFrame([f])

    # Regression
    pct_change = float(model_reg.predict(X)[0])
    predicted_price = float(current_price + (current_price * (pct_change / 100)))

    # Classification
    direction = int(model_clf.predict(X)[0])
    proba = float(model_clf.predict_proba(X).max())

    # Confidence
    confidence = compute_confidence(pct_change, proba)

    result = {
        "sector": sector_name,
        "company": company_name,
        "predicted_price": round(predicted_price, 2),
        "predicted_change_percent": round(pct_change, 2),
        "direction": direction,
        "confidence": confidence,
        "timestamp": datetime.utcnow().isoformat(),
        "input_fundamentals": features
    }

    # Save to Firestore
    db.collection("predictions") \
      .document(sector_name) \
      .collection(company_name) \
      .document("results") \
      .set(result)

    # Free memory (IMPORTANT for Render)
    del model_reg
    del model_clf
    del models
    gc.collect()

    return result

# =====================================
# API ROUTE
# =====================================
@app.route("/predict/<sector>", methods=["POST"])
def predict(sector):
    if sector not in SECTOR_FEATURES:
        return jsonify({"error": "Invalid sector"}), 400

    data = request.json
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    try:
        company_name = data.pop("Company")
        current_price = float(data.pop("current_price"))
    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400

    try:
        result = run_prediction(
            sector_name=sector,
            company_name=company_name,
            features=data,
            current_price=current_price
        )
        return jsonify(result)

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

# =====================================
# Health Check
# =====================================
@app.route("/")
def home():
    return jsonify({"status": "API Running Successfully"})

@app.route("/health", methods=["GET"])
def health_check():
    status = {
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }

    try:
        db.collection("health_check").document("test").set({
            "ping": True,
            "time": datetime.utcnow().isoformat()
        })
        status["services"]["firebase"] = "connected"
    except Exception as e:
        status["services"]["firebase"] = str(e)

    return jsonify(status), 200

# =====================================
# Run Server
# =====================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
