# ✅ Step 3: Build Flask Web App to Deploy the Model
# ---------------------------------------------------
# This script creates a simple web app where users can input property details
# and get predicted price based on the trained ML model.

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("linear_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        area_sqft = int(request.form["area_sqft"])

        input_data = np.array([[bedrooms, bathrooms, area_sqft]])
        prediction = model.predict(input_data)[0]

        return render_template("index.html", prediction_text=f"Estimated Price: ${prediction:.2f}K")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
