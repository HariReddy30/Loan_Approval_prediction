
# Import required libraries
from flask import Flask, render_template, request
import pandas as pd
import joblib

# Create Flask application
app = Flask(__name__)

# Load the trained Machine Learning model
model = joblib.load("model.pkl")

# Load the saved StandardScaler
scaler = joblib.load("scaler.pkl")

# Load the saved LabelEncoder
label_encoder = joblib.load("label_encoder.pkl")


# STEP 1: Display the homepage
@app.route("/")
def home():
    return render_template("index.html")


# STEP 2: Receive applicant details
@app.route("/predict", methods=["POST"])
def predict():

     # Read and validate user input
    try:
        income = float(request.form["income"])
        cibil = int(request.form["cibil"])
        age = int(request.form["age"])
        amount = float(request.form["amount"])

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            error="Please enter valid applicant details."
        )

    # STEP 3: Validate applicant details
    if income <= 0 or amount <= 0:
        return "Annual Income and Loan Amount must be positive."

    if cibil < 300 or cibil > 900:
        return "CIBIL Score must be between 300 and 900."

    if age < 0:
        return "Account Age cannot be negative."

    # STEP 4: Create a DataFrame using applicant details
    new_data = pd.DataFrame({
        "Annual_Income": [income],
        "CIBIL_Score": [cibil],
        "Account_Age": [age],
        "Loan_Amount": [amount]
    })

    # STEP 5: Scale the input values
    scaled_data = scaler.transform(new_data)

    # STEP 6: Predict loan status
    prediction = model.predict(scaled_data)

    # STEP 7: Convert numerical prediction into text
    result = label_encoder.inverse_transform(prediction)[0]

    # STEP 8: Display prediction and retain entered values
    return render_template(
        "index.html",
        prediction=result,
        income=income,
        cibil=cibil,
        age=age,
        amount=amount
    )


# Start the Flask application
if __name__ == "__main__":
    app.run(debug=True)
