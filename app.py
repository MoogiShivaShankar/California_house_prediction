import json
import pickle
import logging
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)

# Load the Model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['GET', 'POST'])
def predict_api():
    app.logger.info("Received a request to /predict_api")
    app.logger.info(f"Request method: {request.method}")

    if request.method == 'POST':
        app.logger.info(f"Request JSON: {request.json}")
        try:
            data = request.json['data']
            app.logger.info(f"Received data: {data}")

            input_array = np.array(list(data.values())).reshape(1, -1)
            app.logger.info(f"Input array shape: {input_array.shape}")

            new_data = scalar.transform(input_array)
            app.logger.info(f"Transformed data shape: {new_data.shape}")

            output = regmodel.predict(new_data)
            app.logger.info(f"Prediction output: {output[0]}")

            return jsonify(float(output[0]))
        except Exception as e:
            app.logger.error(f"An error occurred: {str(e)}")
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"message": "Send a POST request to this endpoint"}), 200
    
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction price is{}".format(output))

if __name__ == "__main__":
    app.run(debug=True)