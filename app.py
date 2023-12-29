from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained machine failure prediction model
model_failure = load_model('machine_failure_prediction_model.h5')

# Load the trained failure type prediction model
model_type = load_model('machine_failure_type_prediction_model.h5')

# Define a function to preprocess the input data
def preprocess_input(temperature, process_temperature, torque, tool_wear):

    temp_diff = temperature - process_temperature
    
    # Return the preprocessed input as a NumPy array
    return np.array([[temperature, process_temperature, torque, tool_wear,temp_diff ]])

# Define a function to make predictions
def predict_failure(input_data):
    # Use the machine failure prediction model to make predictions
    prediction = model_failure.predict(input_data)

    predicted_class = np.argmax(prediction)
    # Return the prediction
    return predicted_class

def predict_type(input_data):
    # Use the failure type prediction model to make predictions
    prediction = model_type.predict(input_data)
    predicted_class = np.argmax(prediction)
    # Return the prediction
    return predicted_class

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input values from the form
    temperature = float(request.form['temperature'])
    process_temperature = float(request.form['process_temperature'])
    torque = float(request.form['torque'])
    tool_wear = float(request.form['tool_wear'])
    
    # Preprocess the input data
    input_data = preprocess_input(temperature, process_temperature, torque, tool_wear)
    
    # Make predictions
    failure_prediction = predict_failure(input_data)
    type_prediction = predict_type(input_data)
    
    # Render the results template with the predictions
    return render_template('index.html', failure_prediction=failure_prediction, type_prediction=type_prediction)

if __name__ == '__main__':
    # Feature scaling 
    app.run(debug=True)
