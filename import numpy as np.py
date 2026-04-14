import numpy as np
import pickle
from flask import Flask, request, render_template

# Load the trained model
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Predict eligibility
    prediction = model.predict(final_features)
    
    # Return result to the user
    result = "Eligible" if prediction[0] == 1 else "Not Eligible"
    
    return render_template('index.html', prediction_text=f'Scholarship Status: {result}')

if __name__ == "__main__":
    app.run(debug=True)
