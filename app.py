import numpy as np
from flask import Flask, request, jsonify, render_template 
import pickle

# Creating Path variables


app = Flask(__name__)
model = pickle.load(open('final_model/LightGBM.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
#     For rendering results on HTML GUI
    age=request.form.get('Age')
    gender=request.form.get('Gender')
    location=request.form.get('Location')
    subscription_length=request.form.get('subscription_length')
    Monthly_Bill=request.form.get('Monthly_Bill')
    total_usage=request.form.get('total_usage')    
    
    final_features = [np.array([age,gender,location,subscription_length,Monthly_Bill,total_usage])]
    prediction = model.predict(final_features)
    output = int(prediction[0])
    return render_template('index.html', prediction_text=(output+1))

if __name__ == "__main__":
    app.run(debug=True)