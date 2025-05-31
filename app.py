from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        fever = int(request.form['fever'])
        cough = int(request.form['cough'])
        fatigue = int(request.form['fatigue'])
        headache = int(request.form['headache'])

        data = np.array([[fever, cough, fatigue, headache]])
        prediction = model.predict(data)[0]
        return render_template("index.html", prediction_text=f"Predicted Disease: {prediction}")
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
