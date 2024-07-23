from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)

# Set the secret key
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
# Load your heart disease data
heart_data = pd.read_csv('/workspaces/Heart_Disease_Prediction/docs/heart_disease_data.csv')

# Split the data into training and test sets
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('heart.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        features = [float(x) for x in request.form.values()]

        # Convert to numpy array and reshape
        features_arr = np.array(features).reshape(1, -1)

        # Predict
        prediction = model.predict(features_arr)
        
        # Store prediction value in session
        session['prediction'] = int(prediction[0])
        
        # Redirect to the result page
        return redirect(url_for('result'))


@app.route('/result')
def result():
    # Get the prediction value from the session
    prediction = session.get('prediction', None)
    print(prediction)
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)