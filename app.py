from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import os

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH_c = 'model_cnn_Covid19.h5'


# Load your trained model
model_c = load_model(MODEL_PATH_c)



def model_predict1(img_path, model_c):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)

    ## Scaling

    x = np.expand_dims(x, axis=0)

    res = model_c.predict(x)
    res = np.argmax(res, axis=1)
    if res == 0:
        res = "Affected"
    else:
        res = "Not Affected"

    return res



@app.route("/")
def hello():
    return render_template('about.html')


@app.route("/paper", methods=['GET'])
def download_file():
    path = "Disease Detection using Machine and Deep Learning Models.pdf"

    return send_file(path, as_attachment=True)

@app.route("/images", methods=['GET'])
def download_images():
    path = "Disease.zip"

    return send_file(path, as_attachment=True)

@app.route("/Heart", methods=['GET', 'POST'])
def Heart():
    return render_template("heart.html")


@app.route("/Heart/Heart_res", methods=['GET', 'POST'])
def Heart_res():
    global res
    temp_array = list()

    if request.method == 'POST':

        age = int(request.form['age'])
        blood_pressure = int(request.form['blood_pressure'])
        cholesterol = int(request.form['cholesterol'])
        max_HR = int(request.form['max_HR'])
        ST_depression = float(request.form['ST_depression'])

        chest_pain = request.form['chest_pain']
        if chest_pain == "2":
            temp_array = temp_array + [1, 0, 0]
        elif chest_pain == "3":
            temp_array = temp_array + [0, 1, 0]
        elif chest_pain == "4":
            temp_array = temp_array + [0, 0, 1]

        ECG_result = request.form['ECG_result']
        if ECG_result == "0":
            temp_array = temp_array + [1, 0, 0]
        elif ECG_result == "1":
            temp_array = temp_array + [0, 1, 0]
        elif ECG_result == "2":
            temp_array = temp_array + [0, 0, 1]

        exercise_angina = request.form['exercise_angina']
        if exercise_angina == "0":
            temp_array = temp_array + [1, 0]
        elif exercise_angina == "1":
            temp_array = temp_array + [0, 1]

        slope_of_ST = request.form['slope_of_ST']
        if slope_of_ST == "1":
            temp_array = temp_array + [1, 0, 0]
        elif slope_of_ST == "2":
            temp_array = temp_array + [0, 1, 0]
        elif slope_of_ST == "3":
            temp_array = temp_array + [0, 0, 1]

        temp_array = [age, blood_pressure, cholesterol, max_HR, ST_depression] + temp_array

        data = np.array([temp_array])

        # Loading the Ridge Regression model
        filename = 'rf_classifier_heart.pkl'
        classifier = pickle.load(open(filename, 'rb'))

        # Predicting output
        model_prediction = classifier.predict(data)

        if model_prediction == 1:
            res = 'Affected'
        else:
            res = 'Not affected'

        # return res
    return render_template('result.html', prediction=res, this='Heart')


@app.route("/Diabetics")
def Diabetics():
    return render_template("diabetics.html")


@app.route("/Diabetics/Diabetics_res", methods=['GET', 'POST'])
def Diabetics_res():
    global res
    if request.method == 'POST':

        Age = int(request.form['Age'])
        BMI = float(request.form['BMI'])
        BloodPressure = int(request.form['BloodPressure'])
        Insulin = int(request.form['Insulin'])
        Glucose = int(request.form['Glucose'])
        SkinThickness = int(request.form['SkinThickness'])
        Pregnancies = int(request.form['Pregnancies'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])

        pred_args = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

        # Loading the Ridge Regression model
        filename = 'lr_classifier_diabetics.pkl'
        classifier = pickle.load(open(filename, 'rb'))
        model_prediction = classifier.predict([pred_args])

        if model_prediction == 1:
            res = 'Affected'
        else:
            res = 'Not affected'
        # return res
    return render_template('result.html', prediction=res)



@app.route("/Covid19")
def Covid19():
    return render_template("covid19.html")


@app.route("/Covid19/Covid19_res", methods=['GET', 'POST'])
def Covid19_res():
    global res2
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'upload', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        res2 = model_predict1(file_path, model_c)

    return render_template("result.html", prediction=res2, this='Covid19')



if __name__ == "__main__":
    app.run(debug=True)
