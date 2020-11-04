import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [x for x in request.form.values()]
    # sex encoding
    sex_enc = {'Male': 1, 'Female': 0}
    input_pred = pd.DataFrame({'Gender': sex_enc[features[0]], 'Height': int(features[1]), 'Weight': int(features[2])}, index=[0])
    poly = PolynomialFeatures(2, include_bias=False)

    input_pred_enriched = pd.concat((pd.DataFrame(poly.fit_transform(input_pred[['Height', 'Weight']]),
                                                  columns=['Height', 'Weight', 'Height_2', 'Weight_2', 'interaction']),
                                     input_pred[['Gender']]), axis=1)

    preds_proba = model.predict_proba(input_pred_enriched)

    output_class_prob = round(preds_proba.max(axis=1)[0], 2) * 100

    output_class = preds_proba.argmax(axis=1)[0]

    output_descr = {0 : 'Extremely Weak',
                    1 : 'Weak',
                    2 : 'Normal',
                    3 : 'Overweight',
                    4 :'Obese',
                    5 : 'Extreme Obese'}

    return render_template('index.html', prediction_text='Your BMI is {} which means that you are {} with a probability of {} %'.format(output_class,
                                                                                                             output_descr[output_class],
                                                                                                             output_class_prob))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)

    # sex encoding
    sex_enc = {'Male': 1, 'Female': 0}
    input_pred = pd.DataFrame({'Gender': sex_enc[data['Gender']], 'Height': int(data['Height']), 'Weight': int(data['Weight'])},
                              index=[0])
    poly = PolynomialFeatures(2, include_bias=False)

    input_pred_enriched = pd.concat((pd.DataFrame(poly.fit_transform(input_pred[['Height', 'Weight']]),
                                                  columns=['Height', 'Weight', 'Height_2', 'Weight_2', 'interaction']),
                                     input_pred[['Gender']]), axis=1)

    preds_proba = model.predict_proba(input_pred_enriched)

    output_class_prob = round(preds_proba.max(axis=1)[0], 2) * 100

    output_class = preds_proba.argmax(axis=1)

    return jsonify(BMI=output_class.tolist())

if __name__ == "__main__":
    app.run(debug=True)