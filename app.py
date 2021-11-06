from flask import Flask, render_template, request, session, url_for, redirect
from flask.views import MethodView
import random
import pickle

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pytz

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heart_failure.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), unique=True, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    anaemia = db.Column(db.Integer, nullable=False)
    creatinine_phosphokinase = db.Column(db.Integer, nullable=False)
    diabetes = db.Column(db.Integer, nullable=False)
    ejection_fraction = db.Column(db.Integer, nullable=False)
    high_blood_pressure = db.Column(db.Integer, nullable=False)
    platelets = db.Column(db.Float(asdecimal=True), nullable=False)
    serum_creatinine = db.Column(db.Float(asdecimal=True), nullable=False)
    serum_sodium = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.Integer, nullable=False)
    smoking = db.Column(db.Integer, nullable=True)
    heart_failure = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now(pytz.timezone('Asia/Tokyo')))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        posts = User.query.all()
        return render_template('index.html', title='心臓病予測アプリ', message='あなたの血液情報から心臓病に罹患するか診断します。')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        msg = '心臓病で死亡するか予測するために各種情報を入力してください'
        return render_template('predict.html', title='心臓病予測', message=msg)

    if request.method == 'POST':
        username="username" + str(datetime.now())
        reg = pickle.load(open('./model/decision_tree_model.pkl', 'rb'))
        x1 = request.form['age']
        anaemia = request.form.get('anaemia')
        if anaemia == "はい":
            x2 = 1
        else:
            x2 = 0
        x3 = request.form['creatinine_phosphokinase']
        diabetes = request.form.get('diabetes')
        if diabetes == "はい":
            x4 = 1
        else:
            x4 = 0
        x5 = request.form['ejection_fraction']
        high_blood_pressure = request.form.get('high_blood_pressure')
        if high_blood_pressure == "はい":
            x6 = 1
        else:
            x6 = 0
        x7 = request.form['platelets']
        x8 = request.form['serum_creatinine']
        x9 = request.form['serum_sodium']
        sex = request.form.get('sex')
        if sex == "男":
            x10 = 1
        else:
            x10 = 0
        smoking = request.form.get('smoking')
        if smoking == "はい":
            x11 = 1
        else:
            x11 = 0

        x = [[int(x1), x2, int(x3), x4, int(x5), float(x6), float(x7) ,float(x8) ,int(x9), x10, x11]]
        heart_failure = reg.predict(x)
        if heart_failure == 1:
            result = "死亡します"
        else:
            result = "死亡しません"
        heart_failure = 'あなたは心臓病で{}。'.format(result)

        post = User(username=username, age=x1, anaemia=anaemia, creatinine_phosphokinase=x3, diabetes=diabetes, ejection_fraction=x5, 
                    high_blood_pressure=high_blood_pressure, platelets=x7, serum_creatinine=x8, serum_sodium=x9, sex=sex, smoking=smoking, heart_failure=result)
        db.session.add(post)
        db.session.commit()

        return render_template('result.html', title='予測結果', message=heart_failure)

@app.template_filter('sum')
def sum_filter(data):
    total = 0
    for item in data:
        total += item
    return total
    
app.jinja_env.filters['list_sum'] = sum_filter

@app.context_processor
def sample_processor():
    def total(n):
        total = 0
        for i in range(n + 1):
            total += i
        return total
    return dict( total = total )

@app.route('/result', methods=['GET'])
def result():
    return render_template('result.html', title='Result Page', message='これは次のページのサンプルです', data=['A','B','C'])

app.secret_key = b'asdfghjkl'

if __name__ == '__main__':
    app.run(host='localhost', debug=True)