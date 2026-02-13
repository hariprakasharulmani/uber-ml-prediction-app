import numpy as np
from flask import Flask,request, render_template
import pickle
import math

app = Flask(__name__)
model2 = pickle.load(open('pickle.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_feature = [int(i) for i in request.form.values()]
    final_feature = np.array(int_feature).reshape(1,-1)
    prediction = model2.predict(final_feature)
    output= round(prediction[0],2)
    return render_template('index.html', predict_text = 'Number of weekly rides {}'.format(math.floor(output)))

if __name__ == '__main__':
    app.run()
