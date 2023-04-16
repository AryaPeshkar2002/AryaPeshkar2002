
from copyreg import pickle
import sklearn as sk
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template , request

data = pd.read_csv("Rainfall_India.csv")
'''data = data.replace('-',0.0)'''
data.to_csv('Rainfall_India.csv')
x = data.drop(['PrecipitationSumInches'], axis=1)
y = data['PrecipitationSumInches']
y = y.values.reshape(-1,1)  #mtlb dekhna he

day_index = 798
days = [i for i in range(y.size)]

app = Flask(__name__)

@app.route('/')
def home():
  return render_template('test.html') 

#@app.route('/predict')
#def predict():

clf = LinearRegression()
clf.fit(x,y)

inp = np.array([[1],[56],[39],[43],[36],[28],[93],[43],[30.13]])
inp = inp.reshape(1,-1)

#pickle.dump(clf , open('model.pkl' , 'wb'))
#model = pickle.load(open('model.pkl' , 'rb'))
 



print('Amount of rainfall:',clf.predict(inp))

