import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

model = Sequential()
model.add(Dense(5, activation='linear', input_shape=(8,) ))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

model.load_weights("../Models/100-0.4767-0.7773.hdf5")

def DiabetesPrediction():
    Pregnancies = request.args.get("Pregnancies")
    Glucose = request.args.get("Glucose")
    BloodPressure = request.args.get("BloodPressure")
    SkinThickness = request.args.get("SkinThickness")
    Insulin = request.args.get("Insulin")
    BMI = request.args.get("BMI")
    DiabetesPedigreeFunction = request.args.get("DiabetesPedigreeFunction")
    Age = request.args.get("Age")
       
    if Pregnancies == None or Glucose == None:
        return render_template('Diabetes.html', Output = '')
    
    Input = pd.DataFrame({
        'Pregnancies': [ float(Pregnancies) ],
        'Glucose': [ float(Glucose) ],
        'BloodPressure': [ float(BloodPressure) ],
        'SkinThickness': [ float(SkinThickness) ],
        'Insulin': [ float(Insulin) ],
        'BMI': [ float(BMI) ],
        'DiabetesPedigreeFunction': [ float(DiabetesPedigreeFunction) ],
        'Age': [float(Age)]
    })
    ModelOutput = model.predict(Input)[0][0]

    return render_template('Diabetes.html', Output = ModelOutput)