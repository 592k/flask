# import Libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


# App Initialization the flask app
app = Flask(__name__) #Initialize the flask App
# model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')
    # str_var = 'Hello Deep Learning'
    #return render_template('index.html', testing = str_var)
    # testing -> var in HTML

if __name__ == "__main__":
    app.run(debug=True)
