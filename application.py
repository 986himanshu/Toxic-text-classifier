from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from TextClassifier.pipeline.predict_pipeline import PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def check_toxicity():
    if request.method=='GET':
        return render_template('home.html')
    else:
        text = request.form.get('user_text')
        print(text)

        predict_pipeline=PredictPipeline()
        if predict_pipeline.predict(text)>0.5:
            result= "Toxic"
        else:
            result= "Non-Toxic"
        return render_template('home.html',results=result)
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        


