from flask import Flask, render_template,redirect, url_for, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

@app.route('/')
def success():
   return render_template('index.html')

@app.route('/index',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
       review = request.form['review']
       loaded_model = tf.keras.models.load_model("E:\Python_ML_DS\Flask\model")
       pred = loaded_model.predict(np.array([(review)]))
       if pred >= 0.5:
           return "Positive"
       else:
           return "Negative"
      
if __name__ == '__main__':
   app.run(debug = True)