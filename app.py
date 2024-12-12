from flask import Flask, render_template, request
import pickle


app = Flask(__name__)

@app.route('/')
def welcome():

    # return "Thank you visiting the iris model"
     return render_template('home.html')

# http://127.0.0.1:5000/predict?sepal_len=1&sepal_wid=1&petal_len=1&petal_wid=1

@app.route('/predict', methods =['post','get'])
def predict(): 
      print(" i am inside predict")
      sep_len = request.form.get('sepal_len')
      sep_wid = request.form.get('sepal_wid')
      petal_len = request.form.get('petal_len')
      petal_wid = request.form.get('petal_wid')

      model = pickle.load(open('model_iris.pkl', 'rb'))
      y_predict= model.predict([[sep_len,sep_wid,petal_len,petal_wid]])
      if y_predict==[0]:
            data ="setosa"
      elif y_predict==[1]:
            data ="versicolor"
      else:
            data="verginica"            

      print(data)
      return render_template('prediction.html', data="flower is  "+data)
     
     
     
     




app.run(debug= True)