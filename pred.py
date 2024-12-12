# load the model 

import pickle
model = pickle.load(open("model_iris.pkl","rb"))

print(model.predict([[1,1,1,0]]))


