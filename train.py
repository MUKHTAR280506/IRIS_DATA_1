# Load required libraries

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the Iris data set from scikit learn
df1 = datasets.load_iris()
dataframe = pd.DataFrame(df1.data, columns= df1.feature_names)
dataframe["target"] = df1.target

# Split the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(dataframe.drop("target", axis=1),dataframe["target"], test_size=0.2, random_state =87)


# Initialize a model 
model =  RandomForestClassifier()

# Train the model
model.fit(x_train,y_train)

# Make prediction from model
y_predict = model.predict(x_test)

# print the accuracy and classification report
print("Accuracy score of the model :", accuracy_score(y_test,y_predict))
print("Classification report of the model\n", classification_report(y_test, y_predict))

# saving the model
pickle.dump(model, open("model_iris.pkl", "wb"))








