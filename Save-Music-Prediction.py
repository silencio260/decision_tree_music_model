#Model Persistence

import pandas as pd
#from sklearn.externals import joblib
import joblib

#read csv file
music_data = pd.read_csv('music.csv')

print(music_data)


X=music_data.drop(columns=['genre'])
print("----INPUT DATA , INDEPENDENT VARIABLES---")
print(X)

y=music_data['genre']
print("----OUTPUT DATA , DEPENDENT VARIABLES---")
print(y)

from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()

model.fit(X,y)

#You can choose to save the model used with below code

joblib.dump(model, 'music-recommender.joblib') #Save the model
print("----Model Saved Successfully For Persistence---")

#prediction=model.predict([ [21,1] ,[22,0]  ])
#print("---PREDICTION RESULT---")
#print(prediction)

