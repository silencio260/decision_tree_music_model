#Test Prediction Accuracy Score
import pandas as pd

music_data=pd.read_csv('music.csv')
print("---THE DATASET----")
print(music_data);

#Create Input Set and Output Set

#return columns without genre
X=music_data.drop(columns=['genre'])

#input Set
print("-----INPUT SET----")
print(X)

#Create output set which is the genre
y=music_data['genre']
from sklearn.model_selection import train_test_split

#Alocate 20% of Data for Testing thererfore 80% for trainng
#reducing amount of Training Data will reduce accuracy
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2) #returns a tuple , so it can be unpacked into 4 variables

#To train a model Add
  

print("-----OUTPUT SET----")
print(y)

#Build our Model Using Decission Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
#Fit Our Data Into the Model
#i.e , the input and Output Set
model.fit(X_train,y_train)

#Ask Our Model To Make a Prediction
#Predict the genre of music for a 21 year old male and a 22 year old Female
predictions=model.predict( X_test)

from sklearn.metrics import accuracy_score
#To test the accuracy compare it with the actual values
score=accuracy_score(y_test, predictions)
score=score * 100
print("----Model Accuracy Score-----")
print("Model Is "+str(score)+"% Accurate")
