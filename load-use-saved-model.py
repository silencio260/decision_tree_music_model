#Make predictions with the saved Model
from sklearn.tree import DecisionTreeClassifier
import joblib

model = joblib.load('music-recommender.joblib')
predictions = model.predict([[13,0]])

print("---Print Prediction Based On Saved Model---")
print(predictions)