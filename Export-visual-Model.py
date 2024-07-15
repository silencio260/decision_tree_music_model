#Export our model in Visual Format

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

music_data = pd.read_csv('music.csv')
X=music_data.drop(columns=['genre'])
y=music_data['genre']

model = DecisionTreeClassifier()
model.fit(X,y)

#.dot is a graph description Language
#Textual Language for describing Graphs
tree.export_graphviz(model, out_file='music-recommender.dot',
                    feature_names=['age','gender'] , 
                    class_names=sorted(y.unique()) ,
                    label='all' ,
                    rounded=True,
                    filled=True)
                    
#View .dot file with vscode and search for dot in extension
#To download plugin for viewing