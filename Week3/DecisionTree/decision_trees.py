"""
Based on the Coursera IBM Machine Learning with Python course, Lab. Decision Trees.

LM: July 19th, 2019.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree


df = pd.read_csv("/home/yshiba/Downloads/drug200.csv", sep=",")
print(df[0:5])
print(df.size)


# Pre-processing
feature_matrix = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(feature_matrix[0:5])

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
feature_matrix[:,1] = le_sex.transform(feature_matrix[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
feature_matrix[:,2] = le_BP.transform(feature_matrix[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
feature_matrix[:,3] = le_Chol.transform(feature_matrix[:,3])

print(feature_matrix[0:5])

response_vector = df["Drug"]
print(df[0:5])


# Setting up the Decision Tree
# X = feature_matrix, y = response_vector
X_trainset, X_testset, y_trainset, y_testset = train_test_split(feature_matrix, response_vector, test_size=0.3, random_state=3)


# Printing the shape. Ensure that the dimensions match for each of the two cases.
print(X_trainset.shape, y_trainset.shape)
print(X_testset.shape, y_testset.shape)


# Modeling
# Creating an instance of the DecisionTreeClassifier called drugTree
# specifying criterion="entropy" to see the information gain of each node.
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(drugTree)  # shows the default parameters

# fitting the data with the training feature matrix X_trainset and training response vector y_trainset
drugTree.fit(X_trainset, y_trainset)


# Prediction
# Making predictions on the testing dataset and storing it into a variable called predTree
predTree = drugTree.predict(X_testset)
# printing out predTreee and y_testset to visually compare the predictions to the actual values
print(predTree [0:5])
print(y_testset [0:5])


# Evaluation
print("DecisionTree's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

# "Accuracy classification score" computes subset accuracy: the set of labels predicted for a sample must
# exactly match the corresponding set of labels in y_true.
# In multilabel classification, the fn returns the subset accuracy. If the entire set of predicted labels
# for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise, it's 0.0

# Calculating the accuracy score without sklearn


# Visualization
dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out = tree.export_graphviz(drugTree, feature_names=featureNames, out_file=dot_data, class_names=np.unique(y_trainset), filled=True, special_characters=True, rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')
plt.show()