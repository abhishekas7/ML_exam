import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as mtp

import pandas as pd

data = load_iris()
a = data.data
b = data.target

# dataSet = pd.read_csv("data.csv")
# c = dataSet.iloc[[2,1]]
# a=np.array(c)
# print(c)
# d = dataSet.iloc[:,[1,2]]
# b=np.array(d)
# print(d)


x_train, x_test, y_train, y_test = train_test_split(a, b,  test_size=0.3,random_state=42)

model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)

predict = model.predict(x_test)
print(predict)

accuracy = accuracy_score(y_test, predict)
print("Accuracy of the algorithm", accuracy*100)
mtp.figure(figsize=[10,10])
tree.plot_tree(model,filled=True)
mtp.show()
