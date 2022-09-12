import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data
data = pd.read_csv("database/breastcancer.csv")
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

#to see clustering

plt.scatter(M.radius_mean, M.texture_mean, color="r", label="M", alpha=0.3)
plt.scatter(B.radius_mean, B.texture_mean, color="g", label="B", alpha=0.3)
plt.legend()
plt.show()

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
x_data = data.drop(["diagnosis"],axis=1)
y = data.diagnosis.values

#normalization
x = (x_data-np.min(x_data))/ (np.max(x_data)-np.min(x_data))

#train test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

#train
dt.fit(x_train, y_train)

#predict
y_pred=dt.predict(x_test)

#score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#graph
