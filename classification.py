import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#data
data = pd.read_csv('database/diabetes.csv')
#data.info(verbose=True)
#print(data.describe())

data_copy = data.copy(deep=True)
#print(data_copy.isnull().sum())
data_copy[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = data_copy[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)
#print(data_copy.isnull().sum())
data_copy["Glucose"].fillna(data_copy["Glucose"].mean(), inplace = True)
data_copy["BloodPressure"].fillna(data_copy["BloodPressure"].mean(), inplace = True)
data_copy["SkinThickness"].fillna(data_copy["SkinThickness"].median(), inplace = True)
data_copy["Insulin"].fillna(data_copy["Insulin"].median(), inplace = True)
data_copy["BMI"].fillna(data_copy["BMI"].median(), inplace = True)

#ikili ilişkiler
#sns.pairplot(data_copy, hue="Outcome")
#plt.show()

#korelasyon için heatmap
#plt.figure(figsize=(12,10))
#p=sns.heatmap(data_copy.corr(), annot=True, cmap="RdYlGn")
#plt.show()

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

#get x,y
x = pd.DataFrame(sc_x.fit_transform(data_copy.drop(["Outcome"], axis=1)),
                 columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"])
y=data_copy.Outcome

#train & test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state = 0, stratify=y)

#model
from sklearn.neighbors import KNeighborsClassifier
test_score = []
train_score = []
for i in range (1,15):
    knn = KNeighborsClassifier(i)
    knn.fit(x_train, y_train)
    train_score.append(knn.score(x_train,y_train))
    test_score.append(knn.score(x_test, y_test))
#max scores
max_train_score = max(train_score)
train_score_ind = [i for i, v in enumerate(train_score) if v == max_train_score]
print("Max train score:{} & K = {}".format(max_train_score*100, list(map(lambda x: x+1,train_score_ind))))

max_test_score = max(test_score)
test_score_ind = [i for i, v in enumerate(test_score) if v == max_test_score]
print("Max test score:{} & K = {}".format(max_test_score*100, list(map(lambda x: x+1,test_score_ind))))

#graph to find K value
""""
plt.figure(figsize=(12,  5))
p = sns.lineplot(train_score, marker="*", label= "Train Score")
p = sns.lineplot(test_score, marker="o", label= "Test Score")
plt.show()
"""

#train for K=11
knn = KNeighborsClassifier(11)
knn.fit(x_train, y_train)
score=knn.score(x_test, y_test)
print("Score with K=11: {}".format(score))

#conf matrix
from sklearn.metrics import confusion_matrix
y_pred = knn.predict(x_test)
confusion_matrix(y_test, y_pred)
ct = pd.crosstab(y_test, y_pred, rownames=["True"], colnames=["Predicted"],margins_name=True)
print(ct)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))