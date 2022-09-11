import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('database/arabafiyathız.csv')
x = data['fiyat'].values.reshape(-1,1)
y = data['hiz'].values.reshape(-1,1)

#algorithm
import sklearn.linear_model as lm
from sklearn.preprocessing import PolynomialFeatures
reg=lm.LinearRegression()
polynomial_reg = PolynomialFeatures(degree=5)
x_polynomial = polynomial_reg.fit_transform(x,y)

#fit
reg.fit(x_polynomial, y)

#prediction
y_pred =reg.predict(x_polynomial)

#score
from sklearn.metrics import r2_score
score= r2_score(y,y_pred)*100
print("score:",score)

#input-output
fiyatal=input("Arabanın fiyatı:")
x_polynomial_pred = polynomial_reg.fit_transform([[int(fiyatal)]],y[0])
y_pred2 = reg.predict(x_polynomial_pred)
print("Tahmini max hız:",y_pred2)

#graph
plt.plot(x, y_pred, color="blue", label="poly")
plt.legend()
plt.scatter(x,y, color="red")
plt.xlabel("Araç Fiyat")
plt.ylabel("Maximum Hız")
plt.show()


