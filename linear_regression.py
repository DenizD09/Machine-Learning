import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('database/deneyimmaas.csv')
deneyim = data['deneyim'].values.reshape(-1,1)
maas = data['maas'].values.reshape(-1,1)

#algorithm
import sklearn.linear_model as lm
reg=lm.LinearRegression()

#data split
import sklearn.model_selection as ms
x_train, x_test, y_train, y_test = ms.train_test_split(deneyim, maas, test_size=1/3, random_state=0 )

#train
reg.fit(x_train, y_train)

#prediction
y_pred=reg.predict(x_test)
print("Deneyimler:",x_test)
print("Tahmin edilen maaslar:",y_pred)

#score
import sklearn.metrics as mt
score = mt.r2_score(y_test, y_pred)*100
print("score:",score)

#input-output
deneyim_giris=input("Deneyim Giriniz:")
maas_tahmini = reg.predict([[int(deneyim_giris)]])
print("Verilmesi gereken maaş:",maas_tahmini)


#graph
plt.scatter(deneyim,maas, color="red")
plt.scatter(x_test,y_pred, color="blue")
plt.show()