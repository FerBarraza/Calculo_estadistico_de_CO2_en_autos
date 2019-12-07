import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
#%matplotlib inline



df = pd.read_csv("FuelConsumption.csv")

# un vistazo dentro del set de datos
#df.head()
print("Leyendo los datos\n")
print(df.head(50))

# Sumarizar los datos
#df.describe()
print("\nExploración descriptiva de los datos\n")
print(df.describe())

print("\nExploración en detalle de algunos de los datos\n")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(50))

print("\nPresione enter para continuar\n")
aux=input()

print("\nGrafico de los datos detallados anteriormente\n")
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Consumo de nafta")
plt.ylabel("Emisión CO2")
plt.title("Relación de emision de CO2 y consumo de nafta")
print("\nRelación de emision de CO2 y consumo de nafta\n")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Tamaño de motor")
plt.ylabel("Emisión CO2")
plt.title("Relación de emision de CO2 y tamaño de motor")
print("\nRelación de emision de CO2 y tamaño de motor\n")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Cilindros")
plt.ylabel("Emisión CO2")
plt.title("Relación de emision de CO2 y cantidad de cilindros")
print("\nRelación de emision de CO2 y cantidad de cilindros\n")
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#### Entrenar distribución de los datos
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients

print ('\nRegresión lineal simple\n')
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Tamaño de motor")
plt.ylabel("Emisión CO2")
plt.title("Regresión lineal")
plt.show()

#Evaluacion
from sklearn.metrics import r2_score
print ('\nEvaluación final de los resultados\n')
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Error medio absoluto: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Suma residual de los cuadrados (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
