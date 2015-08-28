import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X_train = [[6], [8], [10], [14],   [18]]
y_train = [[7], [9], [13], [17.5], [18]]

X_test = [[6],  [8],   [11], [16]]
y_test = [[8], [12], [15], [18]]

regressor = LinearRegression()
regressor.fit(X_train, y_train)

regressor.score(X_train, y_train)
prediction = regressor.predict(X_test)

print 'Simple Linear Regression:\n'
for i, item in enumerate(prediction):
    print 'Prediction: {0} Actual {1}'.format(item, y_test[i])

xx = np.linspace(0,26,100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))

plt.plot(xx, yy)

quadraticFeaturizer = PolynomialFeatures(degree=4)

x_train_quad = quadraticFeaturizer.fit_transform(X_train)
x_test_quad = quadraticFeaturizer.transform(X_test)

regressor_quad = LinearRegression()
regressor_quad.fit(x_train_quad, y_train)

xx_quad = quadraticFeaturizer.transform(xx.reshape(xx.shape[0], 1))

print 'Rsq: {}'.format(regressor_quad.score(x_train_quad, y_train))

plt.plot(xx, regressor_quad.predict(xx_quad), c='r', linestyle='--')
plt.axis([0,25,0,25])
plt.grid = True
plt.scatter(X_train, y_train)
plt.show()
