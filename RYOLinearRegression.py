from __future__ import division
import numpy as np

xTest = [1,2,3,4,5]
yTest = [2,3,4,5,6]


def getVariance(x):
    n = len(x)
    xbar = sum(x)/len(x)
    num = 0
    for i in x:
        num += (i - xbar)**2
    return num / (len(x)-1)
print 'Home brewed variance calculated as: {}'.format(getVariance(xTest))
print 'Numpy calculated variance as: {}'.format(np.var(xTest, ddof=1))


def getCoVariance(x, y):
    nx = len(x)
    ny = len(y)
    xbar = sum(x)/len(x)
    ybar = sum(y)/len(y)
    num = 0
    if not (nx == ny):
        return 'must be vectors of equal length'
    else:
        for index, i in enumerate(x):
            num += ((i - xbar) * (y[index] - ybar))
    return num / (len(x)-1)
print getCoVariance(xTest, yTest)
print np.cov(xTest, yTest)[0][0]

def getTheta(var, covar):
    return covar/var

print getTheta(getVariance(xTest), getCoVariance(xTest, yTest))

def getAlpha(theta, x, y):
    return (sum(y)/len(y))-theta*(sum(x)/len(x))

varTest = getVariance(xTest)
covarTest = getCoVariance(xTest, yTest)
thetaTest = getTheta(varTest, covarTest)

alphaTest = getAlpha(thetaTest, xTest, yTest)

print 'Alpha Test: {}'.format(alphaTest)

print 'Model h(x) = {0} + {1} * x'.format(alphaTest, thetaTest)

def predictY(alpha, theta, x):
    return alpha + theta * x
print predictY(alphaTest, thetaTest, 6)
