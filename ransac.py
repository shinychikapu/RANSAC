from copy import copy
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import random

def sse(y, yhat):
    return (yhat - y)**2

def mse(y, yhat):
    return np.sum((yhat - y)**2)/len(y)

class Least_Square():
    def __init__(self):
        self.params = None
    def fit(self, Y, X):
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same length.")
        #y = mx + c = Ap, where A = [[x, 1]] and p = [[m], [c]]
        A = np.vstack([X, np.ones(len(X))]).T
        self.m, self.c = np.linalg.lstsq(A, Y, rcond=None)[0]
        #return the model object
        return self
    
    def fit_plane(self, X, Y, Z):
        if len(X) != len(Y) or len(Z) != len(Y):
            raise ValueError("X, Y, Z must have the same length.")
        
        # aX + bY + c = Z 
        # AX = B, where A = [[X, Y, 1]] and X = [[a], [b], [c]] and B = Z
        A = np.vstack([X, Y, np.ones(len(X))]).T #transpose to get column vector
        B = Z.T
        fit = np.linalg.inv(A.T @ A) @ A.T @ B
        self.a = fit[0]
        self.b = fit[1]
        self.c = fit[2]
        return self
    
    def predict(self, X):
        prediction =  X * self.m + self.c
        return prediction
    
    def predict_plane(self, X, Y):
        prediction = X * self.a + Y * self.b + self.c
        return prediction


class RANSAC():
    def __init__(self,  model, loss_fun, n = 5, k = 500, t = 0.5):
        '''This is the constructor for the ransac object.

        Attribute
        ------------
        model: model object implementing fit and predict
        loss: a loss function to assert model fit
        n : Minimum number of data points to estimate the parameters
        k : maximum iterations allowed
        t : threshold value to determine if point are fit well
        '''

        self.default_inlier_prob = 0.8 #use when k is not provided
        
        self.model = model
        self.loss_fun = loss_fun
        self.n = n
        self.t = t
        self.k = k
        self.bestFit = None
        self.x_inliers = []
        self.y_inliers = []
        self.z_inliers = []
        self.mostInlnliers = 0
        self.bestErr = np.inf
    #line fit
    def fit(self, X, Y):
        for i in range(self.k):
            randomIDs = random.sample(range(0, len(X)), self.n) #get n random IDs
            #sampling random points
            maybeInliersX = np.asarray([X[e] for e in (randomIDs)])
            maybeInliersY = np.asarray([Y[e] for e in (randomIDs)])

            #fit model
            maybeModel = self.model.fit(maybeInliersY, maybeInliersX)

            #fit remaining points to model
            yhat = maybeModel.predict(X) 
            loss = self.loss_fun(Y, yhat)

            #get the inliers
            inliersX = np.asarray([X[e] for e in range(len(loss)) if loss[e] <self.t])
            inliersY = np.asarray([Y[e] for e in range(len(loss)) if loss[e] < self.t])

            #check if have sufficient inliers
            if len(inliersX) > self.mostInlnliers and len(inliersX) >= 2:
                #fit model to inliers
                best = self.model.fit(inliersX, inliersY) 
                best_err = np.sum((best.predict(X) - Y)**2)/len(Y)
                if(best_err < self.bestErr):
                    self.bestFit = best
                    #store the inliers
                    self.x_inliers = inliersX #inliers
                    self.y_inliers = inliersY #inliers
                    self.mostInlnliers = len(inliersX)
                

        return self
    #plane fit
    def fit_3d(self, X, Y, Z):
        for i in range(self.k):

            randomIDs = random.sample(range(0, len(X)), self.n) #get n random IDs

            #sampling random points
            maybeInliersX = np.asarray([X[e] for e in (randomIDs)])
            maybeInliersY = np.asarray([Y[e] for e in (randomIDs)])
            maybeInliersZ = np.asarray([Z[e] for e in (randomIDs)])

            #fit model
            maybeModel = self.model.fit_plane(maybeInliersY, maybeInliersX, maybeInliersZ)
            
            #fit remaining points to model
            zhat = maybeModel.predict_plane(X, Y) 
            loss = self.loss_fun(Z, zhat)

            #get the inliers
            inliersX = np.asarray([X[e] for e in range(len(loss)) if loss[e] < self.t])
            inliersY = np.asarray([Y[e] for e in range(len(loss)) if loss[e] < self.t])
            inliersZ = np.asarray([Z[e] for e in range(len(loss)) if loss[e] < self.t])

            #check if have sufficient inlier
            if len(inliersX) >= self.mostInlnliers and len(inliersX) >= 3:
                self.bestFit = self.model.fit_plane(inliersX, inliersY,inliersZ) 
                #store the inliers
                self.x_inliers = inliersX 
                self.y_inliers = inliersY
                self.z_inliers = inliersZ 
                self.mostInlnliers = len(inliersX)
        
        if (self.bestFit == None):
            print("No model found")   
        return self


# #setting the trend
# a = 1
# b = 2
# c = 3

# #inliers
# x = np.random.normal(0, 100, 25)
# y = np.random.normal(0, 100, 25)
# noises = np.random.normal(0, 5, 25)
# z = a * x + b * y + c + noises

# #outliers
# x_noises = np.random.normal(0,100,25)
# y_noises = np.random.normal(0,100,25)
# z_noises = np.random.normal(0,100,25)

# #full dataset
# x_ = np.append(x, x_noises)
# y_ = np.append(y, y_noises)
# z_ = np.append(z, z_noises)

# #Least square line
# lsq = LinearReg()
# lsq.fit_plane(x_, y_, z_)
# zhat = lsq.predict_plane(x_, y_)

# xx, yy = np.meshgrid(x_, y_)
# zz = lsq.a * xx + lsq.b * yy + lsq.c

# xxx, yyy, zzz = np.meshgrid(x_, y_, z_)

# #real plane
# zreal = a*xx + b*yy + c

# ransacc = RANSAC(model = LinearReg(), 
#                  loss_fun = sse,
#                  metric = mse,
#                  n = 20,
#                  k = 600,
#                  t = 20,
#                  d = 5)
# ransacc.fit_3d(x_, y_, z_)

#Dummy data
np.random.seed(789)

#define slope and intercept
slope = 1
intercept = 0

#get the data with the trend
x_ = np.random.uniform(0,100,75) #75 points
y_ = slope * x_ + intercept
noises = np.random.uniform(0, 5, 75)
y_ = y_ + noises
#generate noises
noise_x = np.random.uniform(0, 100, 25)
noise_y = np.random.uniform(0, 100, 25)
x = np.append(x_, noise_x)
y = np.append(y_, noise_y)
plt.scatter(x, y)
plt.scatter(x_, y_, c = 'r') #inliers
plt.plot(x, slope* x + intercept, 'k-')

sd = np.std(y)

ransac_op = RANSAC(model = Least_Square(), 
                   loss_fun = sse, 
                   k = 2000,
                    n = 2,
                    t = sd
                    )
ransac_mod = ransac_op.fit(x,y)