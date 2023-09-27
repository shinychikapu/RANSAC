# Random Sample Consensus

## Introduction

Interpreting data or data visualization often involves fitting the data set into a model such as least square or linear regression. These procedures required using the entire data set to find optimal parameters for the model. However, the result often is heavily affected by outliers, resulting in parameters that tend to favour fitting all of the data rather than capturing its trend.

## Algorithm

RANSAC follows these basic procedures:

1. Picks a number of points **-n** from the data set to fit the model (usually the minimum points required to construct that model)
2. Fit a **model** to the picked data and calculate the model **error** on the entire data set
3. Data points whose error is lower than some threshold **t** are considered inliers
4. Fit a model on the inliers if the number of inliers found in this iteration is larger than the previous ones
5. Repeat the previous steps **k** times

Pseudo-code:
```
class RANSAC:
    def init(self, model, loss_fun, n, k, t)
        initilize the parameters and define containers 
        for the best fit model and number of inliers

    def fit(self, data):
        randomIDS = random(0, len(data), n) #pick n random points
        maybe_inliers = data[randomIDs, :]
        
        #fit model and calculate loss
        maybe_model = model.fit(data)
        loss = loss_fun(data_ground truth, maybe_model.predict(data))
        
        #get inliers
        inliers = data[loss <  t]
        if(inliers > num_inliers):
            bestFit = model.fit(inliers)
            num_inliers = len(inliers)
            inliers = inliers
```
The fit function will depends on the dimension of your data. RANSAC is applicable to every type of model however modification of the fit function is needed.

## Application

RANSAC has broad application where we might have problems concerning outliers but its most seen application is in image sticthing. Ofter, you have to match feature points between 2 images, however, not every match is accurate. RANSAC helps eliminate false matches by using the homography matrix follows the same procedures as the 1D  or 3D least square method.
