import numpy as np
import pandas as pd
import os.path
from random import randint

# -*- coding: utf-8 -*-
"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

#Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'


# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID':'int', 'year':'int', 'movie':'str'}, names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', dtype={'userID':'int', 'gender':'str', 'age':'int', 'profession':'int'}, names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', dtype={'userID':'int', 'movieID':'int', 'rating':'int'}, names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)

#####
##
## COLLABORATIVE FILTERING
##
#####


def np_pearson_cor(x, y):
    if x.sum() == 0 or y.sum() == 0: return 0
    #xv = x - x.mean(axis=0)
    #yv = y - y.mean(axis=0)
    #return 0
    xv = np.copy(x)
    yv = np.copy(y)
    xv[xv==0]=np.nan
    xv[xv!=np.nan]=xv-np.nanmean(xv, axis=0)
    xv[xv==np.nan]=0
    xmean = 0
    ymean = 0
    length = 0
    for val in x:
        if val != 0:
            xmean += val
            length += 1
    if length > 0:
        xmean = xmean / length
    else:
        xmean = 0
    length = 0
    for val in y:
        if val != 0:
            ymean += val
            length += 1
    if length > 0:
        ymean = ymean / length
    else:
        ymean = 0
    for i in range(0, xv.shape[0]):
        if xv[i] != 0:
            xv[i] = xv[i] - xmean
    for i in range(0, yv.shape[0]):
        if yv[i] != 0:
            yv[i] = yv[i] - ymean
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    if np.outer(xvss, yvss) != 0:
        result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    else:
        return 0
    # bound the values to -1 to 1 in the event of precision issues
    return np.minimum(np.abs(result[0][0]), 1.0)


def predict_collaborative_filtering(movies, users, ratings, predictions):

    utilityMatrix = np.zeros((movies.shape[0]+1, users.shape[0]+1))
    movieMeanVector = np.zeros(movies['movieID'].shape[0]+1)

    for row in ratings[['userID', 'movieID', 'rating']].to_numpy():
        utilityMatrix[row[1]][row[0]] = row[2]

    k = 0
    """
    for row in utilityMatrix:
        s = 0
        length = 0
        for rating in row:
            s += rating
            if rating > 0: length+=1
        if length > 0: movieMeanVector[k]=s/length
        k+=1
    """
    for i in range(0, len(utilityMatrix)):
        row = utilityMatrix[i]
        if (np.sum(row) == 0): continue
        pearsonCor = []
        for j in range(i+1, len(utilityMatrix)):
            row2 = utilityMatrix[j]
            pearson = 0
            """
            if row2.sum() != 0:
                xNormalized = row - np.mean(row, axis=0)
                yNormalized = row2 - np.mean(row2, axis=0)
                result = np.dot(xNormalized, yNormalized) \
                         / np.sqrt(np.outer(np.sum(xNormalized**2, axis=0), np.sum(yNormalized**2, axis=0)))
                if (result[0][0] > 1): pearson = 1
                elif (result[0][0] < -1): pearson = -1
                else: pearson = result[0][0]
            """
            #pearsonCor.append([pearson, j])
            pearsonCor.append([np_pearson_cor(row, row2), j])
        maxx1 = [-1, 0]
        maxx2 = [-1, 0]
        for cor in pearsonCor:
            if maxx1 > maxx2:
                temp = maxx1
                maxx1 = maxx2
                maxx2 = temp
            if cor[0] > maxx1[0]:
                maxx1[0] = cor[0]
                maxx1[1] = cor[1]
            elif cor[0] > maxx2[0]:
                maxx2[0] = cor[0]
                maxx2[1] = cor[1]
        for entry in range(0, len(row)):
            if row[entry] == 0:
                weightSum = maxx1[0] + maxx2[0]
                if weightSum == 0: utilityMatrix[i][entry] = 0
                else:
                    weightedAverage = (maxx1[0] * utilityMatrix[maxx1[1]][entry]
                                       + maxx2[0] * utilityMatrix[maxx2[1]][entry]) / weightSum
                    utilityMatrix[i][entry] = weightedAverage
    finalPredictions = []
    i = 1
    for row in predictions[['userID', 'movieID']].to_numpy():
        finalPredictions.append([i, utilityMatrix[row[1]][row[0]]])
        i += 1
    return finalPredictions

#####
##
## LATENT FACTORS
##
#####
    
def predict_latent_factors(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass
    
    
#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
    return predict_collaborative_filtering(movies, users, ratings, predictions)

#####
##
## RANDOM PREDICTORS
##
#####
    
#By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)
    #predict_collaborative_filtering(movies, users, ratings, predictions)
    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]

#####
##
## SAVE RESULTS
##
#####    

predictions = predict_final(movies_description, users_description, ratings_description, predictions_description)

#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    #Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n'+'\n'.join(predictions)
    
    #Writes it down
    submission_writer.write(predictions)