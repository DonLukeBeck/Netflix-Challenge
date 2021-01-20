import numpy as np
import pandas as pd
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

# Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'
corr_file = "./data/corr.csv"

# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID': 'int', 'year': 'int', 'movie': 'str'},
                                 names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';',
                                dtype={'userID': 'int', 'gender': 'str', 'age': 'int', 'profession': 'int'},
                                names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';',
                                  dtype={'userID': 'int', 'movieID': 'int', 'rating': 'int'},
                                  names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)
corr_description = pd.read_csf(corr_file, delimiter=';', header=None)


#####
##
## COLLABORATIVE FILTERING
##
#####

# for user-user, A should be a vector of all movies for some user
def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def np_pearson_cor(x, y):
    if np.nansum(x, axis=0) == 0 or np.nansum(y, axis=0) == 0: return 0
    xv = x - np.nanmean(x, axis=0)
    xv = np.nan_to_num(xv, copy=False)
    yv = y - np.nanmean(y, axis=0)
    yv = np.nan_to_num(yv, copy=False)
    xvss = np.sum((xv * xv), axis=0)
    yvss = np.sum((yv * yv), axis=0)
    if np.outer(xvss, yvss) != 0:
        result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    else:
        return 0
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result[0][0], 1.0), -1.0)


def predict_collaborative_filtering(movies, users, ratings, predictions):
    ratingsMatrix = np.zeros(users.shape[0] + 1, movies.shape[0] + 1)
    userMatrix = np.zeros(users.shape[0] + 1, movies.shape[0] + 1)

    for row in ratings[['userID', 'movieID', 'rating']].to_numpy():
        ratingsMatrix[row[0]][row[1]] = row[2]
        userMatrix[row[0]][row[1]] = row[2]

    # generate the rating matrix and user to user matrix once and save them to CSV
    # So that we can just load it the next time
    # And don't have to do all calculations again for a huge speedup

    for i in range(0, len(ratingsMatrix)):
        row = ratingsMatrix[i]
        if np.nansum(row) == 0:
            continue
        pearsonCor = []
        for j in range(i + 1, len(ratingsMatrix)):
            row2 = ratingsMatrix[j]
            pearsonCor.append([np_pearson_cor(row, row2), j])
        """
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
            if row[entry] == np.nan:
                weightSum = maxx1[0] + maxx2[0]
                if weightSum == 0:
                    ratingsMatrix[i][entry] = 0
                else:
                    weightedAverage = (maxx1[0] * ratingsMatrix[maxx1[1]][entry]
                                       + maxx2[0] * ratingsMatrix[maxx2[1]][entry]) / weightSum
                    ratingsMatrix[i][entry] = weightedAverage
    """
    finalPredictions = []
    i = 1
    for row in predictions[['userID', 'movieID']].to_numpy():
        if ratingsMatrix[row[1]][row[0]] != np.nan:
            finalPredictions.append([i, ratingsMatrix[row[1]][row[0]]])
        else:
            finalPredictions.append([i, 0])
        i += 1
    return finalPredictions


#####
##
## LATENT FACTORS
##
#####

def predict_latent_factors(movies, users, ratings, predictions):
    """
    ## TO COMPLETE
    ratingsMatrix = np.empty((users.shape[0] + 1, movies.shape[0] + 1))
    ratingsMatrix[:] = np.nan
    # movieMeanVector = np.zeros(movies['movieID'].shape[0]+1)

    for row in ratings[['userID', 'movieID', 'rating']].to_numpy():
        ratingsMatrix[row[0]][row[1]] = row[2]
    """
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

# By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)
    # predict_collaborative_filtering(movies, users, ratings, predictions)
    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]


#####
##
## SAVE RESULTS
##
#####

predictions = predict_final(movies_description, users_description, ratings_description, predictions_description)

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formats data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it down
    submission_writer.write(predictions)
