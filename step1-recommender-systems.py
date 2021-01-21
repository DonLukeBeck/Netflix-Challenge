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
similarities_file = "./data/similarities.csv"

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
similarities_description = pd.read_csv(similarities_file, delimiter=" ", header=None)


#####
##
## COLLABORATIVE FILTERING
##
#####

# for user-user, A should be a vector of all movies for some user
def cosine_similarity(A, B):
    if np.linalg.norm(A) * np.linalg.norm(B) == 0:
        return 0
    else:
        return np.maximum(np.minimum(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B)), 1), -1)

def save_similarities(users, ratingsMatrix, meanVector):
    # normalize all rows by their mean value
    for i in range(0, ratingsMatrix.shape[0]):
        for j in range(0, ratingsMatrix.shape[1]):
            if ratingsMatrix[i][j] != 0:
                ratingsMatrix[i][j] -= meanVector[i]

    # compute similarity matrix and save it to similarities.csv
    similarityMatrix = np.zeros((users.shape[0] + 1, users.shape[0] + 1))
    similarityMatrix[0][0] = 1
    for i in range(1, similarityMatrix.shape[0]):
        for j in range(1, similarityMatrix.shape[1]):
            if similarityMatrix[i][j] == 0:
                if i == j:
                    similarityMatrix[i][j] = 1
                else:
                    similarity = cosine_similarity(ratingsMatrix[i], ratingsMatrix[j])
                    similarityMatrix[i][j] = similarity
                    similarityMatrix[j][i] = similarity
    np.savetxt("./data/similarities.csv", similarityMatrix, delimiter=";")

def predict_collaborative_filtering(movies, users, ratings, predictions):
    ratingsMatrix = np.zeros((users.shape[0] + 1, movies.shape[0] + 1))
    print(movies.shape)
    print(users.shape)
    print(ratingsMatrix.shape)
    for row in ratings[['userID', 'movieID', 'rating']].to_numpy():
        ratingsMatrix[row[0]][row[1]] = row[2]

    matrixCopy = np.copy(ratingsMatrix)

    # compute the mean vector
    meanVector = np.zeros(users['userID'].shape[0] + 1)
    i = 0
    for row in ratingsMatrix:
        s = 0
        length = 0
        for rating in row:
            s += rating
            if rating > 0: length += 1
        if length > 0: meanVector[i] = s / length
        i += 1
    # 6040 users, 3706 movies

    # normalize all rows then compute and save Pearson correlation values in similarities.csv

    # save_similarities(users, ratingsMatrix, meanVector)

    similaritiesSorted = np.argsort(-similarities_description)

    pass


#####
##
## LATENT FACTORS
##
#####

def predict_latent_factors(movies, users, ratings, predictions):
    # initialize the ratings matrix with 0 values
    ratingsMatrix = np.zeros((users.shape[0] + 1, movies.shape[0] + 1))

    # add ratings to the array
    for row in ratings[['userID', 'movieID', 'rating']].to_numpy():
        ratingsMatrix[row[0]][row[1]] = row[2]

    # compute the mean vector and use it to normalize all rows
    meanVector = np.zeros(users['userID'].shape[0]+1)
    i=0
    for row in ratingsMatrix:
        s = 0
        length = 0
        for rating in row:
            s += rating
            if rating > 0: length += 1
        if length > 0: meanVector[i] = s / length
        i += 1
    for i in range(0, ratingsMatrix.shape[0]):
        for j in range(0, ratingsMatrix.shape[1]):
            if ratingsMatrix[i][j] != 0:
                ratingsMatrix[i][j] -= meanVector[i]

    # apply SVD on the ratingsMatrix and then compute Q and P, keeping only 50 factors
    U, sigma, Vt = np.linalg.svd(ratingsMatrix)
    Q = U[:, :50]
    sigma = sigma[:50]
    P = np.diag(sigma) @ Vt[:50, :]

    # compute the latent factors by using Stochastic Gradient Descent
    for k in range(0, 10):
        for x in range(1, ratingsMatrix.shape[0]):
            for i in range(1, ratingsMatrix.shape[1]):
                if ratingsMatrix[x][i] != 0:
                    errorDerivative = 2 * (ratingsMatrix[x][i] - np.dot(Q[x, :], P[:, i]))
                    Q[x, :] += 0.0001 * (errorDerivative * P[:, i] - 0.6 * Q[x, :])
                    P[:, i] += 0.0001 * (errorDerivative * Q[x, :] - 0.6 * P[:, i])

    # predict final values for each user and movie
    finalPredictions = []
    i = 1
    for row in predictions[['userID', 'movieID']].to_numpy():
        finalPredictions.append([i, np.maximum
            (np.minimum(np.dot(Q[row[0], :], P[:, row[1]]) + meanVector[row[0]], 5), 1)])
        i += 1
    return finalPredictions



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
