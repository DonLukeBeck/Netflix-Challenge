import numpy as np
import pandas as pd
from random import randint

# -*- coding: utf-8 -*-
"""
FRAMEWORK FOR DATAMINING CLASS

#### IDENTIFICATION
NAME: ALEXANDRU
SURNAME: LUNGU
STUDENT ID: 5069602
KAGGLE ID: alexandrulungu

NAME: LUCA
SURNAME: BECHEANU
STUDENT ID: 4996240
KAGGLE ID: lucabecheanu

### NOTES
This files is an example of what your code should look like. 
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
movie_similarities_file = './data/movieSimilarities.csv'
baseline_matrix_file = './data/baselineMatrix.csv'
global_movie_similarities_file = './data/movieSimilaritiesGlobal.csv'

# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'])
similarities_description = pd.read_csv(movie_similarities_file, delimiter=';', header=None)
baseline_matrix_description = pd.read_csv(baseline_matrix_file, delimiter=';', header=None)
global_similarities_description = pd.read_csv(global_movie_similarities_file, delimiter=";", header=None)

def cosine_similarity(A, B):
    if np.linalg.norm(A) * np.linalg.norm(B) == 0:
        return 0
    else:
        return np.maximum(np.minimum(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B)), 1), -1)


def save_similarities(movies, ratingsMatrix, meanVector, name):
    # normalize all rows by their mean value
    for i in range(0, ratingsMatrix.shape[0]):
        for j in range(0, ratingsMatrix.shape[1]):
            if ratingsMatrix[i][j] != 0:
                ratingsMatrix[i][j] -= meanVector[i]

    # compute similarity matrix and save it to similarities.csv
    similarityMatrix = np.zeros((movies.shape[0] + 1, movies.shape[0] + 1))
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
    np.savetxt(str("./data/" + name + ".csv"), similarityMatrix, delimiter=";")


def predict_collaborative_filtering_item_item(movies, users, ratings, predictions):
    ratingsMatrix = np.zeros((movies.shape[0] + 1, users.shape[0] + 1))

    for row in ratings[['userID', 'movieID', 'rating']].to_numpy():
        ratingsMatrix[row[1]][row[0]] = row[2]

    # copy the matrix as we will modify the other one
    matrixCopy = np.copy(ratingsMatrix)

    # compute the mean vector
    meanVector = np.zeros(movies['movieID'].shape[0] + 1)
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
    # uncomment to save

    # save_similarities(movies, ratingsMatrix, meanVector)

    # initialize predictions array and start indexing from 1 as the first row is full of 0 values
    finalPredictions = []
    i = 1
    for row in predictions[['userID', 'movieID']].to_numpy():
        # if there is already a value in the matrix, we do not need to predict anything
        if matrixCopy[row[1]][row[0]] != 0:
            finalPredictions.append([i, matrixCopy[row[1]][row[0]]])
            continue
        # sort the similarities in reverse order and get their indexes (highest similarity is first index)
        similaritiesSorted = np.argsort(-similarities_description[row[1]])
        # if the sum of similarities is 0, we check if the user has any ratings and set the prediction
        # equal to his average rating, otherwise we use 3 as it is the median of 1 and 5
        if np.sum(similarities_description[row[1]]) == 0:
            if meanVector[row[1]] != 0:
                finalPredictions.append([i, meanVector[row[1]]])
            else:
                finalPredictions.append([i, 3])
        # predict the rating based on the k=100 nearest neighbors (most similar users)
        else:
            k = 15
            weightedSum = 0
            weightSum = 0
            # iterate over all indexes, stop when we found k neighbors
            for sIndex in similaritiesSorted:
                if k == 0:
                    break
                # compute the weighted sum of the ratings of the most similar users
                if matrixCopy[sIndex][row[0]] != 0:
                    weightedSum += matrixCopy[sIndex][row[0]] \
                           * similarities_description[row[1]][sIndex]
                    weightSum += similarities_description[row[1]][sIndex]
                    k -= 1
            # if the weighted sum and sum of weights are both non-zero, add the weighted average to the predictions
            # array and bound the rating between 1 and 5; otherwise, assume user would have rated it 3
            if weightSum != 0 and weightedSum != 0:
                finalPredictions.append([i, np.maximum(np.minimum(weightedSum/weightSum, 5), 1)])
            else:
                finalPredictions.append([i, 3])
        i += 1
    return finalPredictions


def compute_baseline_matrix(matrix, meanVector, averageRating):
    userMeanVector = np.zeros(matrix.shape[1] + 1)
    i = 0
    for row in matrix.T:
        s = 0
        length = 0
        for rating in row:
            s += rating
            if rating > 0: length += 1
        if length > 0: userMeanVector[i] = s / length
        i += 1
    for i in range(1, matrix.shape[0]):
        for j in range(1, matrix.shape[1]):
            if matrix[i][j] == 0:
                if userMeanVector[i] == 0 or meanVector[i] == 0:
                    matrix[i][j] = averageRating
                else:
                    userBias = userMeanVector[j] - averageRating
                    movieBias = meanVector[i] - averageRating
                    matrix[i][j] = np.maximum(np.minimum(averageRating + userBias + movieBias, 5), 1)
    np.savetxt("./data/baselineMatrix.csv", matrix, delimiter=";")


def predict_collaborative_filtering_item_item_global(movies, users, ratings, predictions):
    ratingsMatrix = np.zeros((movies.shape[0] + 1, users.shape[0] + 1))

    baselineMatrix = baseline_matrix_description.to_numpy().reshape((ratingsMatrix.shape[0], ratingsMatrix.shape[1]))

    for row in ratings[['userID', 'movieID', 'rating']].to_numpy():
        ratingsMatrix[row[1]][row[0]] = row[2]

    # compute the mean vector
    meanVector = np.zeros(movies['movieID'].shape[0] + 1)
    i = 0
    for row in ratingsMatrix:
        s = 0
        length = 0
        for rating in row:
            s += rating
            if rating > 0: length += 1
        if length > 0: meanVector[i] = s / length
        i += 1
    averageMovieRating = np.mean(meanVector[meanVector != 0])

    # 6040 users, 3706 movies

    # normalize all rows then compute and save Pearson correlation values in similarities.csv
    # uncomment to save similarities and baseline matrix

    # baselineMeanVector = np.zeros(movies['movieID'].shape[0] + 1)
    # i = 0
    # for row in baselineMatrix:
    #     s = 0
    #     length = 0
    #     for rating in row:
    #         s += rating
    #         if rating > 0: length += 1
    #     if length > 0: baselineMeanVector[i] = s / length
    #     i += 1
    # save_similarities(movies, baselineMatrix, baselineMeanVector, "movieSimilaritiesGlobal")
    # compute_baseline_matrix(ratingsMatrix, meanVector, averageMovieRating)

    # initialize predictions array and start indexing from 1 as the first row is full of 0 values
    finalPredictions = []
    i = 1
    for row in predictions[['userID', 'movieID']].to_numpy():
        # sort the similarities in reverse order and get their indexes (highest similarity is first index)
        similaritiesSorted = np.argsort(-global_similarities_description[row[1]])
        # if the sum of similarities is the mean, we check if the user has any ratings and set the prediction
        # equal to his average rating, otherwise we use the average movie rating
        if np.mean(global_similarities_description[row[1]]) == meanVector[row[1]]:
            if meanVector[row[1]] != 0:
                finalPredictions.append([i, meanVector[row[1]]])
            else:
                finalPredictions.append([i, averageMovieRating])
        # predict the rating based on the k=100 nearest neighbors (most similar users)
        else:
            k = 100
            weightedSum = 0
            weightSum = 0
            # iterate over all indexes, stop when we found k neighbors
            for sIndex in similaritiesSorted:
                if k == 0:
                    break
                # compute the weighted sum of the ratings of the most similar users
                if baselineMatrix[sIndex][row[0]] != 0:
                    weightedSum += baselineMatrix[sIndex][row[0]] \
                           * global_similarities_description[row[1]][sIndex]
                    weightSum += global_similarities_description[row[1]][sIndex]
                    k -= 1
            # if the weighted sum and sum of weights are both non-zero, add the weighted average to the predictions
            # array and bound the rating between 1 and 5; otherwise, assume user gives it the average movie rating
            if weightSum != 0 and weightedSum != 0:
                finalPredictions.append([i, np.maximum(np.minimum(weightedSum/weightSum, 5), 1)])
            else:
                finalPredictions.append([i, averageMovieRating])
        i += 1
    return finalPredictions

def predict_latent_factors_movies(movies, users, ratings, predictions):
    # initialize the ratings matrix with 0 values
    ratingsMatrix = np.zeros((movies.shape[0] + 1, users.shape[0] + 1))

    # add ratings to the array
    for row in ratings[['userID', 'movieID', 'rating']].to_numpy():
        ratingsMatrix[row[1]][row[0]] = row[2]

    # compute the mean vector and use it to normalize all rows
    meanVector = np.zeros(movies['movieID'].shape[0]+1)
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
            (np.minimum(np.dot(Q[row[1], :], P[:, row[0]]) + meanVector[row[1]], 5), 1)])
        i += 1
    return finalPredictions

def predict_latent_factors_movies_global(movies, users, ratings, predictions):
    # initialize the ratings matrix with 0 values
    ratingsMatrix = np.zeros((movies.shape[0] + 1, users.shape[0] + 1))

    # add ratings to the array
    for row in ratings[['userID', 'movieID', 'rating']].to_numpy():
        ratingsMatrix[row[1]][row[0]] = row[2]

    # compute the mean vector and use it to normalize all rows
    meanVector = np.zeros(movies['movieID'].shape[0]+1)
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

    userMeanVector = np.zeros(users['userID'].shape[0] + 1)
    i = 0
    for row in ratingsMatrix.T:
        s = 0
        length = 0
        for rating in row:
            s += rating
            if rating > 0: length += 1
        if length > 0: userMeanVector[i] = s / length
        i += 1

    averageMovieRating = np.mean(meanVector[meanVector != 0])

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
                    movieBias = meanVector[x] - averageMovieRating
                    userBias = userMeanVector[i] - averageMovieRating
                    errorDerivative = 2 * (ratingsMatrix[x][i] -
                                           (averageMovieRating + movieBias + userBias + np.dot(Q[x, :], P[:, i])))
                    Q[x, :] += 0.0001 * (errorDerivative * P[:, i] - 0.6 * Q[x, :])
                    P[:, i] += 0.0001 * (errorDerivative * Q[x, :] - 0.6 * P[:, i])

    # predict final values for each user and movie
    finalPredictions = []
    i = 1
    for row in predictions[['userID', 'movieID']].to_numpy():
        finalPredictions.append([i, np.maximum
            (np.minimum(np.dot(Q[row[1], :], P[:, row[0]]) + meanVector[row[1]], 5), 1)])
        i += 1
    return finalPredictions

def predict(movies, users, ratings, predictions):
    return predict_latent_factors_movies(movies, users, ratings, predictions)

#####
##
## SAVE RESULTS
##
#####    

## //!!\\ TO CHANGE by your prediction function
predictions = predict(movies_description, users_description, ratings_description, predictions_description)

#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    #Formats data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n'+'\n'.join(predictions)
    
    #Writes it dowmn
    submission_writer.write(predictions)