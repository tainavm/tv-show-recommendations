import numpy as np
import pandas as pd
import time
import os
from surprise import Reader
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise.model_selection import cross_validate

# Read the file
data = pd.io.parsers.read_csv('../datasets/ratings-oficial.dat',
    names=['user_id', 'movie_id', 'rating'],
    engine='python', delimiter='::')

movie_data = pd.io.parsers.read_csv('../datasets/tv-shows-oficial.dat',
    names=['movie_id', 'title'],
    engine='python', delimiter=',')


# Create the ratings matrix of shape (𝑚×𝑢) with rows as movies and columns as users
ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

print(ratings_mat)

normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

print(normalised_mat)

A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = np.linalg.svd(A)

def top_cosine_similarity(data, movie_id):
    index = movie_id - 1 # Movie id starts from 1
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
#     return sort_indexes[:top_n]
    return sort_indexes[1:11]

def get_movie_id(movie_title):
    movie_id = movie_data[movie_data.title == title].movie_id.values[0]
    return movie_id

# Helper function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Baseado no seu gosto por "{}" você deveria assistir:'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))
    i = 0
    for id in top_indexes + 1:
        i = i + 1
        print('{0}: {1}'.format(i, movie_data[movie_data.movie_id == id].title.values[0]))

title = 'Criminal Minds'

movie_id = get_movie_id(title)
k = 50
sliced = V.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, movie_id)

start_time = time.time()
print_similar_movies(movie_data, movie_id, indexes)

print ('\nTotal Runtime: {:.2f} seconds'.format(time.time() - start_time))
