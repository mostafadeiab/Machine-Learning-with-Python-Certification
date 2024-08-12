# -*- coding: utf-8 -*-
# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# get data files
books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

# add your code here - consider creating a new cell for each section of code

users = df_ratings['user'].value_counts()
df_ratings = df_ratings[df_ratings['user'].isin(users[users >= 200].index)]
books = df_ratings['isbn'].value_counts()
df_ratings = df_ratings[df_ratings['isbn'].isin(books[books >= 100].index)]

df = pd.merge(df_ratings, df_books, on='isbn')
df = df.groupby(['title', 'user']).rating.mean().reset_index()

mtx = df.pivot(index='title', columns='user', values='rating').fillna(0)
book_mtx = csr_matrix(mtx.values)

# Train the KNN model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(book_mtx)

# function to return recommended books - this will be tested
def get_recommends(book = ""):
  book_idx = mtx.index.get_loc(book)

  # Find the K nearest neighbors (5 in this case)
  distances, indices = model.kneighbors(mtx.iloc[book_idx, :].values.reshape(1, -1), n_neighbors=6)

  # Create a list of recommendations
  recommended_books = []
  for i in range(1, len(distances.flatten())):
      recommended_books.append((mtx.index[indices.flatten()[i]], float(distances.flatten()[i])))

  return [book,recommended_books]

books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2):
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You haven't passed yet. Keep trying!")

test_book_recommendation()