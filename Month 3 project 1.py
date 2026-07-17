# ==========================================================
# Content-Based Movie Recommendation System
# ==========================================================

# Import Libraries
import pandas as pd
import ast
import re
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# Load Dataset
# ==========================================================

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets using title
movies = movies.merge(credits, on="title")

# ==========================================================
# Select Required Columns
# ==========================================================

movies = movies[['movie_id',
                 'title',
                 'overview',
                 'genres',
                 'keywords',
                 'cast']]

# ==========================================================
# Remove Missing Values
# ==========================================================

movies.dropna(inplace=True)

# ==========================================================
# Functions to Convert JSON Columns
# ==========================================================

def convert(text):
    names = []
    for item in ast.literal_eval(text):
        names.append(item['name'])
    return names


def fetch_cast(text):
    names = []
    counter = 0
    for item in ast.literal_eval(text):
        if counter < 3:
            names.append(item['name'])
            counter += 1
        else:
            break
    return names


# ==========================================================
# Convert JSON Columns
# ==========================================================

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(fetch_cast)

# ==========================================================
# Clean Overview
# ==========================================================

movies['overview'] = movies['overview'].apply(lambda x: x.split())

# ==========================================================
# Remove Spaces from Multi-word Names
# ==========================================================

movies['genres'] = movies['genres'].apply(
    lambda x: [i.replace(" ", "") for i in x])

movies['keywords'] = movies['keywords'].apply(
    lambda x: [i.replace(" ", "") for i in x])

movies['cast'] = movies['cast'].apply(
    lambda x: [i.replace(" ", "") for i in x])

# ==========================================================
# Create Tags Column
# ==========================================================

movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
)

# ==========================================================
# Create New DataFrame
# ==========================================================

new_df = movies[['movie_id', 'title', 'tags']].copy()

# Convert list into string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Convert to lowercase
new_df['tags'] = new_df['tags'].str.lower()

# Remove punctuation
new_df['tags'] = new_df['tags'].apply(
    lambda x: re.sub(r'[^a-zA-Z0-9 ]', '', x)
)

# ==========================================================
# Vectorization
# ==========================================================

cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(new_df['tags']).toarray()

# ==========================================================
# Cosine Similarity
# ==========================================================

similarity = cosine_similarity(vectors)

# ==========================================================
# Recommendation Function
# ==========================================================

def recommend(movie_name):

    movie_name = movie_name.lower()

    if movie_name not in new_df['title'].str.lower().values:
        print("\nMovie not found in the dataset.")
        return

    index = new_df[new_df['title'].str.lower() == movie_name].index[0]

    distances = similarity[index]

    recommended_movies = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print("\nTop 5 Recommended Movies:\n")

    for movie in recommended_movies:
        print(new_df.iloc[movie[0]].title)

# ==========================================================
# Example
# ==========================================================

recommend("Avatar")

# ==========================================================
# Save Models
# ==========================================================

pickle.dump(new_df, open("movies.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

print("\nModels saved successfully!")
