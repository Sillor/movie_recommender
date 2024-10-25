from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

CORS(app)

# Preload data and model
movies_data = None
user_movie_matrix = None
svd = None
user_movie_matrix_svd = None
movie_to_id = None
id_to_movie = None

# Load IMDb movie data and ratings without filters
def load_data():
    try:
        movies = pd.read_csv('./imdb_data/title.basics.tsv', sep='\t', na_values='\\N', low_memory=False)
        ratings = pd.read_csv('./imdb_data/title.ratings.tsv', sep='\t', na_values='\\N', low_memory=False)

        # Filter and merge data
        movies['isAdult'] = pd.to_numeric(movies['isAdult'], errors='coerce').fillna(0).astype(int)
        movies = movies[(movies['titleType'] == 'movie') & (movies['isAdult'] == 0)]
        movies = movies[['tconst', 'primaryTitle', 'genres', 'startYear']].merge(ratings, on='tconst')
        movies.dropna(subset=['averageRating', 'numVotes'], inplace=True)

        # Convert startYear to numeric
        movies['startYear'] = pd.to_numeric(movies['startYear'], errors='coerce').fillna(0).astype(int)

        return movies
    except Exception as e:
        return None

# Create user-item matrix based on available movie ratings
def create_user_item_matrix(movies_data):
    genres_expanded = movies_data['genres'].str.get_dummies(sep=',')
    user_movie_matrix = movies_data[['tconst', 'averageRating']].join(genres_expanded)
    user_movie_matrix.set_index('tconst', inplace=True)
    return user_movie_matrix

# Train the SVD model
def train_svd_model(user_movie_matrix, n_components=50):
    try:
        n_components = min(n_components, user_movie_matrix.shape[1])
        svd = TruncatedSVD(n_components=n_components)
        user_movie_matrix_svd = svd.fit_transform(user_movie_matrix)
        return svd, user_movie_matrix_svd
    except Exception as e:
        return None, None

# Recommend movies based on SVD
def recommend_by_svd(svd, user_movie_matrix_svd, user_movie_matrix, movies_data, movie_to_id, id_to_movie, favorite_movies_with_years, start_year=None, end_year=None, min_rating=0.0, min_num_votes=0, top_k=5):
    try:
        # Get the indices for the favorite movies (these should always be included)
        favorite_movie_ids = []
        for movie_title, movie_year in favorite_movies_with_years:
            movie_id = movies_data[(movies_data['primaryTitle'] == movie_title) & (movies_data['startYear'] == movie_year)]['tconst'].values
            if len(movie_id) > 0:
                favorite_movie_ids.append(movie_id[0])

        if not favorite_movie_ids:
            return []

        # Apply filters to the movies
        filtered_movies = movies_data.copy()

        if start_year:
            filtered_movies = filtered_movies[filtered_movies['startYear'] >= start_year]
        if end_year:
            filtered_movies = filtered_movies[filtered_movies['startYear'] <= end_year]
        filtered_movies = filtered_movies[filtered_movies['averageRating'] >= min_rating]
        filtered_movies = filtered_movies[filtered_movies['numVotes'] >= min_num_votes]

        # Ensure favorite movies are in the dataset
        filtered_movie_ids = set(filtered_movies['tconst'].tolist())
        for fav_movie_id in favorite_movie_ids:
            if fav_movie_id not in filtered_movie_ids:
                favorite_movie_data = movies_data[movies_data['tconst'] == fav_movie_id]
                filtered_movies = pd.concat([filtered_movies, favorite_movie_data])

        # Update the user_movie_matrix based on filtered data
        user_movie_matrix_filtered = create_user_item_matrix(filtered_movies)

        # Recompute SVD on the filtered dataset
        svd, user_movie_matrix_svd_filtered = train_svd_model(user_movie_matrix_filtered)
        if svd is None or user_movie_matrix_svd_filtered is None:
            return []

        # Get the embeddings for the favorite movies
        favorite_embeddings = svd.transform(user_movie_matrix_filtered.loc[favorite_movie_ids])
        user_embedding = np.mean(favorite_embeddings, axis=0).reshape(1, -1)

        # Calculate cosine similarities with the filtered movies
        similarities = cosine_similarity(user_embedding, user_movie_matrix_svd_filtered)[0]

        # Sort by similarity and get top K recommendations
        top_movie_indices = similarities.argsort()[::-1]
        matrix_index_to_movie_id = dict(enumerate(user_movie_matrix_filtered.index))
        recommended_movie_ids = [matrix_index_to_movie_id[idx] for idx in top_movie_indices if matrix_index_to_movie_id[idx] in id_to_movie]

        # Filter out favorite movies from recommendations
        recommended_movie_ids = [movie_id for movie_id in recommended_movie_ids if movie_id not in favorite_movie_ids]

        # Return only the top K recommendations
        return [(id_to_movie[movie_id], movies_data[movies_data['tconst'] == movie_id]['startYear'].values[0]) for movie_id in recommended_movie_ids[:top_k]] if recommended_movie_ids else []
    
    except Exception as e:
        return []

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    try:
        data = request.get_json()
        favorite_movies_with_years = data.get('favorite_movies', [])
        start_year = data.get('start_year', None)
        end_year = data.get('end_year', None)
        min_rating = data.get('min_rating', 0.0)
        min_num_votes = data.get('min_num_votes', 0)
        
        # New parameter for number of recommendations
        num_recommendations = data.get('num_recommendations', 5)  # Default is 5

        if movies_data is None:
            return jsonify({"error": "Movie data is not loaded."}), 500

        # Pass num_recommendations to the recommend_by_svd function as top_k
        recommendations = recommend_by_svd(svd, user_movie_matrix_svd, user_movie_matrix, movies_data, movie_to_id, id_to_movie, favorite_movies_with_years, start_year, end_year, min_rating, min_num_votes, top_k=num_recommendations)

        if not recommendations:
            return jsonify({"error": "No recommendations found."}), 404

        # Convert NumPy types to native Python types for JSON serialization
        recommendations = [(title, int(year)) for title, year in recommendations]

        return jsonify({"recommendations": recommendations}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load data and model when the app starts
    movies_data = load_data()
    if movies_data is not None:
        user_movie_matrix = create_user_item_matrix(movies_data)
        svd, user_movie_matrix_svd = train_svd_model(user_movie_matrix)
        movie_to_id = movies_data.set_index('primaryTitle')['tconst'].to_dict()
        id_to_movie = movies_data.set_index('tconst')['primaryTitle'].to_dict()

    app.run(debug=True)
