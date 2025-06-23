import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Create movie-user matrix
movie_user_matrix = ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)

# Dimensionality reduction
pca = PCA(n_components=50)
movie_features = pca.fit_transform(movie_user_matrix)

# KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(movie_features)

# Attach clusters to movies
movie_user_matrix['cluster'] = clusters
movie_cluster_df = movie_user_matrix[['cluster']].reset_index().merge(movies, on='movieId')

# Recommendation function
def recommend_movies(movie_name):
    try:
        movie_name = movie_name.strip()
        movie_row = movies[movies['title'] == movie_name]

        if movie_row.empty:
            return None, f"❌ Movie '{movie_name}' not found in the dataset."

        movie_id = movie_row['movieId'].values[0]
        cluster_id = movie_cluster_df[movie_cluster_df['movieId'] == movie_id]['cluster'].values[0]
        similar = movie_cluster_df[movie_cluster_df['cluster'] == cluster_id]
        recommendations = similar[similar['movieId'] != movie_id]['title'].sample(min(5, len(similar))).values
        return recommendations, None

    except Exception as e:
        return None, f"⚠️ Error: {e}"

# Streamlit UI
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System (Unsupervised)")
st.write("Type the **exact movie name** from the dataset (e.g., `Toy Story (1995)`):")

user_input = st.text_input("Enter Movie Title")

if st.button("Get Recommendations"):
    if user_input:
        results, error = recommend_movies(user_input)
        if error:
            st.error(error)
        else:
            st.success("✅ Recommendations:")
            for title in results:
                st.write(f"- {title}")
    else:
        st.warning("⚠️ Please enter a movie name.")
