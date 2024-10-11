# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import requests  # For fetching movie posters

# # OMDb API Key (replace with your own key)
# OMDB_API_KEY = '34c1bf3c'

# # Function to fetch movie poster from OMDb API
# def fetch_movie_poster(title):
#     url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
#     response = requests.get(url)
#     data = response.json()

#     if 'Poster' in data and data['Poster'] != 'N/A':
#         return data['Poster']
#     return None

# # Load the preprocessed Netflix dataset
# @st.cache_data
# def load_data():
#     df = pd.read_csv('netflix_titles.csv')  # Ensure the file path is correct
#     return df

# # Preprocess and combine features for content-based filtering
# def preprocess_data(df):
#     df['director'] = df['director'].fillna('')
#     df['cast'] = df['cast'].fillna('')
#     df['listed_in'] = df['listed_in'].fillna('')
#     df['description'] = df['description'].fillna('')
    
#     # Combine features
#     df['combined_features'] = df['type'] + ' ' + df['director'] + ' ' + df['cast'] + ' ' + df['listed_in'] + ' ' + df['description']
#     return df

# # Build the recommendation system (content-based)
# def build_recommender(df):
#     tfidf = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = tfidf.fit_transform(df['combined_features'])
#     cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#     return cosine_sim

# # Get content-based recommendations
# def get_content_recommendations(title, df, cosine_sim):
#     try:
#         idx = df[df['title'].str.contains(title, case=False)].index[0]
#     except IndexError:
#         return "Title not found!"
    
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:11]
    
#     movie_indices = [i[0] for i in sim_scores]
#     return df['title'].iloc[movie_indices]

# # Main application
# def main():
#     st.title('Netflix Movie Recommendation System')

#     # Load and preprocess the data
#     df = load_data()
#     df = preprocess_data(df)
#     cosine_sim = build_recommender(df)

#     # Get user input
#     movie_title = st.text_input('Enter a movie title:')

#     if st.button('Get Recommendations'):
#         if movie_title:
#             recommendations = get_content_recommendations(movie_title, df, cosine_sim)
#             if isinstance(recommendations, str):
#                 st.write(recommendations)
#             else:
#                 st.write('Top 10 Recommendations:')
#                 cols = st.columns(3)  # Create 4 columns for the grid
#                 for i, movie in enumerate(recommendations):
#                     with cols[i % 3]:  # Place the movie in the corresponding column
#                         st.write(f"**{movie}**")
                        
#                         # Fetch and display the poster
#                         poster_url = fetch_movie_poster(movie)
#                         if poster_url:
#                             st.image(poster_url, width=200)  # Display the movie poster
#                         else:
#                             st.write("Poster not available.")
#         else:
#             st.write('Please enter a valid movie title.')

# if __name__ == '__main__':
#     main()

# ---------------------------------------------------------


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import requests  # For fetching movie posters
# import joblib  # For loading the saved KNN model

# # OMDb API Key (replace with your own key)
# OMDB_API_KEY = '34c1bf3c'

# # Function to fetch movie poster from OMDb API
# def fetch_movie_poster(title):
#     url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
#     response = requests.get(url)
#     data = response.json()

#     if 'Poster' in data and data['Poster'] != 'N/A':
#         return data['Poster']
#     return None

# # Load the preprocessed Netflix dataset
# @st.cache_data
# def load_data():
#     df = pd.read_csv('preprocessed_netflix_data.csv')  # Ensure the file path is correct
#     return df

# # Load the KNN model
# @st.cache_resource
# def load_knn_model():
#     model = joblib.load('knn_model.pkl')  # Update with your model's path
#     return model

# # Preprocess and combine features for content-based filtering
# def preprocess_data(df):
#     df['director'] = df['director'].fillna('')
#     df['cast'] = df['cast'].fillna('')
#     df['listed_in'] = df['listed_in'].fillna('')
#     df['description'] = df['description'].fillna('')
    
#     # Combine features
#     df['combined_features'] = df['type'] + ' ' + df['director'] + ' ' + df['cast'] + ' ' + df['listed_in'] + ' ' + df['description']
#     return df

# # Build the recommendation system (content-based)
# def build_recommender(df):
#     tfidf = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = tfidf.fit_transform(df['combined_features'])
#     cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#     return cosine_sim

# # Get content-based recommendations
# def get_content_recommendations(title, df, cosine_sim):
#     try:
#         idx = df[df['title'].str.contains(title, case=False)].index[0]
#     except IndexError:
#         return "Title not found!"
    
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:11]
    
#     movie_indices = [i[0] for i in sim_scores]
#     return df['title'].iloc[movie_indices]

# # Get KNN recommendations
# def get_knn_recommendations(title, df, knn_model):
#     try:
#         idx = df[df['title'].str.contains(title, case=False)].index[0]
#     except IndexError:
#         return "Title not found!"

#     movie_id = df['movie_id_encoded'].iloc[idx]  # Adjust as per your encoding
#     knn_indices = knn_model.kneighbors([[movie_id]], n_neighbors=11, return_distance=False)[0][1:]  # Exclude the first one (itself)
    
#     return df['title'].iloc[knn_indices]

# # Main application
# def main():
#     st.title('Netflix Movie Recommendation System')

#     # Load and preprocess the data
#     df = load_data()
#     df = preprocess_data(df)
#     cosine_sim = build_recommender(df)
#     knn_model = load_knn_model()  # Load KNN model

#     # Get user input
#     movie_title = st.text_input('Enter a movie title:')

#     if st.button('Get Recommendations'):
#         if movie_title:
#             # Get content-based recommendations
#             content_recommendations = get_content_recommendations(movie_title, df, cosine_sim)
            
#             # Get KNN recommendations
#             knn_recommendations = get_knn_recommendations(movie_title, df, knn_model)
            
#             if isinstance(content_recommendations, str):
#                 st.write(content_recommendations)
#             else:
#                 st.write('Top 10 Content-Based Recommendations:')
#                 cols = st.columns(3)
#                 for i, movie in enumerate(content_recommendations):
#                     with cols[i % 3]:
#                         st.write(f"**{movie}**")
#                         poster_url = fetch_movie_poster(movie)
#                         if poster_url:
#                             st.image(poster_url, width=200)
#                         else:
#                             st.write("Poster not available.")
            
#             if isinstance(knn_recommendations, str):
#                 st.write(knn_recommendations)
#             else:
#                 st.write('Top 10 KNN Recommendations:')
#                 cols = st.columns(3)
#                 for i, movie in enumerate(knn_recommendations):
#                     with cols[i % 3]:
#                         st.write(f"**{movie}**")
#                         poster_url = fetch_movie_poster(movie)
#                         if poster_url:
#                             st.image(poster_url, width=200)
#                         else:
#                             st.write("Poster not available.")
#         else:
#             st.write('Please enter a valid movie title.')

# if __name__ == '__main__':
#     main()
# ---------------------------------
import streamlit as st
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer

# Your IMDb API key
IMDB_API_KEY = '34c1bf3c'  # Replace with your actual API key

# Load your movie dataset
df = pd.read_csv('preprocessed_netflix_data.csv')  # Update with the actual path to your dataset

# Create a list of movie titles for the dropdown
movies_list = df['title'].tolist()

# Combining text features for content-based similarity
df['combined_features'] = df['type'] + ' ' + df['director'] + ' ' + df['cast'] + ' ' + df['listed_in'] + ' ' + df['description']

# Step 1: Content-Based Model (TF-IDF + Cosine Similarity)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# KNN Preparation
def train_knn_model():
    # Separate features and target variable for KNN
    X = df.drop(columns=['type'])  # Features
    y = df['type']  # Target variable

    # Apply oversampling to address class imbalance
    oversampler = RandomOverSampler()
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Encode categorical variables for KNN
    label_encoder = LabelEncoder()
    X_resampled_encoded = X_resampled.apply(label_encoder.fit_transform)

    # Feature scaling (KNN benefits from feature scaling)
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled_encoded)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled_scaled, y_resampled, test_size=0.2, random_state=42)

    # Initialize K-Nearest Neighbors classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)

    # Train the KNN model
    knn_classifier.fit(X_train, y_train)
    
    return knn_classifier, X_resampled_scaled

# Train the model once when the app starts
knn_classifier, X_resampled_scaled = train_knn_model()

# Function to get content-based recommendations
def get_content_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = df[df['title'].str.contains(title, case=False)].index[0]
    except IndexError:
        return [], []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:5]  # Get top 4 recommendations
    show_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[show_indices]

# Function to fetch movie poster using IMDb API
def fetch_movie_poster(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={IMDB_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if 'Poster' in data and data['Poster'] != 'N/A':
        return data['Poster']
    
    # Logging if no poster found
    st.warning(f"No poster found for {title}. Response: {data}")
    return None

# Voting Classifier: Combine KNN and Content-Based Model
def voting_classifier(title):
    content_based_recommendations = get_content_recommendations(title)

    # Get KNN predictions
    try:
        idx = df[df['title'].str.contains(title, case=False)].index[0]
        knn_pred = knn_classifier.predict([X_resampled_scaled[idx]])[0]
    except IndexError:
        return None

    return {
        "KNN_Prediction": knn_pred,
        "Content-Based_Recommendations": content_based_recommendations,
    }

# Streamlit app title
st.title("Movie Recommendation System")

# Dropdown for movie selection
selectvalue = st.selectbox("Select movie from dropdown", movies_list)

if st.button("Recommend"):
    result = voting_classifier(selectvalue)

    if result:
        st.write("KNN Prediction (Type):", result["KNN_Prediction"])
        st.write("Content-Based Recommendations:")

        # Display movie posters and titles
        cols = st.columns(4)  # Create 5 columns for display
        for col, title in zip(cols, result["Content-Based_Recommendations"]):
            poster_url = fetch_movie_poster(title)
            if poster_url:  # Check if the poster URL is valid
                with col:
                    st.image(poster_url, width=150)  # Display movie poster
                    st.write(title)  # Display movie title
            else:
                with col:
                    st.write(f"No poster available for {title}.")  # Display message if no poster
    else:
        st.error("No recommendations found or title not found in the dataset.")
