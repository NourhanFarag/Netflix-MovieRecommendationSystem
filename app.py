
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
