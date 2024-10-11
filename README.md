## Project: Movie Recommendation System Using Machine Learning!
# Table of Contents
1. Project Overview
2. Features
3. Dataset
4. Usage
5. Models and Techniques


# Project Overview
This project is a movie recommendation system designed to suggest movies to users based on their preferences. The system integrates multiple recommendation techniques, including collaborative filtering, content-based filtering, and deep learning embeddings.

The project was deployed using Streamlit to create an easy-to-use web interface where users can enter a movie and receive tailored recommendations.

* [Click here to run it live on server](https://movierecommendergp.streamlit.app/)

# Features
- Hybrid Recommendation System: Combines collaborative filtering and content-based filtering for accurate suggestions.
- Deep Learning: Utilizes embedding models for advanced recommendation.
- Deployed Application: Interactive web app built with Streamlit.
- Evaluation Metrics: Performance evaluated using precision, recall, and error metrics like MAE and RMSE.

# Dataset
* [Dataset link](https://www.kaggle.com/datasets/shivamb/netflix-shows?select=netflix_titles.csv)

# Usage
Once the app is running, you can:
1. Enter the name of a movie you like.
2. The system will generate movie recommendations based on content, collaborative filtering, or a hybrid approach.
You can also experiment with different models and techniques within the provided Jupyter notebooks.

# Models and Techniques
1. Collaborative Filtering: We implemented KNN-based collaborative filtering to suggest movies similar to a user's preferences.
2. Content-Based Filtering: Using TF-IDF and GloVe embeddings, we recommend movies with similar content (e.g., genres, cast, description).
3. Neural Collaborative Filtering: Deep learning models that use user and movie embeddings to make recommendations.
4. Voting Classifier: Combines predictions from multiple classifiers for a more robust recommendation.

```bash
Author: Nourhan Farag
Email: nourhanfarag2002@gmail.com

```
