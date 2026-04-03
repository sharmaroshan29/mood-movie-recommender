

# Importing the required libraries

import pandas as pd
import streamlit as st
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import requests
import time

df = pd.read_csv("movies.csv")
df = df[['title', 'overview', 'genres']]

df = df.dropna()
df.reset_index(drop=True, inplace=True)

def extract_genres(text):
    generes = []
    for item in ast.literal_eval(text):
        generes.append(item['name'])
    return " ".join(generes)

df['genres'] = df['genres'].apply(extract_genres)


def map_mood(genre):
    if "Comedy" in genre:
        return "Happy"
    elif "Drama" in genre:
        return "Sad"
    elif "Romance" in genre:
        return "Romantic"
    elif "Action" in genre:
        return "Excited"
    elif "Horror" in genre:
        return "Scary"
    else:
        return "Neutral"
df["mood"] = df["genres"].apply(map_mood)

tfidf = TfidfVectorizer(stop_words= 'english')

X = tfidf.fit_transform(df['overview'])



# Training the model



y = df['mood']

model = LogisticRegression(max_iter = 1000)
model.fit(X, y)

def fetch_poster(movie_title):
    formatted_title = movie_title.replace(" ", "%20")
    return f"https://placehold.co/300x450?text={formatted_title}"


import random

def recommend_movies(user_mood):
    filtered = df[df['mood'] == user_mood]

    if filtered.empty:
        return filtered

    
    filtered = filtered.drop_duplicates(subset='title')

    X_test = tfidf.transform(filtered['overview'])
    probs = model.predict_proba(X_test)

    mood_index = list(model.classes_).index(user_mood)
    filtered['score'] = probs[:, mood_index]
    filtered = filtered.drop_duplicates(subset='title')

    top_pool = filtered.sort_values(by='score', ascending=False).head(20)

    return top_pool.sample(min(7, len(top_pool)))


# User interface


st.set_page_config(page_title="Movie recommender", layout = "wide")

st.title("🎬 Mood-Based Movie Recommender")
st.markdown("✨ Tell us your mood and we will find the perfect match for you")

st.divider()

mood = st.selectbox(
    "How are you feeling?",
    ["Happy", "Sad", "Romantic", "Excited", "Scary"]

)

if st.button("Recommend Movies"):
    with st.spinner("Finding perfect movies for you... 🎥"):
        time.sleep(1.5)   # simulate processing
        results = recommend_movies(mood)
    

    if results.empty:
        st.warning("No movies found for this mood 😔")
        st.stop()

    st.markdown("Top Pick")

    top = results.iloc[0]
    poster = fetch_poster(top['title'])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(poster)

    with col2:
        st.subheader(top['title'])
        st.write(top['overview'])
        st.success(f"Perfect match for youyr mood: {mood}")

    st.divider()

    st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stSelectbox label {
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

    cols = st.columns(3)

    results = results.reset_index(drop=True)
    top = results.iloc[0]
    rest = results.iloc[1:]

    for i, (_, row) in enumerate(rest.iterrows()):
     with cols[i % 3]:
        poster = fetch_poster(row['title'])  

        st.image(poster)
        st.markdown(f"### {row['title']}")
        st.caption(row['genres'])
        st.markdown("⭐ Matches your mood")




 




