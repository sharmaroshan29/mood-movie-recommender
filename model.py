# X = tfidf.fit_transform(df['overview'])
# y = df['mood']

# model = LogisticRegression(max_iter = 1000)
# model.fit(X, y)

# def fetch_poster(movie_title):
#     formatted_title = movie_title.replace(" ", "%20")
#     return f"https://placehold.co/300x450?text={formatted_title}"


# import random

# def recommend_movies(user_mood):
#     filtered = df[df['mood'] == user_mood]

#     if filtered.empty:
#         return filtered

    
#     filtered = filtered.drop_duplicates(subset='title')

#     X_test = tfidf.transform(filtered['overview'])
#     probs = model.predict_proba(X_test)

#     mood_index = list(model.classes_).index(user_mood)
#     filtered['score'] = probs[:, mood_index]
#     filtered = filtered.drop_duplicates(subset='title')

#     top_pool = filtered.sort_values(by='score', ascending=False).head(20)

#     return top_pool.sample(min(7, len(top_pool)))

