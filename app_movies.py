import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

API_KEY = "2c18dd94a8f1025e5de4ac1e07d164ab"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"

@st.cache_data
def get_movies():
    df = pd.read_csv('movies_metadata.csv', low_memory=False)
    df = df[['id', 'title', 'overview', 'release_date']].copy()
    df.dropna(subset=['overview', 'release_date'], inplace=True)
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df = df[df['year'] >= 1960].dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    df['id'] = df['id'].astype(str)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data
def build_sim_matrix(movies):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = tfidf.fit_transform(movies['overview'])
    return linear_kernel(matrix, matrix)

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    try:
        res = requests.get(url, timeout=5)
        data = res.json()
        path = data.get('poster_path')
        if path:
            return f"{POSTER_BASE}/{path}", round(data.get('vote_average', 0), 1)
    except:
        pass
    return "https://via.placeholder.com/300x450/cccccc/666666?text=No+Poster", None

def recommend(title, movies, sim_matrix):
    idx_map = pd.Series(movies.index, index=movies['title']).to_dict()
    if title not in idx_map:
        return []
    idx = idx_map[title]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in scores[1:11]]
    return movies[['title', 'id', 'year']].iloc[top_indices].to_dict('records')

st.set_page_config(page_title="Reel Suggest", page_icon="🎞️")
st.markdown("<h1 style='text-align: center;'>🎞️ Reel Suggest</h1>", unsafe_allow_html=True)
st.caption("Pick a movie, slide the years, get similar flicks.")

with st.spinner("Loading database..."):
    movies = get_movies()
    sim = build_sim_matrix(movies)

col1, col2 = st.columns([2, 1])
with col1:
    selected = st.selectbox("Your favorite movie:", movies['title'].values)
with col2:
    years = st.slider("Release year range:", 1960, 2026, (1990, 2026), step=1)

if st.button("Find my next watch", use_container_width=True):
    raw_recs = recommend(selected, movies, sim)
    filtered = [m for m in raw_recs if years[0] <= m['year'] <= years[1]]
    if not filtered:
        st.warning("Nothing in that year range. Broaden the slider.")
        st.stop()
    st.subheader(f"Showing {len(filtered)} recommendations")
    cols = st.columns(2)
    for i, movie in enumerate(filtered):
        poster, rating = fetch_poster(movie['id'])
        with cols[i % 2]:
            st.image(poster, use_container_width=True)
            st.markdown(f"**{movie['title']}** ({movie['year']})")
            if rating:
                st.markdown(f"⭐ {rating}/10")
            st.divider()