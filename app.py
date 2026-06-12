import streamlit as st
import pickle
import pandas as pd
import requests

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🍿",
    layout="wide"
)

# API KEY (Hardcoded to remove sidebar configuration)
API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    try:
        movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
        similarity = pickle.load(open("similarity_dict.pkl", "rb"))
        movies = pd.DataFrame(movies_dict)
        # Validate that required columns exist
        required_cols = {'movie_id', 'title'}
        if not required_cols.issubset(movies.columns):
            st.error(
                f"Error: Pickle data is missing required columns. "
                f"Found columns: {list(movies.columns)}. "
                f"Expected at least: {required_cols}. "
                f"Please re-run the training notebook (1.ipynb) to regenerate the pickle files."
            )
            return pd.DataFrame(columns=['movie_id', 'title', 'tags']), []
        return movies, similarity
    except FileNotFoundError:
        st.error("Error: 'movies_dict.pkl' or 'similarity_dict.pkl' not found. Ensure models are trained and saved.")
        return pd.DataFrame(columns=['movie_id', 'title', 'tags']), []

movies, similarity = load_data()

# ------------------ API CALLS ------------------
def fetch_poster(poster_path):
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return "https://via.placeholder.com/500x750?text=No+Poster"

def fetch_movie_details(movie_id):
    """Fetch movie details from TMDB using api.tmdb.org to bypass blocks"""
    try:
        url = f"https://api.tmdb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()
        return {
            "title": data.get("title", "Unknown"),
            "poster": fetch_poster(data.get('poster_path')),
            "overview": data.get("overview", "No overview available."),
            "rating": data.get("vote_average", "N/A"),
            "release_date": data.get("release_date", "N/A")
        }
    except Exception:
        return {
            "title": "Error fetching",
            "poster": "https://via.placeholder.com/500x750?text=Error",
            "overview": "Error fetching details.", 
            "rating": "N/A", 
            "release_date": "N/A"
        }

# ------------------ CORE LOGIC ------------------
def parse_tmdb_results(api_results):
    """Helper to format API results into lists"""
    names, posters, details = [], [], []
    for item in api_results[:5]:
        names.append(item.get("title", "Unknown"))
        posters.append(fetch_poster(item.get("poster_path")))
        details.append({
            "overview": item.get("overview", "No overview available."),
            "rating": item.get("vote_average", "N/A"),
            "release_date": item.get("release_date", "N/A")
        })
    return names, posters, details

def recommend(query):
    query_lower = query.lower().strip()
            
    # 1. LOCAL DATASET BASED RECOMMENDATION (ML Model)
    local_match = movies[movies['title'].str.lower() == query_lower]
    if not local_match.empty:
        movie_index = local_match.index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        names, posters, details = [], [], []
        for i in movies_list:
            movie_id = movies.iloc[i[0]].movie_id
            m_details = fetch_movie_details(movie_id)
            names.append(m_details['title'] if m_details['title'] != 'Unknown' else movies.iloc[i[0]].title)
            posters.append(m_details['poster'])
            details.append(m_details)
        return "local", (names, posters, details)
        
    # 2. GLOBAL SEARCH (ANY MOVIE NOT IN DATASET)
    try:
        search_url = f"https://api.tmdb.org/3/search/movie?api_key={API_KEY}&query={query}"
        search_data = requests.get(search_url, timeout=5).json()
        if 'results' in search_data and search_data['results']:
            movie_id = search_data['results'][0]['id']
            # Get recommendations for this external movie
            rec_url = f"https://api.tmdb.org/3/movie/{movie_id}/recommendations?api_key={API_KEY}"
            rec_data = requests.get(rec_url, timeout=5).json()
            if 'results' in rec_data and rec_data['results']:
                return "global", parse_tmdb_results(rec_data['results'])
            else:
                # If no recommendations, just return similar movies via search
                return "global_related", parse_tmdb_results(search_data['results'][1:6])
    except Exception:
        pass
        
    return "not_found", ([], [], [])

# ------------------ UI ------------------
st.title("🍿 AI-Powered Movie Recommendation System")
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }
    </style>
    """, unsafe_allow_html=True
)

st.write("### Discover your next favorite movie!")

# Simple Text Input Search
user_query = st.text_input("🎬 What movie are you looking for today?", placeholder="Type 'Avatar', 'Batman'...")

if st.button("✨ Show Recommendations", type="primary"):
    if not user_query.strip():
        st.warning("⚠️ Please enter a movie name.")
    else:
        with st.spinner("Analyzing similarities and fetching data..."):
            rec_type, results = recommend(user_query)
            names, posters, details = results
            
            if rec_type == "not_found" or not names:
                st.error("No recommendations found. Try a different movie.")
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.subheader(f"Because you searched for **{user_query.title()}**:")
                
                # Display as a grid using Streamlit columns
                cols = st.columns(5)
                
                for i, col in enumerate(cols):
                    if i < len(names):
                        with col:
                            st.image(posters[i])
                            st.markdown(f"**{names[i]}**")
                            st.caption(f"⭐ {details[i]['rating']} | 📅 {details[i]['release_date']}")
                            with st.popover("Read Overview"):
                                st.write(details[i]['overview'])
