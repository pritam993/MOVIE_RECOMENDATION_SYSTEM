
import os
import html
import pickle
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# TMDB credentials: Streamlit Secrets first, local .env second
try:
    TMDB_API_KEY = str(
        st.secrets.get(
            "TMDB_API_KEY",
            os.getenv("TMDB_API_KEY", "")
        )
    ).strip()

    TMDB_ACCESS_TOKEN = str(
        st.secrets.get(
            "TMDB_ACCESS_TOKEN",
            os.getenv("TMDB_ACCESS_TOKEN", "")
        )
    ).strip()

except Exception:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
    TMDB_ACCESS_TOKEN = os.getenv("TMDB_ACCESS_TOKEN", "").strip()


TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

HF_REPO_ID = "pritamkundu/cinematch-files"


# =====================================================
# 2. SESSION STATE DEFAULTS
# =====================================================

defaults = {
    "recommendation_results": None,
    "searched_movie": "",
    "requested_count": 10,
    "surprise_results": None,
    "surprise_movie": "",
    "search_query_val": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# 3. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: #101018;
    color: white;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    color: white !important;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #191923;
}

/* HERO SECTION */

.hero-box {
    background:
        radial-gradient(
            circle at 85% 20%,
            rgba(124,58,237,0.35),
            transparent 35%
        ),
        linear-gradient(
            120deg,
            #38205d,
            #21152f,
            #171722
        );

    border: 1px solid #68439a;
    border-radius: 22px;
    padding: 45px 35px;
    margin-bottom: 30px;
    min-height: 280px;
    box-sizing: border-box;
    width: 100%;
}

.hero-small {
    color: #c4a5ff;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 3px;
    margin-bottom: 18px;
}

.hero-title {
    font-size: clamp(36px, 5vw, 58px);
    font-weight: 850;
    line-height: 1.15;
    color: white;
    margin-bottom: 18px;
}

.hero-description {
    font-size: 17px;
    color: #d0c6dc;
    line-height: 1.8;
}

/* SECTION TITLES */

.section-title {
    font-size: 25px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 18px;
    color: white;
}

/* MOVIE CARDS */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #191923;
    border: 1px solid #343442;
    border-radius: 14px;
}

.movie-title {
    font-size: 16px;
    font-weight: 750;
    color: white;
    min-height: 45px;
    margin-top: 10px;
    line-height: 1.4;
}

.movie-rating {
    color: #facc15;
    font-size: 14px;
    margin-top: 5px;
}

.poster-placeholder {
    height: 270px;
    width: 100%;
    border-radius: 10px;
    background: #252532;
    display: flex;
    justify-content: center;
    align-items: center;
    color: #b5b5c5;
    text-align: center;
    font-size: 15px;
}

/* BUTTONS */

.stButton > button,
.stFormSubmitButton > button {
    background: #7c3aed;
    color: white;
    border: none;
    border-radius: 9px;
    min-height: 43px;
    font-weight: 650;
    transition: 0.2s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #9333ea;
    color: white;
    border: none;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #999999;
    padding: 30px 10px;
    font-size: 13px;
}

@media (max-width: 768px) {
    .hero-box {
        padding: 30px 22px;
        min-height: auto;
    }

    .hero-title {
        font-size: 36px;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. LOAD PICKLE FILES FROM HUGGING FACE
# =====================================================

def find_pickle_file(filename):

    local_path = BASE_DIR / filename

    # Use local file if available
    if local_path.exists():
        return local_path

    # Otherwise download from Hugging Face
    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="dataset"
    )

    return Path(downloaded_path)


@st.cache_resource(show_spinner="Loading CineMatch recommendation model...")
def load_data():

    with open(find_pickle_file("df.pkl"), "rb") as f:
        movie_df = pickle.load(f)

    with open(find_pickle_file("tfidf.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    with open(find_pickle_file("tfidf_matrix.pkl"), "rb") as f:
        matrix = pickle.load(f)

    with open(find_pickle_file("indices.pkl"), "rb") as f:
        movie_indices = pickle.load(f)

    return movie_df, vectorizer, matrix, movie_indices


# IMPORTANT: ACTUALLY LOAD THE DATA
# This was missing in your previous code.

try:

    df, tfidf, tfidf_matrix, indices = load_data()

    if not isinstance(df, pd.DataFrame):
        st.error("Invalid movie dataset format.")
        st.stop()

    required_columns = {"title", "genres"}

    if not required_columns.issubset(df.columns):
        st.error("Dataset must contain title and genres columns.")
        st.stop()

    if tfidf_matrix.shape[0] != len(df):
        st.error("Dataset and TF-IDF matrix dimensions do not match.")
        st.stop()

    if not hasattr(indices, "get"):
        st.error("Invalid movie indices file.")
        st.stop()

except Exception as error:

    st.error("Unable to load CineMatch data from Hugging Face.")
    st.exception(error)
    st.stop()


# =====================================================
# 5. TMDB API AUTHENTICATION
# =====================================================

def tmdb_request(endpoint, params=None):

    request_params = dict(params or {})
    headers = {"accept": "application/json"}

    if TMDB_ACCESS_TOKEN:

        headers["Authorization"] = f"Bearer {TMDB_ACCESS_TOKEN}"

    elif TMDB_API_KEY:

        if TMDB_API_KEY.startswith("eyJ"):
            headers["Authorization"] = f"Bearer {TMDB_API_KEY}"
        else:
            request_params["api_key"] = TMDB_API_KEY

    else:
        return {
            "_error": "TMDB credentials are missing. Check Streamlit Secrets."
        }

    try:

        response = requests.get(
            f"{TMDB_BASE_URL}/{endpoint}",
            params=request_params,
            headers=headers,
            timeout=20
        )

        if not response.ok:

            try:
                error_data = response.json()
                message = error_data.get(
                    "status_message",
                    response.text
                )

            except ValueError:
                message = response.text

            return {
                "_error": f"TMDB error {response.status_code}: {message}"
            }

        return response.json()

    except requests.RequestException as error:

        return {
            "_error": f"Connection error: {error}"
        }


# =====================================================
# 6. GET TRENDING MOVIES
# =====================================================

@st.cache_data(ttl=1800, show_spinner=False)
def get_trending_movies():

    return tmdb_request("trending/movie/week")


# =====================================================
# 7. GET MOVIE POSTERS
# =====================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_poster(movie_name):

    data = tmdb_request(
        "search/movie",
        {
            "query": movie_name,
            "include_adult": False
        }
    )

    if data.get("_error"):
        return None

    results = data.get("results", [])

    if not results:
        return None

    normalized_name = movie_name.strip().lower()

    selected = next(
        (
            movie for movie in results
            if movie.get("title", "").strip().lower() == normalized_name
        ),
        results[0]
    )

    poster_path = selected.get("poster_path")

    if poster_path:
        return TMDB_IMAGE_URL + poster_path

    return None


# =====================================================
# 8. GENRE PROCESSING
# =====================================================

def extract_genres(genre_text):

    if isinstance(genre_text, list):

        return {
            str(genre).strip().lower()
            for genre in genre_text
            if str(genre).strip()
        }

    if not isinstance(genre_text, str):
        return set()

    return {
        genre.strip().lower()
        for genre in genre_text.split()
        if genre.strip()
    }


# =====================================================
# 9. TF-IDF RECOMMENDATION MODEL
# =====================================================

def recommend_movies(movie_name, n=10):

    movie_index = indices.get(movie_name)

    if movie_index is None:
        return pd.DataFrame()

    if isinstance(movie_index, pd.Series):

        if movie_index.empty:
            return pd.DataFrame()

        movie_index = movie_index.iloc[0]

    movie_index = int(movie_index)

    if not 0 <= movie_index < len(df):
        return pd.DataFrame()

    similarity = tfidf_matrix[movie_index].dot(tfidf_matrix.T)

    if hasattr(similarity, "toarray"):
        similarity_scores = similarity.toarray().ravel()
    else:
        similarity_scores = np.asarray(similarity).ravel()

    selected_genres = extract_genres(
        df.iloc[movie_index]["genres"]
    )

    similar_indices = np.argsort(similarity_scores)[::-1]

    recommendations = []

    seen_titles = {
        movie_name.strip().lower()
    }

    for idx in similar_indices:

        if idx == movie_index:
            continue

        score = float(similarity_scores[idx])

        if score < 0.03:
            break

        movie = df.iloc[idx]

        title = str(movie["title"]).strip()

        if not title or title.lower() in seen_titles:
            continue

        candidate_genres = extract_genres(
            movie["genres"]
        )

        if selected_genres and candidate_genres:

            if not selected_genres.intersection(candidate_genres):
                continue

        seen_titles.add(title.lower())

        recommendations.append({
            "index": idx,
            "similarity": score
        })

        if len(recommendations) >= n:
            break

    if not recommendations:
        return pd.DataFrame()

    result = df.iloc[
        [item["index"] for item in recommendations]
    ].copy()

    result["similarity"] = [
        item["similarity"]
        for item in recommendations
    ]

    return result


# =====================================================
# 10. DISPLAY MOVIE CARD
# =====================================================

def display_movie_card(movie, poster=None, key="movie"):

    title = str(
        movie.get("title", "Unknown Movie")
    ).strip()

    safe_title = html.escape(title)

    with st.container(border=True):

        if poster:

            st.image(
                poster,
                use_container_width=True
            )

        else:

            st.markdown(
                '<div class="poster-placeholder">🎞️ Poster unavailable</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            f'<div class="movie-title">{safe_title}</div>',
            unsafe_allow_html=True
        )

        rating = movie.get("vote_average")

        if rating is not None and pd.notna(rating):

            st.markdown(
                f'<div class="movie-rating">⭐ {float(rating):.1f}/10</div>',
                unsafe_allow_html=True
            )

        genre = movie.get("genres", "")

        if isinstance(genre, list):

            st.caption(", ".join(map(str, genre)))

        elif isinstance(genre, str) and genre.strip():

            st.caption(genre)

        with st.expander("Movie Details"):

            overview = movie.get("overview", "")

            if isinstance(overview, str) and overview.strip():
                st.write(overview)
            else:
                st.write("Description unavailable.")

            release_date = movie.get("release_date")

            if release_date:
                st.caption(f"Release date: {release_date}")


# =====================================================
# 11. DISPLAY MOVIE GRID
# =====================================================

def display_movie_grid(movies, prefix):

    if len(movies) == 0:
        st.info("No movies available.")
        return

    posters = [None] * len(movies)

    missing_poster_indices = []

    for i, movie in enumerate(movies):

        poster_path = movie.get("poster_path")

        if poster_path:

            posters[i] = TMDB_IMAGE_URL + poster_path

        else:

            missing_poster_indices.append(i)

    if TMDB_API_KEY or TMDB_ACCESS_TOKEN:

        if missing_poster_indices:

            with st.spinner("Loading movie posters..."):

                with ThreadPoolExecutor(max_workers=8) as executor:

                    missing_posters = list(
                        executor.map(
                            get_movie_poster,
                            [
                                str(movies[i].get("title", ""))
                                for i in missing_poster_indices
                            ]
                        )
                    )

                for index, poster in zip(
                    missing_poster_indices,
                    missing_posters
                ):

                    posters[index] = poster

    for start in range(0, len(movies), 5):

        columns = st.columns(5)

        for offset, movie in enumerate(
            movies[start:start + 5]
        ):

            position = start + offset

            with columns[offset]:

                display_movie_card(
                    movie,
                    posters[position],
                    key=f"{prefix}_{position}"
                )


# =====================================================
# 12. SIDEBAR
# =====================================================

movie_titles = sorted(
    df["title"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

with st.sidebar:

    st.title("🎬 CineMatch")

    st.markdown("---")

    st.write(
        "Discover movies that match your taste "
        "and find something new to watch."
    )

    st.markdown("---")

    st.caption(f"Movies available: {len(movie_titles)}")

    st.caption("Less searching. More watching. ❤️")


# =====================================================
# 13. HERO SECTION
# =====================================================

st.html("""
<div class="hero-box">

    <div class="hero-small">
        WELCOME TO CINEMATCH
    </div>

    <div class="hero-title">
        Less searching.<br>More watching.
    </div>

    <div class="hero-description">
        Discover movies that match your taste.<br>
        Find your next favourite movie today.
    </div>

</div>
""")


# =====================================================
# 14. MOVIE SEARCH & SUGGESTIONS
# =====================================================

st.markdown(
    '<div class="section-title">🔍 Find Your Next Movie</div>',
    unsafe_allow_html=True
)

number_of_movies = st.selectbox(
    "Number of recommendations",
    [5, 10, 15, 20],
    index=1,
    key="recommendation_count"
)


def on_suggestion_click(title):

    results = recommend_movies(
        title,
        n=st.session_state.recommendation_count
    )

    st.session_state.recommendation_results = results

    st.session_state.searched_movie = title

    st.session_state.requested_count = (
        st.session_state.recommendation_count
    )

    st.session_state.surprise_results = None

    st.session_state.surprise_movie = ""

    st.session_state.search_query_val = title


search_query = st.text_input(
    "Search movie",
    value=st.session_state.search_query_val,
    placeholder="Start typing a movie name...",
    label_visibility="collapsed",
    key="search_input_widget"
)

if search_query != st.session_state.search_query_val:

    st.session_state.search_query_val = search_query

query_stripped = (
    st.session_state.search_query_val.strip().casefold()
)

suggestions = []

if query_stripped:

    suggestions = [
        title
        for title in movie_titles
        if query_stripped in title.casefold()
    ][:8]

    if suggestions:

        st.caption(
            "💡 Suggested movies (click to select & search):"
        )

        s_cols = st.columns(2)

        for idx, title in enumerate(suggestions):

            with s_cols[idx % 2]:

                st.button(
                    f"🎬 {title}",
                    key=f"sug_btn_{idx}",
                    use_container_width=True,
                    on_click=on_suggestion_click,
                    args=(title,)
                )

    else:

        st.caption("No matching movies found in database.")


search_button = st.button(
    "✨ Discover Movies",
    use_container_width=True,
    key="discover_movies_btn"
)


# =====================================================
# 15. PROCESS NORMAL SEARCH SUBMISSION
# =====================================================

if search_button:

    typed_query = (
        st.session_state.search_query_val.strip()
    )

    matched_movie = next(
        (
            title for title in movie_titles
            if title.casefold() == typed_query.casefold()
        ),
        None
    )

    if matched_movie is None:

        st.warning(
            "⚠️ Please select a valid movie title "
            "from suggestions or type an exact match."
        )

    else:

        with st.spinner(
            "Finding movies that match your taste..."
        ):

            results = recommend_movies(
                matched_movie,
                n=number_of_movies
            )

        st.session_state.recommendation_results = results

        st.session_state.searched_movie = matched_movie

        st.session_state.requested_count = number_of_movies

        st.session_state.surprise_results = None

        st.session_state.surprise_movie = ""

        st.rerun()


# =====================================================
# 16. DISPLAY NORMAL RECOMMENDATIONS
# =====================================================

results = st.session_state.recommendation_results

if results is not None:

    st.markdown("---")

    st.markdown(
        f'<div class="section-title">🍿 Movies similar to '
        f'{html.escape(st.session_state.searched_movie)}</div>',
        unsafe_allow_html=True
    )

    if results.empty:

        st.info(
            "No recommendations found. Please try another movie."
        )

    else:

        st.write(
            f"Showing {len(results)} recommendations."
        )

        display_movie_grid(
            results.to_dict("records"),
            "recommendation"
        )

        if len(results) < st.session_state.requested_count:

            st.caption(
                f"Found {len(results)} suitable recommendations."
            )


# =====================================================
# 17. TRENDING MOVIES
# =====================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🔥 Trending Movies This Week</div>',
    unsafe_allow_html=True
)

if not TMDB_API_KEY and not TMDB_ACCESS_TOKEN:

    st.warning(
        "TMDB credentials are missing. "
        "Please configure your Streamlit Secrets."
    )

else:

    with st.spinner("Discovering trending movies..."):

        trending_data = get_trending_movies()

    if trending_data.get("_error"):

        st.error(
            f"Unable to load trending movies: "
            f"{trending_data['_error']}"
        )

        st.caption(
            "Check your TMDB credentials and internet connection."
        )

    else:

        trending_movies = trending_data.get("results", [])

        if trending_movies:

            display_movie_grid(
                trending_movies[:10],
                "trending"
            )

        else:

            st.warning(
                "TMDB returned no trending movies."
            )


# =====================================================
# 18. SURPRISE ME
# =====================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🎲 Feeling Lucky?</div>',
    unsafe_allow_html=True
)

st.write(
    "Not sure what to watch? "
    "Let CineMatch choose a movie for you!"
)

surprise_button = st.button(
    "🎲 Surprise Me!",
    use_container_width=True,
    key="surprise_me_btn"
)


# =====================================================
# 19. PROCESS SURPRISE ME
# =====================================================

if surprise_button:

    if not movie_titles:

        st.error("No movies found in your dataset.")

    else:

        random_movie = random.choice(movie_titles)

        with st.spinner(
            f"Finding movies similar to {random_movie}..."
        ):

            surprise_results = recommend_movies(
                random_movie,
                n=st.session_state.recommendation_count
            )

        st.session_state.surprise_movie = random_movie

        st.session_state.surprise_results = surprise_results

        st.session_state.requested_count = (
            st.session_state.recommendation_count
        )

        st.session_state.recommendation_results = None

        st.rerun()


# =====================================================
# 20. DISPLAY SURPRISE RECOMMENDATIONS
# =====================================================

surprise_results = st.session_state.surprise_results

if surprise_results is not None:

    st.markdown("---")

    st.subheader(
        f"🍿 Your surprise movie: "
        f"{st.session_state.surprise_movie}"
    )

    if surprise_results.empty:

        st.info(
            "No similar movies found. Click Surprise Me again!"
        )

    else:

        st.write(
            f"Here are {len(surprise_results)} movies you might enjoy."
        )

        display_movie_grid(
            surprise_results.to_dict("records"),
            "surprise"
        )


# =====================================================
# 21. FOOTER
# =====================================================

st.markdown("---")

st.html("""
<div class="footer">

    Made with ❤️ | CineMatch

    <br><br>

    Movie data and images provided by TMDB.

    <br>

    This product uses the TMDB API but is not endorsed or certified by TMDB.

</div>
""")
