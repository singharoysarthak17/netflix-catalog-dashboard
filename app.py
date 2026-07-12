import streamlit as st
import pandas as pd
import plotly.express as px

# ============================================
# PAGE CONFIG (must be the very first Streamlit command)
# ============================================
st.set_page_config(
    page_title="Netflix Catalog Dashboard",
    page_icon="🎬",
    layout="wide"
)

# ============================================
# LOAD & CLEAN DATA (cached so it doesn't reload every click)
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df = df.dropna(subset=['year_added'])
    df['year_added'] = df['year_added'].astype(int)
    df['genres'] = df['listed_in'].str.split(', ')
    df['countries'] = df['country'].str.split(', ')
    return df

df = load_data()

# ============================================
# SIDEBAR FILTERS
# ============================================
st.sidebar.header("Filters")

year_range = st.sidebar.slider(
    "Year Added",
    int(df['year_added'].min()), int(df['year_added'].max()),
    (2015, int(df['year_added'].max()))
)

type_filter = st.sidebar.multiselect(
    "Content Type",
    options=sorted(df['type'].unique()),
    default=sorted(df['type'].unique())
)

filtered_df = df[
    (df['year_added'].between(*year_range)) &
    (df['type'].isin(type_filter))
]

# Pre-compute exploded versions once, reused across tabs
genre_exploded = filtered_df.explode('genres')
country_exploded = filtered_df.explode('countries')

# ============================================
# TITLE + KPI CARDS
# ============================================
st.title("🎬 Netflix Content Catalog Dashboard")
st.markdown("Explore Netflix's content library by year, genre, country, and rating.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Titles", f"{len(filtered_df):,}")
col2.metric("Movies", f"{(filtered_df['type']=='Movie').sum():,}")
col3.metric("TV Shows", f"{(filtered_df['type']=='TV Show').sum():,}")
col4.metric("Countries", f"{country_exploded['countries'].nunique():,}")

st.divider()

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🌍 Countries & Ratings", "🎭 People"])

# --------------------------------------------
# TAB 1: OVERVIEW
# --------------------------------------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Additions by Year")
        trend = filtered_df.groupby(['year_added', 'type']).size().reset_index(name='count')
        fig = px.line(trend, x='year_added', y='count', color='type', markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Rating Distribution")
        rating_counts = filtered_df['rating'].value_counts().reset_index()
        rating_counts.columns = ['rating', 'count']
        fig = px.pie(rating_counts, names='rating', values='count', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Genres (Treemap)")
    genre_counts = genre_exploded['genres'].value_counts().reset_index().head(15)
    genre_counts.columns = ['genre', 'count']
    fig = px.treemap(genre_counts, path=['genre'], values='count', color='count',
                      color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
# TAB 2: TRENDS
# --------------------------------------------
with tab2:
    st.subheader("Movie vs TV Show Share of Annual Additions (%)")
    type_share = filtered_df.groupby(['year_added', 'type']).size().unstack(fill_value=0)
    type_pct = type_share.div(type_share.sum(axis=1), axis=0) * 100
    type_pct_reset = type_pct.reset_index().melt(id_vars='year_added', var_name='type', value_name='pct')
    fig = px.area(type_pct_reset, x='year_added', y='pct', color='type')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fastest-Growing Genres (Pre vs Post 2018)")
    pre_2018 = genre_exploded[genre_exploded['year_added'] < 2018]['genres'].value_counts()
    post_2018 = genre_exploded[genre_exploded['year_added'] >= 2018]['genres'].value_counts()
    growth = pd.DataFrame({'pre_2018': pre_2018, 'post_2018': post_2018}).fillna(0)
    growth = growth[growth['pre_2018'] > 20]
    growth['growth_pct'] = (growth['post_2018'] - growth['pre_2018']) / growth['pre_2018'] * 100
    top_growth = growth.sort_values('growth_pct', ascending=False).head(10).reset_index()
    top_growth.columns = ['genre', 'pre_2018', 'post_2018', 'growth_pct']
    fig = px.bar(top_growth, x='growth_pct', y='genre', orientation='h',
                 color='growth_pct', color_continuous_scale='Greens')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative Growth: Top 5 Genres Over Time")
    top_5_genres = genre_counts.head(5)['genre'].tolist()
    genre_year = genre_exploded.groupby(['year_added', 'genres']).size().reset_index(name='count')
    genre_year_top5 = genre_year[genre_year['genres'].isin(top_5_genres)].sort_values('year_added')
    genre_year_top5['cumulative'] = genre_year_top5.groupby('genres')['count'].cumsum()
    fig = px.bar(genre_year_top5, x='genres', y='cumulative', color='genres',
                 animation_frame='year_added',
                 range_y=[0, genre_year_top5['cumulative'].max() * 1.1 if len(genre_year_top5) else 10])
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
# TAB 3: COUNTRIES & RATINGS
# --------------------------------------------
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Countries")
        top_countries = country_exploded['countries'].value_counts().head(10).reset_index()
        top_countries.columns = ['country', 'count']
        fig = px.bar(top_countries, x='count', y='country', orientation='h',
                     color='count', color_continuous_scale='Blues')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Rating Mix by Top 5 Countries (%)")
        top_5_countries = country_exploded['countries'].value_counts().head(5).index
        top5_df = country_exploded[country_exploded['countries'].isin(top_5_countries)]
        crosstab = pd.crosstab(top5_df['countries'], top5_df['rating'])
        crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        fig = px.imshow(crosstab_pct, text_auto='.0f', color_continuous_scale='Blues',
                         labels=dict(x="Rating", y="Country", color="% of titles"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Type → Country → Rating Breakdown (Sunburst)")
    sunburst_df = country_exploded[country_exploded['countries'].isin(top_countries['country'])].copy()
    sunburst_df = sunburst_df.dropna(subset=['type', 'countries', 'rating'])
    sunburst_df = sunburst_df[sunburst_df['rating'].str.strip() != '']  # remove blank ratings
    fig = px.sunburst(sunburst_df, path=['type', 'countries', 'rating'])
    fig.update_traces(textinfo="label+percent parent")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
# TAB 4: PEOPLE (Directors & Actors)
# --------------------------------------------
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Most Frequent Directors")
        director_counts = filtered_df['director'].dropna().str.split(', ').explode().value_counts().head(15)
        director_counts = director_counts.reset_index()
        director_counts.columns = ['director', 'count']
        fig = px.bar(director_counts, x='count', y='director', orientation='h',
                     color='count', color_continuous_scale='Purples')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Actors Spanning the Most Genres")
        cast_genre = filtered_df[['cast', 'genres']].dropna().copy()
        cast_genre['cast'] = cast_genre['cast'].str.split(', ')
        cast_genre_exploded = cast_genre.explode('cast').explode('genres')
        genre_diversity = cast_genre_exploded.groupby('cast')['genres'].nunique()
        title_count = cast_genre_exploded.groupby('cast').size()
        genre_diversity = genre_diversity[title_count >= 5].sort_values(ascending=False).head(15)
        genre_diversity = genre_diversity.reset_index()
        genre_diversity.columns = ['actor', 'genre_count']
        fig = px.bar(genre_diversity, x='genre_count', y='actor', orientation='h',
                     color='genre_count', color_continuous_scale='Teal')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# RAW DATA (expandable, at the very bottom)
# ============================================
st.divider()
with st.expander("View Raw Filtered Data"):
    st.dataframe(filtered_df[['title', 'type', 'year_added', 'rating', 'country', 'director']])