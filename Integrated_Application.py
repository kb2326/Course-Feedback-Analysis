import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import io
import sqlalchemy
import pymysql
import configparser
from sqlalchemy.sql import text

# Download necessary NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    nltk.download('punkt')

# Database Configuration and Connection Functions
def get_db_config():
    # In production, use a more secure approach for credentials
    # This is a simplified example - consider using environment variables or secrets management
    config = {
        'host': 'localhost',
        'user': 'dbuser',
        'password': 'dbpassword',
        'database': 'feedback_db',
        'port': 3306
    }
    return config

def create_db_connection():
    try:
        config = get_db_config()
        connection_string = f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        engine = sqlalchemy.create_engine(connection_string)
        return engine
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

def load_data_from_database(dashboard_type, filters=None):
    engine = create_db_connection()
    if not engine:
        return None
    
    try:
        # Customize these queries based on your database schema
        if dashboard_type == "axe":
            query = """
            SELECT c.client_name, c.client_sector, c.client_type, 
                   f.rating, f.what_did_you_like, f.what_could_be_improved, 
                   f.datetime_originally_submitted, f.country,
                   a.axe_id, a.axe_name, a.axe_category, a.axe_status
            FROM feedback f
            JOIN clients c ON f.client_id = c.client_id
            JOIN axe_assistants a ON f.axe_id = a.axe_id
            WHERE f.feedback_type = 'axe'
            """
        else:  # course
            query = """
            SELECT c.client_name, c.client_sector, c.client_type, 
                   crs.course_id, crs.course_name, crs.course_category, crs.course_status,
                   f.rating, f.what_did_you_like, f.what_could_be_improved, 
                   f.how_many_hours_it_took, f.course_duration_minutes,
                   f.datetime_originally_submitted, f.country
            FROM feedback f
            JOIN clients c ON f.client_id = c.client_id
            JOIN courses crs ON f.course_id = crs.course_id
            WHERE f.feedback_type = 'course'
            """
        
        # Add dynamic filters if provided
        if filters:
            conditions = []
            params = {}
            
            if 'start_date' in filters and 'end_date' in filters:
                conditions.append("f.datetime_originally_submitted BETWEEN :start_date AND :end_date")
                params['start_date'] = filters['start_date']
                params['end_date'] = filters['end_date']
                
            if 'client_name' in filters:
                conditions.append("c.client_name = :client_name")
                params['client_name'] = filters['client_name']
                
            if 'client_sector' in filters:
                conditions.append("c.client_sector = :client_sector")
                params['client_sector'] = filters['client_sector']
                
            if 'client_type' in filters:
                conditions.append("c.client_type = :client_type")
                params['client_type'] = filters['client_type']
                
            if 'country' in filters:
                conditions.append("f.country = :country")
                params['country'] = filters['country']
                
            if dashboard_type == "course" and 'course_category' in filters:
                conditions.append("crs.course_category = :course_category")
                params['course_category'] = filters['course_category']
                
            if dashboard_type == "course" and 'course_status' in filters:
                conditions.append("crs.course_status = :course_status")
                params['course_status'] = filters['course_status']
                
            if dashboard_type == "axe" and 'axe_category' in filters:
                conditions.append("a.axe_category = :axe_category")
                params['axe_category'] = filters['axe_category']
                
            if dashboard_type == "axe" and 'axe_status' in filters:
                conditions.append("a.axe_status = :axe_status")
                params['axe_status'] = filters['axe_status']
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
                
            # Execute with parameters
            df = pd.read_sql(text(query), engine, params=params)
        else:
            # Execute without parameters
            df = pd.read_sql(query, engine)
            
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load and preprocess data
@st.cache_data
def process_data(df):
    """Process the uploaded dataframe"""
    # Data Cleaning
    df.drop_duplicates(inplace=True)
    
    # Handling missing values
    if 'course_id' in df.columns and 'rating' in df.columns:
        df.dropna(subset=['course_id', 'rating'], inplace=True)
    elif 'axe_id' in df.columns and 'rating' in df.columns:
        df.dropna(subset=['axe_id', 'rating'], inplace=True)
    
    # Convert data types
    numeric_columns = [
        'rating', 'course_duration_minutes', 'how_many_hours_it_took'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    date_columns = [
        'last_updated', 'datetime_originally_submitted'
    ]
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Create additional datetime features if date column exists
    if 'datetime_originally_submitted' in df.columns:
        df['submission_month'] = df['datetime_originally_submitted'].dt.to_period('M')
        df['submission_year'] = df['datetime_originally_submitted'].dt.year
        df['submission_quarter'] = df['datetime_originally_submitted'].dt.quarter
    
    # Add sentiment analysis if text columns exist
    if 'what_did_you_like' in df.columns or 'what_could_be_improved' in df.columns:
        sid = SentimentIntensityAnalyzer()
        
        if 'what_did_you_like' in df.columns:
            df['positive_sentiment'] = df['what_did_you_like'].fillna('').apply(
                lambda x: sid.polarity_scores(str(x))['compound'] if len(str(x)) > 0 else np.nan)
        
        if 'what_could_be_improved' in df.columns:
            df['improvement_sentiment'] = df['what_could_be_improved'].fillna('').apply(
                lambda x: sid.polarity_scores(str(x))['compound'] if len(str(x)) > 0 else np.nan)
    
    return df

# Calculate correlation statistics
@st.cache_data
def calculate_correlations(df):
    # Select numeric columns for correlation
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    suggested_cols = ['rating', 'course_duration_minutes', 'how_many_hours_it_took', 
                    'positive_sentiment', 'improvement_sentiment']
    
    # Use intersection of suggested columns and actual numeric columns
    numeric_cols = [col for col in suggested_cols if col in all_numeric_cols]
    
    if len(numeric_cols) < 2:
        return pd.DataFrame(), pd.DataFrame()
    
    # Basic correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Statistical significance of correlations (p-values)
    p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                           index=corr_matrix.index, 
                           columns=corr_matrix.columns)
    
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i != j:  # Skip diagonal
                mask = ~(df[col1].isna() | df[col2].isna())
                if mask.sum() > 2:  # Need at least 3 points for correlation
                    corr, p_value = stats.pearsonr(df[col1][mask], df[col2][mask])
                    p_values.loc[col1, col2] = p_value
                else:
                    p_values.loc[col1, col2] = np.nan
    
    return corr_matrix, p_values

# Topic modeling function
@st.cache_data
def perform_topic_modeling(df, column, n_topics=5):
    # Prepare text data
    texts = df[column].dropna().astype(str).tolist()
    
    if len(texts) < n_topics * 2:
        return None, None, None, None
    
    # Create vectorizer
    min_df = min(5, max(2, len(texts) // 10))  # Adaptive min_df based on dataset size
    vectorizer = CountVectorizer(max_df=0.9, min_df=min_df, stop_words='english')
    dtm = vectorizer.fit_transform(texts)
    
    if dtm.shape[1] < n_topics:  # Not enough features for the requested topics
        return None, None, None, None
    
    # LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    
    # Get top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics_words = []
    
    for topic_idx, topic in enumerate(lda.components_):
        n_top_words = min(10, len(feature_names))
        top_words_idx = topic.argsort()[:-n_top_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics_words.append(top_words)
    
    return topics_words, lda, vectorizer, dtm

# User segmentation function
@st.cache_data
def segment_users(df, n_clusters=4):
    # Select features for clustering
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    suggested_features = ['rating', 'how_many_hours_it_took', 'positive_sentiment', 'improvement_sentiment']
    
    # Use intersection of suggested features and actual numeric columns
    features = [col for col in suggested_features if col in all_numeric_cols]
    
    if len(features) < 2:  # Need at least 2 features for meaningful clustering
        return None, None
    
    # Create a temporary dataframe with these features
    cluster_df = df[features].copy()
    
    # Replace NaN with mean values
    for col in cluster_df.columns:
        cluster_df[col].fillna(cluster_df[col].mean(), inplace=True)
    
    # Normalize data
    cluster_df = (cluster_df - cluster_df.mean()) / cluster_df.std()
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_copy = df.copy()
    df_copy['cluster'] = kmeans.fit_predict(cluster_df)
    
    return df_copy, kmeans.cluster_centers_

# Main Streamlit App
def main():
    st.set_page_config(page_title="Feedback Analysis Dashboard", 
                     layout="wide", 
                     initial_sidebar_state="expanded")
    
    # Initialize session state for navigation
    if 'dashboard_type' not in st.session_state:
        st.session_state.dashboard_type = None
    
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None
    
    # Landing Page
    if st.session_state.dashboard_type is None:
        st.title("Feedback Analysis Dashboard")
        st.write("### Welcome to the Feedback Analysis Platform")
        st.write("Please select the type of feedback you want to analyze:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            axe_card = st.container()
            with axe_card:
                st.markdown("#### AXE Assistant Feedback")
                st.image("https://via.placeholder.com/300x200?text=AXE+Assistant", use_column_width=True)
                st.write("Analyze feedback related to AXE Assistant interactions")
                if st.button("Access AXE Assistant Dashboard", key="axe_btn"):
                    st.session_state.dashboard_type = "axe"
                    st.experimental_rerun()
        
        with col2:
            course_card = st.container()
            with course_card:
                st.markdown("#### Course Assistant Feedback")
                st.image("https://via.placeholder.com/300x200?text=Course+Assistant", use_column_width=True)
                st.write("Analyze feedback related to course delivery and content")
                if st.button("Access Course Assistant Dashboard", key="course_btn"):
                    st.session_state.dashboard_type = "course"
                    st.experimental_rerun()
        
        # Add some information about the platform
        st.markdown("---")
        st.markdown("### About this platform")
        st.write("""
        This analytics platform provides in-depth analysis of feedback data using advanced
        techniques including sentiment analysis, topic modeling, and user segmentation.
        
        Upload your CSV data or connect to the database to get started with your analysis.
        """)
        
        return
    
    # Dashboard header based on selected type
    if st.session_state.dashboard_type == "axe":
        st.title("AXE Assistant Feedback Analysis Dashboard")
    else:
        st.title("Course Feedback Analysis Dashboard")
    
    # Add a way to go back to the landing page
    if st.sidebar.button("Return to Main Menu"):
        st.session_state.dashboard_type = None
        st.experimental_rerun()
    
    # Run NLTK downloads
    try:
        download_nltk_data()
    except Exception as e:
        st.warning(f"Could not download NLTK data: {e}. Some analysis features may be limited.")
    
    # Data Source Selection
    st.sidebar.header("Data Source")
    data_source = st.sidebar.radio("Select data source:", ["Upload CSV", "Connect to Database"])
    st.session_state.data_source = data_source
    
    # Process data based on selected source
    if data_source == "Upload CSV":
        # File upload section
        st.subheader("Upload Your Feedback Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is None:
            st.info("Please upload a CSV file to get started.")
            
            # Show expected CSV format
            st.subheader("Expected CSV Format")
            if st.session_state.dashboard_type == "course":
                st.write("""
                Your CSV should include some of these columns:
                - course_id: Unique identifier for the course
                - course_name: Name of the course
                - course_category: Category of the course
                - client_name: Name of the client
                - client_sector: Client industry sector
                - client_type: Type of client (e.g., Corporate, Individual, Government)
                - course_status: Status of the course (e.g., Active, Completed, Cancelled)
                - rating: Numeric rating (e.g., 1-5)
                - country: Country of the participant
                - course_duration_minutes: Course duration in minutes
                - how_many_hours_it_took: Hours spent by participant
                - what_did_you_like: Text feedback on positives
                - what_could_be_improved: Text feedback on improvements
                - datetime_originally_submitted: When feedback was submitted
                """)
            else:  # AXE
                st.write("""
                Your CSV should include some of these columns:
                - axe_id: Unique identifier for the AXE assistant
                - axe_name: Name of the AXE assistant
                - axe_category: Category of the AXE assistant
                - client_name: Name of the client
                - client_sector: Client industry sector
                - client_type: Type of client (e.g., Corporate, Individual, Government)
                - axe_status: Status of the AXE assistant (e.g., Active, Inactive)
                - rating: Numeric rating (e.g., 1-5)
                - country: Country of the participant
                - what_did_you_like: Text feedback on positives
                - what_could_be_improved: Text feedback on improvements
                - datetime_originally_submitted: When feedback was submitted
                """)
            
            # Show sample data structure
            if st.session_state.dashboard_type == "course":
                sample_df = pd.DataFrame({
                    'course_id': ['C001', 'C002', 'C001'],
                    'course_name': ['Python Basics', 'Data Analysis', 'Python Basics'],
                    'course_category': ['Programming', 'Data Science', 'Programming'],
                    'client_name': ['ABC Corp', 'XYZ Ltd', 'DEF Inc'],
                    'client_sector': ['Technology', 'Finance', 'Healthcare'],
                    'client_type': ['Corporate', 'Government', 'Individual'],
                    'course_status': ['Completed', 'Active', 'Completed'],
                    'rating': [4.5, 3.8, 5.0],
                    'country': ['USA', 'UK', 'Canada'],
                    'course_duration_minutes': [120, 180, 120],
                    'how_many_hours_it_took': [3, 4, 2.5],
                    'what_did_you_like': ['Great examples', 'Comprehensive content', 'Clear explanations'],
                    'what_could_be_improved': ['More exercises', 'Faster pace', 'Nothing'],
                    'datetime_originally_submitted': ['2023-01-15', '2023-02-20', '2023-03-10']
                })
            else:  # AXE
                sample_df = pd.DataFrame({
                    'axe_id': ['A001', 'A002', 'A001'],
                    'axe_name': ['Support Bot', 'Sales Assistant', 'Support Bot'],
                    'axe_category': ['Customer Support', 'Sales', 'Customer Support'],
                    'client_name': ['ABC Corp', 'XYZ Ltd', 'DEF Inc'],
                    'client_sector': ['Technology', 'Finance', 'Healthcare'],
                    'client_type': ['Corporate', 'Government', 'Individual'],
                    'axe_status': ['Active', 'Active', 'Inactive'],
                    'rating': [4.5, 3.8, 5.0],
                    'country': ['USA', 'UK', 'Canada'],
                    'what_did_you_like': ['Quick responses', 'Helpful suggestions', 'Accurate information'],
                    'what_could_be_improved': ['More personalization', 'Better understanding', 'Nothing'],
                    'datetime_originally_submitted': ['2023-01-15', '2023-02-20', '2023-03-10']
                })
            
            st.write("Sample data structure:")
            st.dataframe(sample_df.head())
            return
        
        # Process the uploaded file
        try:
            df_original = pd.read_csv(uploaded_file, on_bad_lines='skip')
            
            # Display raw data table
            with st.expander("View Raw Data"):
                st.dataframe(df_original.head(100))
                st.write(f"Total rows: {len(df_original)}")
                st.write(f"Columns: {', '.join(df_original.columns)}")
            
            # Process the data
            df = process_data(df_original)
            
            if df.empty:
                st.error("The processed dataset is empty. Please check your CSV format.")
                return
            
        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")
            return
    else:  # Database connection
        st.subheader("Database Connection")
        
        with st.spinner("Connecting to database..."):
            df_original = load_data_from_database(st.session_state.dashboard_type)
            
        if df_original is None:
            st.error("Failed to load data from database. Please check your connection.")
            return
        elif df_original.empty:
            st.warning("No data retrieved from database. Please check your query or filters.")
            return
            
        # Display raw data preview
        with st.expander("View Raw Data"):
            st.dataframe(df_original.head(100))
            st.write(f"Total rows: {len(df_original)}")
            st.write(f"Columns: {', '.join(df_original.columns)}")
        
        # Process data
        df = process_data(df_original)
        
        if df.empty:
            st.error("The processed dataset is empty. Please check your data source.")
            return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter - if datetime column exists
    if 'datetime_originally_submitted' in df.columns and df['datetime_originally_submitted'].notna().any():
        min_date = df['datetime_originally_submitted'].min().date()
        max_date = df['datetime_originally_submitted'].max().date()
        
        # Fix date input by ensuring default values are within range
        default_start = min_date
        default_end = max_date
        
        start_date, end_date = st.sidebar.date_input(
            "Select Date Range",
            value=[default_start, default_end],
            min_value=min_date,
            max_value=max_date
        )
        
        # Apply date filter
        filtered_df = df[(df['datetime_originally_submitted'].dt.date >= start_date) & 
                        (df['datetime_originally_submitted'].dt.date <= end_date)]
    else:
        filtered_df = df
        st.sidebar.info("No date column found for filtering")
    
    # Client name filter - new addition
    if 'client_name' in df.columns and df['client_name'].notna().any():
        all_clients = ["All Clients"] + sorted(df['client_name'].dropna().unique().tolist())
        selected_client = st.sidebar.selectbox("Client Name", all_clients)
        
        if selected_client != "All Clients":
            filtered_df = filtered_df[filtered_df['client_name'] == selected_client]
    
    # Course/AXE category filter - if column exists
    category_column = 'course_category' if st.session_state.dashboard_type == "course" else 'axe_category'
    if category_column in df.columns and df[category_column].notna().any():
        all_categories = ["All Categories"] + sorted(df[category_column].dropna().unique().tolist())
        selected_category = st.sidebar.selectbox(
            "Course Category" if st.session_state.dashboard_type == "course" else "AXE Category", 
            all_categories
        )
        
        if selected_category != "All Categories":
            filtered_df = filtered_df[filtered_df[category_column] == selected_category]
    
    # Client sector filter - if column exists
    if 'client_sector' in df.columns and df['client_sector'].notna().any():
        all_sectors = ["All Sectors"] + sorted(df['client_sector'].dropna().unique().tolist())
        selected_sector = st.sidebar.selectbox("Client Sector", all_sectors)
        
        if selected_sector != "All Sectors":
            filtered_df = filtered_df[filtered_df['client_sector'] == selected_sector]
    
    # Client type filter - if column exists
    if 'client_type' in df.columns and df['client_type'].notna().any():
        all_client_types = ["All Client Types"] + sorted(df['client_type'].dropna().unique().tolist())
        selected_client_type = st.sidebar.selectbox("Client Type", all_client_types)
        
        if selected_client_type != "All Client Types":
            filtered_df = filtered_df[filtered_df['client_type'] == selected_client_type]
    
    # Course/AXE status filter - if column exists
    status_column = 'course_status' if st.session_state.dashboard_type == "course" else 'axe_status'
    if status_column in df.columns and df[status_column].notna().any():
        all_statuses = ["All Statuses"] + sorted(df[status_column].dropna().unique().tolist())
        selected_status = st.sidebar.selectbox(
            "Course Status" if st.session_state.dashboard_type == "course" else "AXE Status", 
            all_statuses
        )
        
        if selected_status != "All Statuses":
            filtered_df = filtered_df[filtered_df[status_column] == selected_status]
    
    # Country filter - if column exists
    if 'country' in df.columns and df['country'].notna().any():
        top_countries = ["All Countries"] + df['country'].value_counts().head(10).index.tolist()
        selected_country = st.sidebar.selectbox("Country", top_countries)
        
        if selected_country != "All Countries":
            filtered_df = filtered_df[filtered_df['country'] == selected_country]
    
    # Check if filtered dataframe is empty
    if filtered_df.empty:
        st.warning("No data available with the current filters. Please adjust your filters.")
        return
    
    # Main dashboard tabs
    tab_list = ["Overview"]
    
    # Only add tabs for which we have relevant data
    item_column = 'course_name' if st.session_state.dashboard_type == "course" else 'axe_name'
    if item_column in filtered_df.columns and 'rating' in filtered_df.columns:
        tab_list.append("Performance Analysis")
    
    if ('positive_sentiment' in filtered_df.columns or 'improvement_sentiment' in filtered_df.columns) and \
       ('what_did_you_like' in filtered_df.columns or 'what_could_be_improved' in filtered_df.columns):
        tab_list.append("Sentiment Analysis")
    
    if 'what_did_you_like' in filtered_df.columns or 'what_could_be_improved' in filtered_df.columns:
        tab_list.append("Topic Modeling")
    
    if filtered_df.select_dtypes(include=[np.number]).shape[1] >= 2:
        tab_list.append("Statistical Analysis")
    
    if filtered_df.select_dtypes(include=[np.number]).shape[1] >= 2:
        tab_list.append("User Segmentation")
    
    tabs = st.tabs(tab_list)
    
    # Overview tab is always available
    with tabs[0]:
        st.header("Dashboard Overview")
        
        # KPI metrics row
        columns = st.columns(4)
        
        metric_idx = 0
        
        if 'rating' in filtered_df.columns:
            avg_rating = filtered_df['rating'].mean()
            columns[metric_idx].metric("Average Rating", f"{avg_rating:.2f} / 5.0")
            metric_idx += 1
        
        if item_column in filtered_df.columns:
            total_items = filtered_df[item_column].nunique()
            columns[metric_idx].metric(
                "Total Courses" if st.session_state.dashboard_type == "course" else "Total AXE Assistants", 
                total_items
            )
            metric_idx += 1
        
        # Always show total entries
        columns[metric_idx].metric("Total Entries", len(filtered_df))
        metric_idx += 1
        
        if 'positive_sentiment' in filtered_df.columns and filtered_df['positive_sentiment'].notna().any():
            avg_sentiment = filtered_df['positive_sentiment'].mean()
            columns[metric_idx].metric("Average Sentiment", f"{avg_sentiment:.2f}")
        
        # High-level charts
        col1, col2 = st.columns(2)
        
        with col1:
            if 'datetime_originally_submitted' in filtered_df.columns and filtered_df['datetime_originally_submitted'].notna().any():
                st.subheader("Feedback Volume Over Time")
                # Group by month and count
                monthly_counts = filtered_df.groupby(pd.Grouper(key='datetime_originally_submitted', freq='M')).size()
                time_data = pd.DataFrame({'date': monthly_counts.index, 'count': monthly_counts.values})
                fig = px.line(time_data, x='date', y='count', title="Monthly Feedback Volume")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No date data available for time series analysis")
        
        with col2:
            if 'rating' in filtered_df.columns:
                st.subheader("Rating Distribution")
                fig = px.histogram(filtered_df, x='rating', nbins=10, 
                                 title="Distribution of Ratings", 
                                 color_discrete_sequence=['#3366CC'])
                fig.update_layout(xaxis_title="Rating", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rating data available for distribution analysis")
        
                    # Client distribution pie chart (NEW)
        if 'client_name' in filtered_df.columns and filtered_df['client_name'].notna().any():
            st.subheader("Feedback Distribution by Client")
            
            # Get top clients by feedback count
            client_counts = filtered_df['client_name'].value_counts().nlargest(8)
            other_count = filtered_df['client_name'].value_counts().iloc[8:].sum() if len(filtered_df['client_name'].value_counts()) > 8 else 0
            
            # Create data for pie chart
            pie_data = pd.DataFrame({
                'client': list(client_counts.index) + ['Other Clients'] if other_count > 0 else list(client_counts.index),
                'count': list(client_counts.values) + [other_count] if other_count > 0 else list(client_counts.values)
            })
            
            fig = px.pie(pie_data, values='count', names='client', 
                       title='Feedback Distribution by Client',
                       hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    
    # Performance Analysis tab
    if "Performance Analysis" in tab_list:
        with tabs[tab_list.index("Performance Analysis")]:
            item_label = "Course" if st.session_state.dashboard_type == "course" else "AXE Assistant"
            st.header(f"{item_label} Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Top Highest Rated {item_label}s")
                # Check if we have enough unique items
                if filtered_df[item_column].nunique() > 0:
                    top_items = filtered_df.groupby(item_column)['rating'].agg(['mean', 'count']).sort_values('mean', ascending=False)
                    n_top = min(10, len(top_items))
                    if n_top > 0:
                        top_items = top_items.head(n_top).reset_index()
                        fig = px.bar(top_items, x='mean', y=item_column, 
                                   labels={'mean': 'Average Rating', item_column: item_label},
                                   title=f"Top {n_top} Highest Rated {item_label}s",
                                   color='mean',
                                   color_continuous_scale=px.colors.sequential.Blues,
                                   hover_data=['count'])
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"Not enough {item_label.lower()} data for analysis")
                else:
                    st.info(f"Not enough {item_label.lower()}s to display ratings")
            
            with col2:
                st.subheader(f"Bottom Lowest Rated {item_label}s")
                min_reviews = st.slider("Minimum number of reviews", 1, 20, 3, key="min_reviews_slider")
                # Check if we have items with enough reviews
                items_with_min_reviews = filtered_df.groupby(item_column).filter(lambda x: len(x) >= min_reviews)
                
                if not items_with_min_reviews.empty and items_with_min_reviews[item_column].nunique() > 0:
                    bottom_items = items_with_min_reviews.groupby(item_column)['rating'].agg(['mean', 'count']).sort_values('mean')
                    n_bottom = min(10, len(bottom_items))
                    if n_bottom > 0:
                        bottom_items = bottom_items.head(n_bottom).reset_index()
                        fig = px.bar(bottom_items, x='mean', y=item_column, 
                                   labels={'mean': 'Average Rating', item_column: item_label},
                                   title=f"Bottom {n_bottom} Lowest Rated {item_label}s (min {min_reviews} reviews)",
                                   color='mean',
                                   color_continuous_scale=px.colors.sequential.Reds_r,
                                   hover_data=['count'])
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"Not enough {item_label.lower()} data with the minimum review threshold")
                else:
                    st.info(f"No {item_label.lower()}s with at least {min_reviews} reviews found")
            
            # Only show category and sector analysis if those columns exist
            category_column = 'course_category' if st.session_state.dashboard_type == "course" else 'axe_category'
            if category_column in filtered_df.columns or 'client_sector' in filtered_df.columns:
                col1, col2 = st.columns(2)
                
                if category_column in filtered_df.columns and filtered_df[category_column].notna().any():
                    with col1:
                        category_label = "Category" if st.session_state.dashboard_type == "course" else "AXE Category"
                        st.subheader(f"Performance by {category_label}")
                        category_perf = filtered_df.groupby(category_column)['rating'].agg(['mean', 'count']).sort_values('mean', ascending=False).reset_index()
                        fig = px.bar(category_perf, x=category_column, y='mean', 
                                   labels={'mean': 'Average Rating', category_column: category_label, 'count': 'Number of Reviews'},
                                   title=f"Average Rating by {item_label} {category_label}",
                                   color='mean',
                                   color_continuous_scale=px.colors.sequential.Viridis,
                                   hover_data=['count'])
                        st.plotly_chart(fig, use_container_width=True)
                
                if 'client_sector' in filtered_df.columns and filtered_df['client_sector'].notna().any():
                    with col2:
                        st.subheader("Performance by Client Sector")
                        sector_perf = filtered_df.groupby('client_sector')['rating'].agg(['mean', 'count']).sort_values('mean', ascending=False).reset_index()
                        fig = px.bar(sector_perf, x='client_sector', y='mean', 
                                   labels={'mean': 'Average Rating', 'client_sector': 'Sector', 'count': 'Number of Reviews'},
                                   title="Average Rating by Client Sector",
                                   color='mean',
                                   color_continuous_scale=px.colors.sequential.Plasma,
                                   hover_data=['count'])
                        st.plotly_chart(fig, use_container_width=True)
            
            # NEW: Performance by client_name (if exists)
            if 'client_name' in filtered_df.columns and filtered_df['client_name'].notna().any():
                st.subheader("Performance by Client")
                client_perf = filtered_df.groupby('client_name')['rating'].agg(['mean', 'count']).sort_values('mean', ascending=False).reset_index()
                
                # Limit to top 15 clients for better visualization
                top_clients = min(15, len(client_perf))
                client_perf = client_perf.head(top_clients)
                
                fig = px.bar(client_perf, x='client_name', y='mean', 
                           labels={'mean': 'Average Rating', 'client_name': 'Client', 'count': 'Number of Reviews'},
                           title=f"Average Rating by Top {top_clients} Clients",
                           color='mean',
                           color_continuous_scale=px.colors.sequential.Blues,
                           hover_data=['count'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Monthly trend - only if we have enough data points
            if 'datetime_originally_submitted' in filtered_df.columns and filtered_df['datetime_originally_submitted'].notna().sum() > 1:
                st.subheader("Rating Trends Over Time")
                # Group by month and calculate mean rating
                monthly_data = filtered_df.groupby(pd.Grouper(key='datetime_originally_submitted', freq='M'))['rating'].mean()
                if len(monthly_data) > 1:  # Check if we have at least 2 months
                    monthly_ratings = pd.DataFrame({'datetime_originally_submitted': monthly_data.index, 'rating': monthly_data.values})
                    
                    fig = px.line(monthly_ratings, x='datetime_originally_submitted', y='rating', 
                                title="Monthly Average Ratings",
                                labels={'datetime_originally_submitted': 'Month', 'rating': 'Average Rating'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough time periods for trend analysis")
            
            # Course duration vs rating - only for course feedback
            if st.session_state.dashboard_type == "course" and 'course_duration_minutes' in filtered_df.columns and 'rating' in filtered_df.columns and \
               filtered_df['course_duration_minutes'].notna().any() and filtered_df['rating'].notna().any():
                st.subheader("Course Duration vs. Rating")
                scatter_df = filtered_df[filtered_df['course_duration_minutes'].notna()]
                if not scatter_df.empty:
                    fig = px.scatter(scatter_df, x='course_duration_minutes', y='rating', 
                                   color='course_category' if 'course_category' in scatter_df.columns else None,
                                   title="Course Duration vs. Rating",
                                   labels={'course_duration_minutes': 'Course Duration (minutes)', 'rating': 'Rating'},
                                   trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical Analysis tab
    if "Statistical Analysis" in tab_list:
        with tabs[tab_list.index("Statistical Analysis")]:
            st.header("Statistical Analysis")
            
            # Check if we have enough numerical data
            numeric_cols = ['rating', 'course_duration_minutes', 'how_many_hours_it_took', 
                          'positive_sentiment', 'improvement_sentiment']
            
            # Filter to columns that actually exist in our data
            numeric_cols = [col for col in numeric_cols if col in filtered_df.columns]
            
            if len(numeric_cols) < 2 or filtered_df[numeric_cols].count().sum() < 10:  # Arbitrary threshold
                st.info("Not enough numerical data for statistical analysis")
            else:
                # Correlation analysis
                st.subheader("Correlation Analysis")
                
                try:
                    corr_matrix, p_values = calculate_correlations(filtered_df)
                    
                    # Check if correlation matrix is valid
                    if corr_matrix.empty or corr_matrix.shape[0] < 2 or corr_matrix.isna().all().all():
                        st.info("Insufficient data to calculate correlations")
                    else:
                        # Heatmap of correlations
                        fig = px.imshow(corr_matrix,
                                    x=corr_matrix.columns,
                                    y=corr_matrix.columns,
                                    color_continuous_scale=px.colors.diverging.RdBu_r,
                                    title="Correlation Matrix of Key Metrics",
                                    range_color=[-1, 1])
                        
                        # Add text annotations
                        annotations = []
                        for i, row in enumerate(corr_matrix.index):
                            for j, col in enumerate(corr_matrix.columns):
                                # Handle potential NaN values
                                if pd.isna(corr_matrix.iloc[i, j]) or (i != j and pd.isna(p_values.iloc[i, j])):
                                    text = "N/A"
                                elif i != j:
                                    text = f"{corr_matrix.iloc[i, j]:.2f}<br>p={p_values.iloc[i, j]:.3f}"
                                else:
                                    text = f"{corr_matrix.iloc[i, j]:.2f}"
                                
                                annotations.append(
                                    dict(
                                        x=j,
                                        y=i,
                                        text=text,
                                        showarrow=False,
                                        font=dict(color="white" if not pd.isna(corr_matrix.iloc[i, j]) and abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                                    )
                                )
                        
                        fig.update_layout(annotations=annotations)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Interpret strong correlations
                        st.subheader("Key Insights from Correlations")
                        
                        strong_correlations = []
                        for i, row in enumerate(corr_matrix.index):
                            for j, col in enumerate(corr_matrix.columns):
                                # Only include valid correlations (not NaN)
                                if (i < j and 
                                    not pd.isna(corr_matrix.iloc[i, j]) and 
                                    not pd.isna(p_values.iloc[i, j]) and
                                    abs(corr_matrix.iloc[i, j]) > 0.3 and 
                                    p_values.iloc[i, j] < 0.05):
                                    corr_value = corr_matrix.iloc[i, j]
                                    direction = "positive" if corr_value > 0 else "negative"
                                    strength = "strong" if abs(corr_value) > 0.5 else "moderate"
                                    strong_correlations.append((row, col, corr_value, direction, strength, p_values.iloc[i, j]))
                        
                        if strong_correlations:
                            for corr in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
                                var1, var2, corr_val, direction, strength, p_val = corr
                                st.write(f"*{var1}* and *{var2}* have a {strength} {direction} correlation (r = {corr_val:.2f}, p = {p_val:.3f})")
                        else:
                            st.write("No statistically significant strong correlations found")
                
                except Exception as e:
                    st.error(f"Error calculating correlations: {e}")
                
                # Statistical significance testing
                st.subheader("Statistical Significance Testing")
                
                # Add client_name to the categorical variables list
                categorical_vars = [col for col in ["course_category" if st.session_state.dashboard_type == "course" else "axe_category", 
                                                  "client_sector", "country", "client_type", 
                                                  "course_status" if st.session_state.dashboard_type == "course" else "axe_status",
                                                  "client_name"] 
                                  if col in filtered_df.columns and filtered_df[col].notna().any()]
                
                if not categorical_vars:
                    st.info("No categorical variables available for group comparison")
                else:
                    comparison_var = st.selectbox("Select variable to compare across groups:", categorical_vars, key="comparison_var")
                    
                    # Only show numerical variables that have data
                    numerical_vars = [col for col in ["rating", "positive_sentiment", "improvement_sentiment", "how_many_hours_it_took"]
                                    if col in filtered_df.columns and filtered_df[col].notna().any()]
                    
                    if not numerical_vars:
                        st.info("No numerical variables available for comparison")
                    else:
                        target_var = st.selectbox("Select target variable:", numerical_vars, key="target_var")
                        
                        # Calculate group statistics
                        try:
                            group_stats = filtered_df.groupby(comparison_var)[target_var].agg(['mean', 'std', 'count']).reset_index()
                            group_stats = group_stats[group_stats['count'] >= 5]  # Filter groups with too few samples
                            
                            if len(group_stats) > 1:
                                # ANOVA test if more than 2 groups
                                groups = [filtered_df[filtered_df[comparison_var] == group][target_var].dropna() 
                                        for group in group_stats[comparison_var]]
                                
                                # Ensure all groups have data
                                groups = [group for group in groups if len(group) >= 5]
                                
                                if len(groups) > 1:  # Need at least 2 groups for ANOVA
                                    f_stat, p_val = stats.f_oneway(*groups)
                                    
                                    st.write(f"ANOVA test result: F = {f_stat:.2f}, p-value = {p_val:.4f}")
                                    if p_val < 0.05:
                                        st.write(f"There is a statistically significant difference in {target_var} between {comparison_var} groups")
                                    else:
                                        st.write(f"No statistically significant difference in {target_var} between {comparison_var} groups")
                                    
                                    # Visualize group comparison
                                    fig = px.box(filtered_df, x=comparison_var, y=target_var, 
                                              title=f"Distribution of {target_var} by {comparison_var}",
                                              points="all", notched=True)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Bar chart with error bars for mean values
                                    fig = px.bar(group_stats, x=comparison_var, y='mean', 
                                              error_y='std', 
                                              title=f"Mean {target_var} by {comparison_var} with Standard Deviation",
                                              color='mean',
                                              color_continuous_scale=px.colors.sequential.Viridis,
                                              hover_data=['count'])
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("Not enough groups with sufficient data for ANOVA test")
                            else:
                                st.info("Not enough groups with sufficient data for comparison")
                        except Exception as e:
                            st.error(f"Error calculating group statistics: {e}")
    
    # Sentiment Analysis tab
    if "Sentiment Analysis" in tab_list:
        with tabs[tab_list.index("Sentiment Analysis")]:
            st.header("Sentiment Analysis")
            
            # Check which sentiment columns we have
            sentiment_cols = []
            if 'positive_sentiment' in filtered_df.columns and filtered_df['positive_sentiment'].notna().any():
                sentiment_cols.append('positive_sentiment')
            if 'improvement_sentiment' in filtered_df.columns and filtered_df['improvement_sentiment'].notna().any():
                sentiment_cols.append('improvement_sentiment')
            
            if not sentiment_cols:
                st.info("No sentiment data available for analysis")
            else:
                col1, col2 = st.columns(2)
                
                if 'positive_sentiment' in sentiment_cols:
                    with col1:
                        st.subheader("Positive Feedback Sentiment Distribution")
                        fig = px.histogram(filtered_df.dropna(subset=['positive_sentiment']), 
                                         x='positive_sentiment', 
                                         title="Distribution of Positive Feedback Sentiment",
                                         labels={'positive_sentiment': 'Sentiment Score'},
                                         color_discrete_sequence=['#2E8B57'])
                        st.plotly_chart(fig, use_container_width=True)
                
                if 'improvement_sentiment' in sentiment_cols:
                    with col2:
                        st.subheader("Improvement Suggestions Sentiment Distribution")
                        fig = px.histogram(filtered_df.dropna(subset=['improvement_sentiment']), 
                                         x='improvement_sentiment', 
                                         title="Distribution of Improvement Feedback Sentiment",
                                         labels={'improvement_sentiment': 'Sentiment Score'},
                                         color_discrete_sequence=['#B22222'])
                        st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment vs rating - only show if we have both columns with data
                if 'positive_sentiment' in sentiment_cols and 'rating' in filtered_df.columns:
                    st.subheader("Sentiment Score vs. Rating")
                    scatter_df = filtered_df.dropna(subset=['positive_sentiment', 'rating'])
                    if not scatter_df.empty:
                        fig = px.scatter(scatter_df, x='positive_sentiment', y='rating', 
                                       title="Positive Sentiment vs. Rating",
                                       labels={'positive_sentiment': 'Sentiment Score', 'rating': 'Rating'},
                                       trendline="ols",
                                       color=category_column if category_column in scatter_df.columns else None)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment by category
                if category_column in filtered_df.columns and filtered_df[category_column].notna().any():
                    category_label = "Category" if st.session_state.dashboard_type == "course" else "AXE Category"
                    sentiment_by_cat = filtered_df.groupby(category_column)[sentiment_cols].mean().reset_index()
                    
                    fig = px.bar(sentiment_by_cat, x=category_column, y=sentiment_cols,
                               barmode='group',
                               title=f"Average Sentiment by {category_label}",
                               labels={'value': 'Average Sentiment', category_column: category_label, 'variable': 'Sentiment Type'},
                               color_discrete_map={'positive_sentiment': '#2E8B57', 'improvement_sentiment': '#B22222'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # NEW: Sentiment by client_name
                if 'client_name' in filtered_df.columns and filtered_df['client_name'].notna().any():
                    # Get top N clients by data volume to avoid overcrowding
                    top_clients = filtered_df['client_name'].value_counts().nlargest(8).index.tolist()
                    client_df = filtered_df[filtered_df['client_name'].isin(top_clients)]
                    
                    sentiment_by_client = client_df.groupby('client_name')[sentiment_cols].mean().reset_index()
                    
                    fig = px.bar(sentiment_by_client, x='client_name', y=sentiment_cols,
                               barmode='group',
                               title="Average Sentiment by Client",
                               labels={'value': 'Average Sentiment', 'client_name': 'Client', 'variable': 'Sentiment Type'},
                               color_discrete_map={'positive_sentiment': '#2E8B57', 'improvement_sentiment': '#B22222'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment by client type
                if 'client_type' in filtered_df.columns and filtered_df['client_type'].notna().any():
                    sentiment_by_client = filtered_df.groupby('client_type')[sentiment_cols].mean().reset_index()
                    
                    fig = px.bar(sentiment_by_client, x='client_type', y=sentiment_cols,
                               barmode='group',
                               title="Average Sentiment by Client Type",
                               labels={'value': 'Average Sentiment', 'client_type': 'Client Type', 'variable': 'Sentiment Type'},
                               color_discrete_map={'positive_sentiment': '#2E8B57', 'improvement_sentiment': '#B22222'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment by status
                status_column = 'course_status' if st.session_state.dashboard_type == "course" else 'axe_status'
                if status_column in filtered_df.columns and filtered_df[status_column].notna().any():
                    status_label = "Course Status" if st.session_state.dashboard_type == "course" else "AXE Status"
                    sentiment_by_status = filtered_df.groupby(status_column)[sentiment_cols].mean().reset_index()
                    
                    fig = px.bar(sentiment_by_status, x=status_column, y=sentiment_cols,
                               barmode='group',
                               title=f"Average Sentiment by {status_label}",
                               labels={'value': 'Average Sentiment', status_column: status_label, 'variable': 'Sentiment Type'},
                               color_discrete_map={'positive_sentiment': '#2E8B57', 'improvement_sentiment': '#B22222'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Word clouds - only if text columns have data
                text_cols = []
                if 'what_did_you_like' in filtered_df.columns and filtered_df['what_did_you_like'].notna().any():
                    text_cols.append(('what_did_you_like', 'Positive Feedback', 'viridis'))
                if 'what_could_be_improved' in filtered_df.columns and filtered_df['what_could_be_improved'].notna().any():
                    text_cols.append(('what_could_be_improved', 'Improvement Suggestions', 'plasma'))
                    
                if text_cols:
                    st.subheader("Word Clouds")
                    columns = st.columns(len(text_cols))
                    
                    for i, (col_name, title, colormap) in enumerate(text_cols):
                        with columns[i]:
                            st.write(f"*Word Cloud - {title}*")
                            all_text = ' '.join(filtered_df[col_name].dropna().astype(str))
                            if len(all_text) > 10:  # Only generate if we have enough text
                                try:
                                    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                                     colormap=colormap, max_words=100).generate(all_text)
                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    ax.imshow(wordcloud, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error generating word cloud: {e}")
                            else:
                                st.info("Not enough text data for word cloud")
    
    # Topic Modeling tab
    if "Topic Modeling" in tab_list:
        with tabs[tab_list.index("Topic Modeling")]:
            st.header("Topic Modeling Analysis")
            
            # Choose column for topic modeling
            text_columns = []
            if 'what_did_you_like' in filtered_df.columns and filtered_df['what_did_you_like'].notna().any():
                text_columns.append(('what_did_you_like', 'Positive Feedback'))
            if 'what_could_be_improved' in filtered_df.columns and filtered_df['what_could_be_improved'].notna().any():
                text_columns.append(('what_could_be_improved', 'Improvement Suggestions'))
            
            if not text_columns:
                st.info("No text data available for topic modeling")
            else:
                # Choose column for topic modeling
                topic_options = [f"{title} ({col})" for col, title in text_columns]
                selected_option = st.radio("Select feedback type for topic analysis:", topic_options)
                
                # Extract the column name from the selected option
                topic_col = next(col for col, title in text_columns if f"{title} ({col})" == selected_option)
                
                # Number of topics
                n_topics = st.slider("Number of topics to extract:", 2, 10, 3, key="n_topics_slider")
                
                # Perform topic modeling if we have enough data
                if filtered_df[topic_col].dropna().shape[0] > max(10, n_topics*2):  # Ensure enough data
                    try:
                        topics_words, lda_model, vectorizer, dtm = perform_topic_modeling(filtered_df, topic_col, n_topics)
                        
                        if topics_words is None:
                            st.warning("Could not extract meaningful topics from the text data. Try reducing the number of topics or ensure there's sufficient text content.")
                        else:
                            # Display topics
                            st.subheader(f"Top {n_topics} Topics in {topic_col}")
                            
                            for i, topic_words in enumerate(topics_words):
                                with st.expander(f"Topic {i+1}"):
                                    st.write(f"Keywords: {', '.join(topic_words)}")
                                    
                                    # Get sample documents for this topic
                                    topic_distribution = lda_model.transform(dtm)
                                    doc_topic = topic_distribution.argmax(axis=1)
                                    docs_in_topic = [j for j, topic in enumerate(doc_topic) if topic == i]
                                    
                                    # Limit to 5 samples or less if we don't have many
                                    sample_limit = min(5, len(docs_in_topic))
                                    if sample_limit > 0:
                                        docs_in_topic = docs_in_topic[:sample_limit]
                                        st.write("Sample feedback:")
                                        texts_list = filtered_df[topic_col].dropna().reset_index(drop=True)
                                        if len(texts_list) > 0:  # Ensure we have texts to display
                                            for j, idx in enumerate(docs_in_topic):
                                                if idx < len(texts_list):  # Check if index is valid
                                                    st.write(f"{j+1}. {texts_list.iloc[idx]}")
                            
                            # Topic distribution chart
                            topic_dist = lda_model.transform(dtm)
                            topic_proportions = topic_dist.mean(axis=0)
                            topic_names = [f"Topic {i+1}" for i in range(n_topics)]
                            
                            topic_df = pd.DataFrame({
                                'Topic': topic_names,
                                'Proportion': topic_proportions,
                                'Keywords': [', '.join(words[:5]) for words in topics_words]
                            })
                            
                            fig = px.bar(topic_df, x='Topic', y='Proportion', 
                                       title="Topic Distribution",
                                       hover_data=['Keywords'],
                                       color='Proportion',
                                       color_continuous_scale=px.colors.sequential.Viridis)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Topic evolution over time if we have enough data
                            if 'datetime_originally_submitted' in filtered_df.columns and filtered_df['datetime_originally_submitted'].notna().sum() > 20:
                                st.subheader("Topic Evolution Over Time")
                                
                                # Group by month - ensure we copy the dataframe first
                                temp_df = filtered_df.copy()
                                temp_df['topic_month'] = temp_df['datetime_originally_submitted'].dt.to_period('M')
                                months = sorted(temp_df['topic_month'].dropna().unique())
                                
                                # Only proceed if we have enough months
                                if len(months) > 1:
                                    topic_evolution = []
                                    
                                    for month in months:
                                        month_df = temp_df[temp_df['topic_month'] == month]
                                        month_texts = month_df[topic_col].dropna()
                                        
                                        if len(month_texts) > 5:  # Ensure enough texts per month
                                            try:
                                                month_dtm = vectorizer.transform(month_texts)
                                                month_topic_dist = lda_model.transform(month_dtm)
                                                topic_props = month_topic_dist.mean(axis=0)
                                                
                                                topic_evolution.append({
                                                    'Month': month.to_timestamp(),
                                                    **{f'Topic {i+1}': prop for i, prop in enumerate(topic_props)}
                                                })
                                            except Exception as e:
                                                st.error(f"Error processing month {month}: {e}")
                                    
                                    if topic_evolution:
                                        evolution_df = pd.DataFrame(topic_evolution)
                                        fig = px.line(evolution_df, x='Month', y=[f'Topic {i+1}' for i in range(n_topics)],
                                                   title="Topic Proportion Over Time",
                                                   labels={'value': 'Topic Proportion', 'variable': 'Topic'})
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.info("Not enough data per month for topic evolution analysis")
                                else:
                                    st.info("Not enough distinct months for time analysis")
                            
                            # NEW: Topic Distribution by Client Name (top clients only)
                            if 'client_name' in filtered_df.columns and filtered_df['client_name'].notna().any():
                                st.subheader("Topic Distribution by Client")
                                
                                # Get top clients by count to avoid too many groups
                                top_clients_list = filtered_df['client_name'].value_counts().nlargest(6).index.tolist()
                                
                                # Get topic distribution for each of the top clients
                                if top_clients_list:  # Only proceed if we have clients
                                    client_topic_data = []
                                    
                                    for client in top_clients_list:
                                        client_df = filtered_df[filtered_df['client_name'] == client]
                                        client_texts = client_df[topic_col].dropna()
                                        
                                        if len(client_texts) > 5:  # Ensure enough texts
                                            try:
                                                client_dtm = vectorizer.transform(client_texts)
                                                client_topic_dist = lda_model.transform(client_dtm)
                                                client_topic_props = client_topic_dist.mean(axis=0)
                                                
                                                for i, prop in enumerate(client_topic_props):
                                                    client_topic_data.append({
                                                        'Client': client,
                                                        'Topic': f'Topic {i+1}',
                                                        'Proportion': prop
                                                    })
                                            except Exception as e:
                                                st.error(f"Error processing client {client}: {e}")
                                    
                                    if client_topic_data:
                                        client_topic_df = pd.DataFrame(client_topic_data)
                                        fig = px.bar(client_topic_df, x='Topic', y='Proportion', 
                                                   color='Client', barmode='group',
                                                   title="Topic Distribution by Client")
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.info("Not enough data for topic distribution by client")
                            
                            # Topic Distribution by Client Type if available
                            if 'client_type' in filtered_df.columns and filtered_df['client_type'].notna().any():
                                st.subheader("Topic Distribution by Client Type")
                                
                                # Get topic distribution for each client type
                                client_types = filtered_df['client_type'].dropna().unique()
                                
                                if len(client_types) > 1:  # Only proceed if we have multiple client types
                                    client_topic_data = []
                                    
                                    for client_type in client_types:
                                        client_df = filtered_df[filtered_df['client_type'] == client_type]
                                        client_texts = client_df[topic_col].dropna()
                                        
                                        if len(client_texts) > 5:  # Ensure enough texts
                                            try:
                                                client_dtm = vectorizer.transform(client_texts)
                                                client_topic_dist = lda_model.transform(client_dtm)
                                                client_topic_props = client_topic_dist.mean(axis=0)
                                                
                                                for i, prop in enumerate(client_topic_props):
                                                    client_topic_data.append({
                                                        'Client Type': client_type,
                                                        'Topic': f'Topic {i+1}',
                                                        'Proportion': prop
                                                    })
                                            except Exception as e:
                                                st.error(f"Error processing client type {client_type}: {e}")
                                    
                                    if client_topic_data:
                                        client_topic_df = pd.DataFrame(client_topic_data)
                                        fig = px.bar(client_topic_df, x='Topic', y='Proportion', 
                                                   color='Client Type', barmode='group',
                                                   title="Topic Distribution by Client Type")
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.info("Not enough data for topic distribution by client type")
                            
                            # Topic Distribution by status if available
                            status_column = 'course_status' if st.session_state.dashboard_type == "course" else 'axe_status'
                            if status_column in filtered_df.columns and filtered_df[status_column].notna().any():
                                status_label = "Course Status" if st.session_state.dashboard_type == "course" else "AXE Status"
                                st.subheader(f"Topic Distribution by {status_label}")
                                
                                # Get topic distribution for each status
                                statuses = filtered_df[status_column].dropna().unique()
                                
                                if len(statuses) > 1:  # Only proceed if we have multiple statuses
                                    status_topic_data = []
                                    
                                    for status in statuses:
                                        status_df = filtered_df[filtered_df[status_column] == status]
                                        status_texts = status_df[topic_col].dropna()
                                        
                                        if len(status_texts) > 5:  # Ensure enough texts
                                            try:
                                                status_dtm = vectorizer.transform(status_texts)
                                                status_topic_dist = lda_model.transform(status_dtm)
                                                status_topic_props = status_topic_dist.mean(axis=0)
                                                
                                                for i, prop in enumerate(status_topic_props):
                                                    status_topic_data.append({
                                                        status_label: status,
                                                        'Topic': f'Topic {i+1}',
                                                        'Proportion': prop
                                                    })
                                            except Exception as e:
                                                st.error(f"Error processing status {status}: {e}")
                                    
                                    if status_topic_data:
                                        status_topic_df = pd.DataFrame(status_topic_data)
                                        fig = px.bar(status_topic_df, x='Topic', y='Proportion', 
                                                   color=status_label, barmode='group',
                                                   title=f"Topic Distribution by {status_label}")
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.info(f"Not enough data for topic distribution by {status_label.lower()}")
                    except Exception as e:
                        st.error(f"Error in topic modeling: {e}")
                else:
                    st.info("Not enough text data for topic modeling. Please adjust your filters or select a different text column.")
    
    # User Segmentation tab
    if "User Segmentation" in tab_list:
        with tabs[tab_list.index("User Segmentation")]:
            st.header("User Segmentation Analysis")
            
            # Check if we have enough numerical data for clustering
            clustering_features = ['rating', 'how_many_hours_it_took', 'positive_sentiment', 'improvement_sentiment']
            valid_features = [col for col in clustering_features if col in filtered_df.columns and filtered_df[col].notna().any()]
            
            if len(valid_features) < 2 or filtered_df[valid_features].dropna().shape[0] < 10:
                st.info("Not enough numerical data for clustering. Need at least 2 features with sufficient data.")
            else:
                n_clusters = st.slider("Number of user segments:", 2, 6, 4, key="n_clusters")
                
                # Perform clustering
                try:
                    segmented_df, cluster_centers = segment_users(filtered_df, n_clusters)
                    
                    if segmented_df is None or 'cluster' not in segmented_df.columns:
                        st.error("Clustering failed. Please check your data or try different parameters.")
                    else:
                        # Display cluster information
                        st.subheader(f"User Segments (K-Means Clustering, {n_clusters} clusters)")
                        
                        # Count users in each cluster
                        cluster_counts = segmented_df['cluster'].value_counts().sort_index()
                        cluster_percentages = (cluster_counts / cluster_counts.sum() * 100).round(1)
                        
                        cluster_df = pd.DataFrame({
                            'Segment': [f"Segment {i+1}" for i in range(n_clusters)],
                            'Count': cluster_counts.values,
                            'Percentage': cluster_percentages.values
                        })
                        
                        # Cluster distribution chart
                        fig = px.bar(cluster_df, x='Segment', y='Count', 
                                   title="Distribution of Users Across Segments",
                                   text='Percentage',
                                   labels={'Count': 'Number of Users', 'Segment': 'User Segment'},
                                   color='Count')
                        fig.update_traces(texttemplate='%{text}%', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Cluster characteristics
                        st.subheader("Segment Characteristics")
                        
                        # Calculate mean values for key metrics by cluster
                        # Only include columns that have data
                        profile_columns = [col for col in ['rating', 'how_many_hours_it_took', 'positive_sentiment', 
                                                       'improvement_sentiment', 'course_duration_minutes'] 
                                         if col in segmented_df.columns and segmented_df[col].notna().any()]
                        
                        if not profile_columns:
                            st.error("No valid metrics for cluster profiling")
                        else:
                            cluster_profiles = segmented_df.groupby('cluster')[profile_columns].mean().reset_index()
                            
                            # Rename cluster to segment
                            cluster_profiles['Segment'] = [f"Segment {i+1}" for i in range(n_clusters)]
                            
                            # Display a table of cluster profiles
                            display_profiles = cluster_profiles.copy()
                            display_profiles.set_index('Segment', inplace=True)
                            del display_profiles['cluster']  # Remove cluster column from display
                            
                            # Format the table
                            st.dataframe(display_profiles.style.format("{:.2f}"))
                            
                            # Create radar chart for each cluster
                            # Normalize values for radar chart - only use metrics that exist in our data
                            metrics = [col for col in profile_columns if col in cluster_profiles.columns]
                            
                            if len(metrics) >= 3:  # Need at least 3 metrics for a meaningful radar chart
                                for metric in metrics:
                                    max_val = cluster_profiles[metric].max()
                                    min_val = cluster_profiles[metric].min()
                                    if max_val > min_val:
                                        cluster_profiles[f"{metric}_norm"] = (cluster_profiles[metric] - min_val) / (max_val - min_val)
                                    else:
                                        cluster_profiles[f"{metric}_norm"] = 0.5
                                
                                # Display radar chart
                                fig = go.Figure()
                                
                                for i, row in cluster_profiles.iterrows():
                                    fig.add_trace(go.Scatterpolar(
                                        r=[row[f"{metric}_norm"] for metric in metrics],
                                        theta=[metric.replace('_', ' ').title() for metric in metrics],
                                        fill='toself',
                                        name=f"Segment {row['cluster']+1}"
                                    ))
                                
                                fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, 1]
                                        )),
                                    showlegend=True,
                                    title="Segment Profiles (Normalized Values)"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Not enough metrics for radar chart visualization. Need at least 3 metrics.")
                        
                        # NEW: Client name distribution within each segment if available
                        if 'client_name' in segmented_df.columns and segmented_df['client_name'].notna().any():
                            st.subheader("Top Clients by Segment")
                            
                            # For each segment, find the top 3 clients
                            for i in range(n_clusters):
                                segment_data = segmented_df[segmented_df['cluster'] == i]
                                top_clients = segment_data['client_name'].value_counts().nlargest(3)
                                
                                st.write(f"**Segment {i+1} Top Clients:**")
                                if not top_clients.empty:
                                    for client, count in zip(top_clients.index, top_clients.values):
                                        st.write(f"- {client}: {count} feedbacks ({(count/len(segment_data)*100):.1f}%)")
                                else:
                                    st.write("- No client data available")
                        
                        # Client type distribution within each segment if available
                        if 'client_type' in segmented_df.columns and segmented_df['client_type'].notna().any():
                            st.subheader("Client Type Distribution by Segment")
                            
                            # Calculate the count and percentage of each client type in each segment
                            client_type_segment = pd.crosstab(
                                segmented_df['cluster'], 
                                segmented_df['client_type'], 
                                normalize='index'
                            ).reset_index() * 100
                            
                            # Rename cluster to segment
                            client_type_segment['Segment'] = [f"Segment {i+1}" for i in range(n_clusters)]
                            
                            # Reshape for plotting
                            client_type_segment_melted = pd.melt(
                                client_type_segment, 
                                id_vars=['cluster', 'Segment'], 
                                var_name='Client Type', 
                                value_name='Percentage'
                            )
                            
                            # Create stacked bar chart
                            fig = px.bar(
                                client_type_segment_melted, 
                                x='Segment', 
                                y='Percentage', 
                                color='Client Type',
                                title="Client Type Distribution within Each Segment (%)",
                                labels={'Percentage': '% of Segment'}
                            )
                            
                            fig.update_layout(barmode='stack')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Course/AXE status distribution within each segment if available
                        status_column = 'course_status' if st.session_state.dashboard_type == "course" else 'axe_status'
                        if status_column in segmented_df.columns and segmented_df[status_column].notna().any():
                            status_label = "Course Status" if st.session_state.dashboard_type == "course" else "AXE Status"
                            st.subheader(f"{status_label} Distribution by Segment")
                            
                            # Calculate the count and percentage of each status in each segment
                            status_segment = pd.crosstab(
                                segmented_df['cluster'], 
                                segmented_df[status_column], 
                                normalize='index'
                            ).reset_index() * 100
                            
                            # Rename cluster to segment
                            status_segment['Segment'] = [f"Segment {i+1}" for i in range(n_clusters)]
                            
                            # Reshape for plotting
                            status_segment_melted = pd.melt(
                                status_segment, 
                                id_vars=['cluster', 'Segment'], 
                                var_name=status_label, 
                                value_name='Percentage'
                            )
                            
                            # Create stacked bar chart
                            fig = px.bar(
                                status_segment_melted, 
                                x='Segment', 
                                y='Percentage', 
                                color=status_label,
                                title=f"{status_label} Distribution within Each Segment (%)",
                                labels={'Percentage': '% of Segment'}
                            )
                            
                            fig.update_layout(barmode='stack')
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error during clustering: {e}")

if __name__ == "__main__":
    main()