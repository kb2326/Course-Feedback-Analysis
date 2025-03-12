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

# Download necessary NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    nltk.download('punkt')

# Load and preprocess data
@st.cache_data
def process_data(df):
    """Process the uploaded dataframe"""
    # Data Cleaning
    df.drop_duplicates(inplace=True)
    
    # Handling missing values
    if 'course_id' in df.columns and 'rating' in df.columns:
        df.dropna(subset=['course_id', 'rating'], inplace=True)
    
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
    st.set_page_config(page_title="Course Feedback Analysis Dashboard", 
                       layout="wide", 
                       initial_sidebar_state="expanded")
    
    st.title("Course Feedback Analysis Dashboard")
    
    # Run NLTK downloads
    try:
        download_nltk_data()
    except Exception as e:
        st.warning(f"Could not download NLTK data: {e}. Some analysis features may be limited.")
    
    # File upload section
    st.subheader("Upload Your Course Feedback Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is None:
        st.info("Please upload a CSV file to get started.")
        
        # Show expected CSV format
        st.subheader("Expected CSV Format")
        st.write("""
        Your CSV should include some of these columns:
        - course_id: Unique identifier for the course
        - course_name: Name of the course
        - course_category: Category of the course
        - client_sector: Client industry sector
        - rating: Numeric rating (e.g., 1-5)
        - country: Country of the participant
        - course_duration_minutes: Course duration in minutes
        - how_many_hours_it_took: Hours spent by participant
        - what_did_you_like: Text feedback on positives
        - what_could_be_improved: Text feedback on improvements
        - datetime_originally_submitted: When feedback was submitted
        
        Not all columns are required, but more columns enable more analyses.
        """)
        
        # Show sample data structure
        sample_df = pd.DataFrame({
            'course_id': ['C001', 'C002', 'C001'],
            'course_name': ['Python Basics', 'Data Analysis', 'Python Basics'],
            'course_category': ['Programming', 'Data Science', 'Programming'],
            'client_sector': ['Technology', 'Finance', 'Healthcare'],
            'rating': [4.5, 3.8, 5.0],
            'country': ['USA', 'UK', 'Canada'],
            'course_duration_minutes': [120, 180, 120],
            'how_many_hours_it_took': [3, 4, 2.5],
            'what_did_you_like': ['Great examples', 'Comprehensive content', 'Clear explanations'],
            'what_could_be_improved': ['More exercises', 'Faster pace', 'Nothing'],
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
    
    # Course category filter - if column exists
    if 'course_category' in df.columns and df['course_category'].notna().any():
        all_categories = ["All Categories"] + sorted(df['course_category'].dropna().unique().tolist())
        selected_category = st.sidebar.selectbox("Course Category", all_categories)
        
        if selected_category != "All Categories":
            filtered_df = filtered_df[filtered_df['course_category'] == selected_category]
    
    # Client sector filter - if column exists
    if 'client_sector' in df.columns and df['client_sector'].notna().any():
        all_sectors = ["All Sectors"] + sorted(df['client_sector'].dropna().unique().tolist())
        selected_sector = st.sidebar.selectbox("Client Sector", all_sectors)
        
        if selected_sector != "All Sectors":
            filtered_df = filtered_df[filtered_df['client_sector'] == selected_sector]
    
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
    if 'course_name' in filtered_df.columns and 'rating' in filtered_df.columns:
        tab_list.append("Course Performance")
    
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
        
        if 'course_name' in filtered_df.columns:
            total_courses = filtered_df['course_name'].nunique()
            columns[metric_idx].metric("Total Courses", total_courses)
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
    
    # Course Performance tab
    if "Course Performance" in tab_list:
        with tabs[tab_list.index("Course Performance")]:
            st.header("Course Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Highest Rated Courses")
                # Check if we have enough unique courses
                if filtered_df['course_name'].nunique() > 0:
                    top_courses = filtered_df.groupby('course_name')['rating'].agg(['mean', 'count']).sort_values('mean', ascending=False)
                    n_top = min(10, len(top_courses))
                    if n_top > 0:
                        top_courses = top_courses.head(n_top).reset_index()
                        fig = px.bar(top_courses, x='mean', y='course_name', 
                                   labels={'mean': 'Average Rating', 'course_name': 'Course'},
                                   title=f"Top {n_top} Highest Rated Courses",
                                   color='mean',
                                   color_continuous_scale=px.colors.sequential.Blues,
                                   hover_data=['count'])
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough course data for analysis")
                else:
                    st.info("Not enough courses to display ratings")
            
            with col2:
                st.subheader("Bottom Lowest Rated Courses")
                min_reviews = st.slider("Minimum number of reviews", 1, 20, 3, key="min_reviews_slider")
                # Check if we have courses with enough reviews
                courses_with_min_reviews = filtered_df.groupby('course_name').filter(lambda x: len(x) >= min_reviews)
                
                if not courses_with_min_reviews.empty and courses_with_min_reviews['course_name'].nunique() > 0:
                    bottom_courses = courses_with_min_reviews.groupby('course_name')['rating'].agg(['mean', 'count']).sort_values('mean')
                    n_bottom = min(10, len(bottom_courses))
                    if n_bottom > 0:
                        bottom_courses = bottom_courses.head(n_bottom).reset_index()
                        fig = px.bar(bottom_courses, x='mean', y='course_name', 
                                   labels={'mean': 'Average Rating', 'course_name': 'Course'},
                                   title=f"Bottom {n_bottom} Lowest Rated Courses (min {min_reviews} reviews)",
                                   color='mean',
                                   color_continuous_scale=px.colors.sequential.Reds_r,
                                   hover_data=['count'])
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough course data with the minimum review threshold")
                else:
                    st.info(f"No courses with at least {min_reviews} reviews found")
            
            # Only show category and sector analysis if those columns exist
            if 'course_category' in filtered_df.columns or 'client_sector' in filtered_df.columns:
                col1, col2 = st.columns(2)
                
                if 'course_category' in filtered_df.columns and filtered_df['course_category'].notna().any():
                    with col1:
                        st.subheader("Performance by Category")
                        category_perf = filtered_df.groupby('course_category')['rating'].agg(['mean', 'count']).sort_values('mean').reset_index()
                        fig = px.bar(category_perf, x='course_category', y='mean', 
                                   labels={'mean': 'Average Rating', 'course_category': 'Category', 'count': 'Number of Reviews'},
                                   title="Average Rating by Course Category",
                                   color='mean',
                                   color_continuous_scale=px.colors.sequential.Viridis,
                                   hover_data=['count'])
                        st.plotly_chart(fig, use_container_width=True)
                
                if 'client_sector' in filtered_df.columns and filtered_df['client_sector'].notna().any():
                    with col2:
                        st.subheader("Performance by Client Sector")
                        sector_perf = filtered_df.groupby('client_sector')['rating'].agg(['mean', 'count']).sort_values('mean').reset_index()
                        fig = px.bar(sector_perf, x='client_sector', y='mean', 
                                   labels={'mean': 'Average Rating', 'client_sector': 'Sector', 'count': 'Number of Reviews'},
                                   title="Average Rating by Client Sector",
                                   color='mean',
                                   color_continuous_scale=px.colors.sequential.Plasma,
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
            
            # Course duration vs rating - only if both columns have data
            if 'course_duration_minutes' in filtered_df.columns and 'rating' in filtered_df.columns and \
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
                                st.write(f"{var1}** and *{var2}* have a {strength} {direction} correlation (r = {corr_val:.2f}, p = {p_val:.3f})")
                        else:
                            st.write("No statistically significant strong correlations found")
                
                except Exception as e:
                    st.error(f"Error calculating correlations: {e}")
                
                # Statistical significance testing
                st.subheader("Statistical Significance Testing")
                
                # Only show categorical variables that have data
                categorical_vars = [col for col in ["course_category", "client_sector", "country"] 
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
                                       color='course_category' if 'course_category' in scatter_df.columns else None)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment by category
                if 'course_category' in filtered_df.columns and filtered_df['course_category'].notna().any():
                    sentiment_by_cat = filtered_df.groupby('course_category')[sentiment_cols].mean().reset_index()
                    
                    fig = px.bar(sentiment_by_cat, x='course_category', y=sentiment_cols,
                               barmode='group',
                               title="Average Sentiment by Course Category",
                               labels={'value': 'Average Sentiment', 'course_category': 'Category', 'variable': 'Sentiment Type'},
                               color_discrete_map={'positive_sentiment': '#2E8B57', 'improvement_sentiment': '#B22222'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Word clouds - only if text columns have data
                text_cols = []
                if 'what_did_you_like' in filtered_df.columns and filtered_df['what_did_you_like'].notna().any():
                    text_cols.append(('what_did_you_like', 'Positive Feedback', 'viridis'))
                if 'what_could_be_improved' in filtered_df.columns and filtered_df['what_could_be_improved'].notna().any():
                    text_cols.append(('what_could_be_improved', 'Improvement Suggestions', 'plasma'))
                    
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
                except Exception as e:
                    st.error(f"Error during clustering: {e}")


                
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
                    except Exception as e:
                        st.error(f"Error in topic modeling: {e}")
                else:
                    st.info("Not enough text data for topic modeling. Please adjust your filters or select a different text column.")
if __name__ == "__main__":
    main()