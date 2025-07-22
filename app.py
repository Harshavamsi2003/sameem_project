# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Knee OA Treatment Comparison Dashboard",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .header-text {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ========== DATA LOADING ==========
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("RESEARCH_DATA.xlsx")
        df.columns = df.columns.str.strip().str.upper()  # Standardize column names
        
        # Ensure required columns exist
        required_cols = ['GROUP', 'AROM_1_F', 'AROM_2_F', 'PROM_1_F', 'PROM_2_F',
                        'VAS_1', 'VAS_2', 'W_P_1', 'W_P_2', 'W_S_1', 'W_S_2', 
                        'W_D_1', 'W_D_2', 'AGE', 'GENDER', 'DURATION']
        
        # Check for missing columns but don't stop execution
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.warning(f"Note: Some expected columns not found - {', '.join(missing)}")
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data()
if df is None:
    st.stop()

# ========== DATA PREP ==========
df['GROUP'] = df['GROUP'].map({1: 'Maitland + Conventional', 2: 'Conventional Alone'})

# Define metrics for analysis
metrics = {
    'AROM': {'pre': 'AROM_1_F', 'post': 'AROM_2_F', 'title': 'Active ROM (Flexion)', 'unit': '°', 'direction': 'increase'},
    'PROM': {'pre': 'PROM_1_F', 'post': 'PROM_2_F', 'title': 'Passive ROM (Flexion)', 'unit': '°', 'direction': 'increase'},
    'VAS': {'pre': 'VAS_1', 'post': 'VAS_2', 'title': 'Pain (VAS)', 'unit': '0-10', 'direction': 'decrease'},
    'WOMAC_P': {'pre': 'W_P_1', 'post': 'W_P_2', 'title': 'WOMAC Pain', 'unit': 'Score', 'direction': 'decrease'},
    'WOMAC_S': {'pre': 'W_S_1', 'post': 'W_S_2', 'title': 'WOMAC Stiffness', 'unit': 'Score', 'direction': 'decrease'},
    'WOMAC_D': {'pre': 'W_D_1', 'post': 'W_D_2', 'title': 'WOMAC Disability', 'unit': 'Score', 'direction': 'decrease'}
}

# Calculate absolute and percentage change
for metric in metrics:
    pre_col = metrics[metric]['pre']
    post_col = metrics[metric]['post']
    
    df[f'{metric}_CHANGE'] = df[post_col] - df[pre_col]
    df[f'{metric}_PCT_CHANGE'] = ((df[post_col] - df[pre_col]) / df[pre_col].replace(0, 0.001)) * 100

# ========== SIDEBAR FILTERS ==========
st.sidebar.header("🔍 Filters")

# Age filter
if 'AGE' in df.columns:
    age_min, age_max = int(df['AGE'].min()), int(df['AGE'].max())
    selected_age = st.sidebar.slider("Select Age Range", age_min, age_max, (age_min, age_max))
else:
    selected_age = (0, 100)

# Gender filter
if 'GENDER' in df.columns:
    gender_options = df['GENDER'].unique()
    selected_genders = st.sidebar.multiselect("Select Gender(s)", gender_options, default=gender_options)
else:
    selected_genders = []

# Duration filter
if 'DURATION' in df.columns:
    duration_min, duration_max = int(df['DURATION'].min()), int(df['DURATION'].max())
    selected_duration = st.sidebar.slider("Treatment Duration (weeks)", duration_min, duration_max, (duration_min, duration_max))
else:
    selected_duration = (0, 20)

# Apply filters
filter_conditions = []
if 'AGE' in df.columns:
    filter_conditions.append((df['AGE'] >= selected_age[0]) & (df['AGE'] <= selected_age[1]))
if 'GENDER' in df.columns and selected_genders:
    filter_conditions.append(df['GENDER'].isin(selected_genders))
if 'DURATION' in df.columns:
    filter_conditions.append((df['DURATION'] >= selected_duration[0]) & (df['DURATION'] <= selected_duration[1]))

filtered_df = df.copy()
if filter_conditions:
    filtered_df = df[np.all(filter_conditions, axis=0)]

# ========== MAIN DASHBOARD ==========
st.title("🦵 Knee Osteoarthritis Treatment Outcomes")
st.markdown("""
Comparing **Maitland Mobilization + Conventional Therapy** vs **Conventional Therapy Alone**  
*Data analyzed from pre-test to post-test measurements*
""")

# ========== GROUP COMPARISON ==========
st.header("📊 Treatment Group Comparison")

# Create tabs for each metric
tabs = st.tabs([metrics[metric]['title'] for metric in metrics])

for i, metric in enumerate(metrics):
    with tabs[i]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Pre-Post Comparison by Group
            fig = px.box(
                filtered_df,
                x='GROUP',
                y=[metrics[metric]['pre'], metrics[metric]['post']],
                color='GROUP',
                labels={'value': f"{metrics[metric]['title']} ({metrics[metric]['unit']})", 'variable': 'Test'},
                title=f"Pre vs Post {metrics[metric]['title']} by Group",
                color_discrete_map={'Maitland + Conventional': '#636EFA', 'Conventional Alone': '#EF553B'}
            )
            fig.update_layout(
                boxmode='group',
                legend_title_text='Treatment Group',
                yaxis_title=f"{metrics[metric]['title']} ({metrics[metric]['unit']})"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Change Distribution by Group
            fig = px.box(
                filtered_df,
                x='GROUP',
                y=f'{metric}_CHANGE',
                color='GROUP',
                labels={'value': f"Change in {metrics[metric]['title']} ({metrics[metric]['unit']})"},
                title=f"Improvement in {metrics[metric]['title']}",
                color_discrete_map={'Maitland + Conventional': '#636EFA', 'Conventional Alone': '#EF553B'}
            )
            fig.update_layout(
                yaxis_title=f"Change in {metrics[metric]['title']} ({metrics[metric]['unit']})"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Group statistics
        group_stats = filtered_df.groupby('GROUP').agg({
            metrics[metric]['pre']: ['mean', 'std'],
            metrics[metric]['post']: ['mean', 'std'],
            f'{metric}_CHANGE': ['mean', 'std', 'count']
        }).reset_index()
        
        # Format the statistics table
        group_stats.columns = ['Group', 'Pre Mean', 'Pre Std', 'Post Mean', 'Post Std', 
                             'Change Mean', 'Change Std', 'Patient Count']
        group_stats['Improvement'] = group_stats['Change Mean'].apply(lambda x: f"{x:.2f} {metrics[metric]['unit']}")
        
        st.markdown("### Group Statistics")
        st.dataframe(group_stats[['Group', 'Pre Mean', 'Post Mean', 'Improvement', 'Patient Count']].style.format({
            'Pre Mean': '{:.2f}',
            'Post Mean': '{:.2f}'
        }), use_container_width=True)

# ========== INDIVIDUAL METRIC TRENDS ==========
st.header("📈 Detailed Metric Trends")

# Select metric for detailed view
selected_metric = st.selectbox(
    "Select Metric for Detailed Analysis",
    list(metrics.keys()),
    format_func=lambda x: metrics[x]['title'],
    key='detailed_metric'
)

# Create detailed view
col1, col2 = st.columns(2)

with col1:
    # Scatter plot of pre vs post with trend lines
    fig = px.scatter(
        filtered_df,
        x=metrics[selected_metric]['pre'],
        y=metrics[selected_metric]['post'],
        color='GROUP',
        trendline="ols",
        title=f"Pre vs Post {metrics[selected_metric]['title']} with Trendlines",
        labels={
            metrics[selected_metric]['pre']: f"Pre-Test {metrics[selected_metric]['title']}",
            metrics[selected_metric]['post']: f"Post-Test {metrics[selected_metric]['title']}"
        },
        color_discrete_map={'Maitland + Conventional': '#636EFA', 'Conventional Alone': '#EF553B'}
    )
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=filtered_df[metrics[selected_metric]['pre']].min(),
        y0=filtered_df[metrics[selected_metric]['pre']].min(),
        x1=filtered_df[metrics[selected_metric]['pre']].max(),
        y1=filtered_df[metrics[selected_metric]['pre']].max()
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Histogram of changes
    fig = px.histogram(
        filtered_df,
        x=f'{selected_metric}_CHANGE',
        color='GROUP',
        nbins=20,
        barmode='overlay',
        title=f"Distribution of Changes in {metrics[selected_metric]['title']}",
        labels={f'{selected_metric}_CHANGE': f"Change in {metrics[selected_metric]['title']} ({metrics[selected_metric]['unit']})"},
        color_discrete_map={'Maitland + Conventional': '#636EFA', 'Conventional Alone': '#EF553B'}
    )
    fig.update_layout(
        bargap=0.1,
        xaxis_title=f"Change in {metrics[selected_metric]['title']} ({metrics[selected_metric]['unit']})"
    )
    st.plotly_chart(fig, use_container_width=True)

# ========== PATIENT DEMOGRAPHICS ==========
st.header("👥 Patient Demographics")

if 'AGE' in df.columns and 'GENDER' in df.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution by group
        fig = px.histogram(
            filtered_df,
            x='AGE',
            color='GROUP',
            nbins=10,
            title="Age Distribution by Treatment Group",
            color_discrete_map={'Maitland + Conventional': '#636EFA', 'Conventional Alone': '#EF553B'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender distribution by group
        gender_counts = filtered_df.groupby(['GROUP', 'GENDER']).size().reset_index(name='Count')
        fig = px.bar(
            gender_counts,
            x='GROUP',
            y='Count',
            color='GENDER',
            title="Gender Distribution by Treatment Group",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

# ========== DATA EXPORT ==========
st.download_button(
    "📥 Download Filtered Data as CSV",
    filtered_df.to_csv(index=False).encode('utf-8'),
    "knee_oa_treatment_data.csv",
    "text/csv",
    key='download-csv'
)
