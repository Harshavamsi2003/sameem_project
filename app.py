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
    .highlight {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .group1-color {
        color: #636EFA;
        font-weight: bold;
    }
    .group2-color {
        color: #EF553B;
        font-weight: bold;
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
st.sidebar.header("🔍 Data Filters")

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
st.title("🦵 Knee Osteoarthritis Treatment Outcomes Dashboard")
st.markdown("""
**Comparing treatment effectiveness:** <span class="group1-color">Maitland Mobilization + Conventional Therapy</span> vs 
<span class="group2-color">Conventional Therapy Alone</span>
""", unsafe_allow_html=True)

# Key Insights Section
st.markdown("### 🔍 Key Insights at a Glance")

# Calculate key metrics for insights
group1_df = filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional']
group2_df = filtered_df[filtered_df['GROUP'] == 'Conventional Alone']

# Insight 1: Overall improvement comparison
avg_arom_improve1 = group1_df['AROM_CHANGE'].mean()
avg_arom_improve2 = group2_df['AROM_CHANGE'].mean()

# Insight 2: Pain reduction comparison
avg_vas_improve1 = group1_df['VAS_CHANGE'].mean()
avg_vas_improve2 = group2_df['VAS_CHANGE'].mean()

# Insight 3: WOMAC disability improvement
avg_womacd_improve1 = group1_df['WOMAC_D_CHANGE'].mean()
avg_womacd_improve2 = group2_df['WOMAC_D_CHANGE'].mean()

# Display insights in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-box">
        <div class="header-text">Active ROM Improvement</div>
        <div><span class="group1-color">Maitland+:</span> {:.1f}°</div>
        <div><span class="group2-color">Conventional:</span> {:.1f}°</div>
        <div>Difference: <b>{:.1f}°</b></div>
    </div>
    """.format(avg_arom_improve1, avg_arom_improve2, avg_arom_improve1 - avg_arom_improve2), 
    unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-box">
        <div class="header-text">Pain Reduction (VAS)</div>
        <div><span class="group1-color">Maitland+:</span> {:.1f} points</div>
        <div><span class="group2-color">Conventional:</span> {:.1f} points</div>
        <div>Difference: <b>{:.1f} points</b></div>
    </div>
    """.format(avg_vas_improve1, avg_vas_improve2, avg_vas_improve1 - avg_vas_improve2), 
    unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-box">
        <div class="header-text">WOMAC Disability</div>
        <div><span class="group1-color">Maitland+:</span> {:.1f} points</div>
        <div><span class="group2-color">Conventional:</span> {:.1f} points</div>
        <div>Difference: <b>{:.1f} points</b></div>
    </div>
    """.format(avg_womacd_improve1, avg_womacd_improve2, avg_womacd_improve1 - avg_womacd_improve2), 
    unsafe_allow_html=True)

# ========== COMPREHENSIVE PRE-POST COMPARISON ==========
st.header("📊 Comprehensive Pre-Post Treatment Comparison")

# Create tabs for each metric
tabs = st.tabs([metrics[metric]['title'] for metric in metrics])

for i, metric in enumerate(metrics):
    with tabs[i]:
        # First row: Pre-Post comparison and change distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Pre-Post Paired Plot
            fig = go.Figure()
            
            for idx, row in filtered_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=['Pre-Test', 'Post-Test'],
                    y=[row[metrics[metric]['pre']], row[metrics[metric]['post']],
                    mode='lines+markers',
                    line=dict(color='#636EFA' if row['GROUP'] == 'Maitland + Conventional' else '#EF553B', width=1),
                    marker=dict(size=8),
                    showlegend=False,
                    opacity=0.4
                ))
            
            # Add group means
            group_means = filtered_df.groupby('GROUP')[[metrics[metric]['pre'], metrics[metric]['post']].mean().reset_index()
            
            for _, group in group_means.iterrows():
                fig.add_trace(go.Scatter(
                    x=['Pre-Test', 'Post-Test'],
                    y=[group[metrics[metric]['pre']], group[metrics[metric]['post']],
                    mode='lines+markers',
                    line=dict(color='#636EFA' if group['GROUP'] == 'Maitland + Conventional' else '#EF553B', width=3),
                    marker=dict(size=12),
                    name=group['GROUP']
                ))
            
            fig.update_layout(
                title=f"Individual Patient Progress: {metrics[metric]['title']}",
                yaxis_title=f"{metrics[metric]['title']} ({metrics[metric]['unit']})",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Improvement Comparison
            fig = go.Figure()
            
            for group in filtered_df['GROUP'].unique():
                group_data = filtered_df[filtered_df['GROUP'] == group]
                fig.add_trace(go.Box(
                    y=group_data[f'{metric}_CHANGE'],
                    name=group,
                    marker_color='#636EFA' if group == 'Maitland + Conventional' else '#EF553B',
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ))
            
            fig.update_layout(
                title=f"Improvement Comparison: {metrics[metric]['title']}",
                yaxis_title=f"Change in {metrics[metric]['title']} ({metrics[metric]['unit']})",
                showlegend=True
            )
            
            # Add reference line for zero change
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=1.5,
                y0=0,
                y1=0,
                line=dict(color="black", width=2, dash="dash")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Second row: Group statistics and distribution
        col3, col4 = st.columns(2)
        
        with col3:
            # Group statistics table
            group_stats = filtered_df.groupby('GROUP').agg({
                metrics[metric]['pre']: ['mean', 'std', 'median'],
                metrics[metric]['post']: ['mean', 'std', 'median'],
                f'{metric}_CHANGE': ['mean', 'std', 'median', 'count']
            }).reset_index()
            
            # Format the statistics table
            group_stats.columns = ['Group', 'Pre Mean', 'Pre Std', 'Pre Median', 
                                 'Post Mean', 'Post Std', 'Post Median',
                                 'Change Mean', 'Change Std', 'Change Median', 'Patient Count']
            
            st.markdown(f"### {metrics[metric]['title']} Statistics")
            st.dataframe(group_stats.style.format({
                'Pre Mean': '{:.1f}',
                'Pre Std': '{:.1f}',
                'Pre Median': '{:.1f}',
                'Post Mean': '{:.1f}',
                'Post Std': '{:.1f}',
                'Post Median': '{:.1f}',
                'Change Mean': '{:.1f}',
                'Change Std': '{:.1f}',
                'Change Median': '{:.1f}'
            }), use_container_width=True)
        
        with col4:
            # Histogram of pre and post values
            fig = px.histogram(
                filtered_df,
                x=[metrics[metric]['pre'], metrics[metric]['post']],
                color='GROUP',
                barmode='group',
                nbins=15,
                title=f"Distribution of {metrics[metric]['title']} Values",
                labels={'value': f"{metrics[metric]['title']} ({metrics[metric]['unit']})"},
                color_discrete_map={'Maitland + Conventional': '#636EFA', 'Conventional Alone': '#EF553B'}
            )
            fig.update_layout(
                xaxis_title=f"{metrics[metric]['title']} ({metrics[metric]['unit']})",
                legend_title="Treatment Group"
            )
            st.plotly_chart(fig, use_container_width=True)

# ========== PATIENT DEMOGRAPHICS ==========
st.header("👥 Patient Demographics Overview")

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
            color_discrete_map={'Maitland + Conventional': '#636EFA', 'Conventional Alone': '#EF553B'},
            marginal="box"
        )
        fig.update_layout(
            xaxis_title="Age (years)",
            legend_title="Treatment Group"
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
            barmode='group',
            text='Count'
        )
        fig.update_layout(
            xaxis_title="Treatment Group",
            yaxis_title="Number of Patients",
            legend_title="Gender"
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

# ========== TREATMENT DURATION ANALYSIS ==========
if 'DURATION' in df.columns:
    st.header("⏳ Treatment Duration Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Duration distribution
        fig = px.histogram(
            filtered_df,
            x='DURATION',
            color='GROUP',
            nbins=10,
            title="Treatment Duration Distribution",
            color_discrete_map={'Maitland + Conventional': '#636EFA', 'Conventional Alone': '#EF553B'}
        )
        fig.update_layout(
            xaxis_title="Duration (weeks)",
            legend_title="Treatment Group"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Duration vs Improvement
        fig = px.scatter(
            filtered_df,
            x='DURATION',
            y='AROM_CHANGE',
            color='GROUP',
            trendline="ols",
            title="Treatment Duration vs Active ROM Improvement",
            labels={'DURATION': 'Duration (weeks)', 'AROM_CHANGE': 'Active ROM Improvement (°)'},
            color_discrete_map={'Maitland + Conventional': '#636EFA', 'Conventional Alone': '#EF553B'}
        )
        fig.update_layout(
            legend_title="Treatment Group"
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

# ========== APP EXPLANATION ==========
st.sidebar.header("ℹ️ About This Dashboard")
st.sidebar.markdown("""
This dashboard compares the effectiveness of two treatments for knee osteoarthritis:

1. <span class="group1-color">**Maitland + Conventional Therapy**</span>  
2. <span class="group2-color">**Conventional Therapy Alone**</span>

**Key Features:**
- Compare pre-test vs post-test results
- Visualize improvements across multiple metrics
- Filter data by patient demographics
- Export filtered data for further analysis

**Metrics Analyzed:**
- Active/Passive Range of Motion (AROM/PROM)
- Pain (VAS)
- WOMAC scores (Pain, Stiffness, Disability)
""", unsafe_allow_html=True)
