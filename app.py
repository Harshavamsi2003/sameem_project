# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Knee OA Treatment Outcomes Dashboard",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header-text {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 15px;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #3498db;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
        margin-top: 25px;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .st-bb {
        background-color: white;
    }
    .st-at {
        background-color: #2980b9;
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

# Calculate changes
for metric in metrics:
    pre_col = metrics[metric]['pre']
    post_col = metrics[metric]['post']
    
    df[f'{metric}_CHANGE'] = df[post_col] - df[pre_col]
    df[f'{metric}_PCT_CHANGE'] = ((df[post_col] - df[pre_col]) / df[pre_col].replace(0, 0.001)) * 100
    df[f'{metric}_IMPROVED'] = df[f'{metric}_CHANGE'] > 0 if metrics[metric]['direction'] == 'increase' else df[f'{metric}_CHANGE'] < 0

# ========== SIDEBAR ==========
st.sidebar.header("🔍 Filter Controls")
st.sidebar.markdown("Customize the data view using these filters:")

# Age filter
if 'AGE' in df.columns:
    age_min, age_max = int(df['AGE'].min()), int(df['AGE'].max())
    selected_age = st.sidebar.slider("Patient Age Range", age_min, age_max, (age_min, age_max))
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

# Group filter
selected_groups = st.sidebar.multiselect(
    "Select Treatment Groups",
    df['GROUP'].unique(),
    default=df['GROUP'].unique()
)

# Apply filters
filter_conditions = []
if 'AGE' in df.columns:
    filter_conditions.append((df['AGE'] >= selected_age[0]) & (df['AGE'] <= selected_age[1]))
if 'GENDER' in df.columns and selected_genders:
    filter_conditions.append(df['GENDER'].isin(selected_genders))
if 'DURATION' in df.columns:
    filter_conditions.append((df['DURATION'] >= selected_duration[0]) & (df['DURATION'] <= selected_duration[1]))
if selected_groups:
    filter_conditions.append(df['GROUP'].isin(selected_groups))

filtered_df = df.copy()
if filter_conditions:
    filtered_df = df[np.all(filter_conditions, axis=0)]

# ========== MAIN DASHBOARD ==========
st.title("🦵 Knee Osteoarthritis Treatment Outcomes Dashboard")
st.markdown("""
**Comparative analysis of Maitland Mobilization + Conventional Therapy vs Conventional Therapy Alone**  
*Evaluating pre-test to post-test changes across key clinical measures*
""")

# ===== KEY METRICS =====
st.markdown('<div class="section-title">📊 Key Treatment Outcomes</div>', unsafe_allow_html=True)

metric_cols = st.columns(3)
with metric_cols[0]:
    st.markdown('<div class="metric-card">'
                '<h3>Total Patients</h3>'
                f'<p style="font-size: 2rem; color: #2980b9;">{len(filtered_df)}</p>'
                '</div>', unsafe_allow_html=True)
    
with metric_cols[1]:
    avg_improve_arom = filtered_df['AROM_CHANGE'].mean()
    st.markdown('<div class="metric-card">'
                '<h3>Avg AROM Improvement</h3>'
                f'<p style="font-size: 2rem; color: #2980b9;">{avg_improve_arom:.1f}°</p>'
                '</div>', unsafe_allow_html=True)
    
with metric_cols[2]:
    avg_improve_vas = filtered_df['VAS_CHANGE'].mean()
    st.markdown('<div class="metric-card">'
                '<h3>Avg Pain Reduction</h3>'
                f'<p style="font-size: 2rem; color: #2980b9;">{abs(avg_improve_vas):.1f} points</p>'
                '</div>', unsafe_allow_html=True)

# ===== COMPREHENSIVE COMPARISON =====
st.markdown('<div class="section-title">📈 Treatment Group Comparison</div>', unsafe_allow_html=True)

# Create tabs for each metric
tabs = st.tabs([metrics[metric]['title'] for metric in metrics])

for i, metric in enumerate(metrics):
    with tabs[i]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Pre-Post Parallel Coordinates Plot
            fig = px.parallel_categories(
                filtered_df,
                dimensions=['GROUP', metrics[metric]['pre'], metrics[metric]['post']],
                color='GROUP',
                title=f"Pre-Post {metrics[metric]['title']} Distribution",
                color_discrete_map={'Maitland + Conventional': '#3498db', 'Conventional Alone': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Improvement Summary
            group_stats = filtered_df.groupby('GROUP').agg({
                metrics[metric]['pre']: ['mean', 'std'],
                metrics[metric]['post']: ['mean', 'std'],
                f'{metric}_CHANGE': ['mean', 'std', 'count']
            }).reset_index()
            
            st.markdown(f"**{metrics[metric]['title']} Improvement Summary**")
            st.dataframe(
                group_stats.style.format({
                    metrics[metric]['pre']: '{:.1f}',
                    metrics[metric]['post']: '{:.1f}',
                    f'{metric}_CHANGE': '{:.1f}'
                }).background_gradient(cmap='Blues'),
                use_container_width=True
            )
        
        with col2:
            # Pre-Post Scatter Plot with Reference Line
            fig = px.scatter(
                filtered_df,
                x=metrics[metric]['pre'],
                y=metrics[metric]['post'],
                color='GROUP',
                trendline="ols",
                title=f"Pre vs Post {metrics[metric]['title']} Comparison",
                labels={
                    metrics[metric]['pre']: f"Pre-Test {metrics[metric]['title']} ({metrics[metric]['unit']})",
                    metrics[metric]['post']: f"Post-Test {metrics[metric]['title']} ({metrics[metric]['unit']})"
                },
                color_discrete_map={'Maitland + Conventional': '#3498db', 'Conventional Alone': '#e74c3c'}
            )
            fig.add_shape(
                type="line", line=dict(dash='dash', color='grey'),
                x0=filtered_df[metrics[metric]['pre']].min(),
                y0=filtered_df[metrics[metric]['pre']].min(),
                x1=filtered_df[metrics[metric]['pre']].max(),
                y1=filtered_df[metrics[metric]['pre']].max()
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Improvement Distribution
            fig = px.box(
                filtered_df,
                x='GROUP',
                y=f'{metric}_CHANGE',
                color='GROUP',
                points="all",
                title=f"Improvement Distribution - {metrics[metric]['title']}",
                color_discrete_map={'Maitland + Conventional': '#3498db', 'Conventional Alone': '#e74c3c'}
            )
            fig.update_layout(
                yaxis_title=f"Change in {metrics[metric]['title']} ({metrics[metric]['unit']})"
            )
            st.plotly_chart(fig, use_container_width=True)

# ===== PATIENT OUTCOMES =====
st.markdown('<div class="section-title">👥 Patient-Level Outcomes</div>', unsafe_allow_html=True)

# Select metric for detailed view
selected_metric = st.selectbox(
    "Select Metric for Detailed Analysis",
    list(metrics.keys()),
    format_func=lambda x: metrics[x]['title'],
    key='detailed_metric'
)

col1, col2 = st.columns(2)

with col1:
    # Individual Patient Progress
    fig = px.line(
        filtered_df.melt(
            id_vars=['PATIENT_NAME', 'GROUP'],
            value_vars=[metrics[selected_metric]['pre'], metrics[selected_metric]['post']],
            var_name='Test',
            value_name='Value'
        ),
        x='Test',
        y='Value',
        color='PATIENT_NAME',
        facet_col='GROUP',
        title=f"Individual Patient Progress - {metrics[selected_metric]['title']}",
        labels={'Value': f"{metrics[selected_metric]['title']} ({metrics[selected_metric]['unit']})"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Response Rate by Group
    response_rates = filtered_df.groupby(['GROUP', f'{selected_metric}_IMPROVED']).size().unstack().fillna(0)
    response_rates['Response Rate'] = response_rates[True] / (response_rates[True] + response_rates[False]) * 100
    
    fig = px.bar(
        response_rates.reset_index(),
        x='GROUP',
        y='Response Rate',
        color='GROUP',
        title=f"Response Rate - {metrics[selected_metric]['title']}",
        labels={'Response Rate': 'Percentage of Patients with Improvement'},
        color_discrete_map={'Maitland + Conventional': '#3498db', 'Conventional Alone': '#e74c3c'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ===== DEMOGRAPHICS =====
st.markdown('<div class="section-title">📋 Patient Demographics</div>', unsafe_allow_html=True)

if 'AGE' in df.columns and 'GENDER' in df.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Distribution
        fig = px.histogram(
            filtered_df,
            x='AGE',
            color='GROUP',
            nbins=15,
            barmode='overlay',
            title="Age Distribution by Treatment Group",
            color_discrete_map={'Maitland + Conventional': '#3498db', 'Conventional Alone': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender Distribution
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

# ===== DATA EXPORT =====
st.markdown('<div class="section-title">💾 Data Export</div>', unsafe_allow_html=True)
st.download_button(
    "📥 Download Filtered Data as CSV",
    filtered_df.to_csv(index=False).encode('utf-8'),
    "knee_oa_treatment_outcomes.csv",
    "text/csv",
    key='download-csv'
)
