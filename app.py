# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Knee OA Treatment Outcomes Dashboard",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    body {
        color: #000000;  /* Black text */
        font-family: Arial, sans-serif;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .header-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #000000;
        margin-bottom: 10px;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #000000;
        border-bottom: 1px solid #dddddd;
        padding-bottom: 5px;
        margin-top: 20px;
    }
    .st-bb {
        background-color: white;
    }
    .st-at {
        background-color: #f0f0f0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# ========== DATA LOADING ==========
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("RESEARCH_DATA.xlsx")
        df.columns = df.columns.str.strip().str.upper()
        
        # Check for required columns
        required_cols = ['GROUP', 'AROM_1_F', 'AROM_2_F', 'PROM_1_F', 'PROM_2_F',
                        'VAS_1', 'VAS_2', 'W_P_1', 'W_P_2', 'W_S_1', 'W_S_2', 
                        'W_D_1', 'W_D_2']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
            
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
st.sidebar.markdown("Customize the data view:")

# Age filter
if 'AGE' in df.columns:
    age_min, age_max = int(df['AGE'].min()), int(df['AGE'].max())
    selected_age = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))
else:
    selected_age = (0, 100)

# Gender filter
if 'GENDER' in df.columns:
    gender_options = df['GENDER'].unique()
    selected_genders = st.sidebar.multiselect("Gender", gender_options, default=gender_options)
else:
    selected_genders = []

# Duration filter
if 'DURATION' in df.columns:
    duration_min, duration_max = int(df['DURATION'].min()), int(df['DURATION'].max())
    selected_duration = st.sidebar.slider("Duration (weeks)", duration_min, duration_max, (duration_min, duration_max))
else:
    selected_duration = (0, 20)

# Group filter
selected_groups = st.sidebar.multiselect(
    "Treatment Groups",
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
st.title("Knee Osteoarthritis Treatment Outcomes")
st.markdown("""
Comparing **Maitland Mobilization + Conventional Therapy** vs **Conventional Therapy Alone**  
*Pre-test to post-test clinical measures analysis*
""")

# ===== KEY METRICS =====
st.markdown('<div class="section-title">📊 Key Outcomes</div>', unsafe_allow_html=True)

metric_cols = st.columns(3)
with metric_cols[0]:
    st.markdown('<div class="metric-card">'
                '<h3>Total Patients</h3>'
                f'<p style="font-size: 1.5rem;">{len(filtered_df)}</p>'
                '</div>', unsafe_allow_html=True)
    
with metric_cols[1]:
    avg_improve_arom = filtered_df['AROM_CHANGE'].mean()
    st.markdown('<div class="metric-card">'
                '<h3>Avg AROM Change</h3>'
                f'<p style="font-size: 1.5rem;">{avg_improve_arom:.1f}°</p>'
                '</div>', unsafe_allow_html=True)
    
with metric_cols[2]:
    avg_improve_vas = filtered_df['VAS_CHANGE'].mean()
    st.markdown('<div class="metric-card">'
                '<h3>Avg Pain Change</h3>'
                f'<p style="font-size: 1.5rem;">{avg_improve_vas:.1f} points</p>'
                '</div>', unsafe_allow_html=True)

# ===== COMPARISON CHARTS =====
st.markdown('<div class="section-title">📈 Treatment Comparisons</div>', unsafe_allow_html=True)

# Select metric for comparison
selected_metric = st.selectbox(
    "Select Metric to Compare",
    list(metrics.keys()),
    format_func=lambda x: metrics[x]['title'],
    key='comparison_metric'
)

col1, col2 = st.columns(2)

with col1:
    # Pre-Post Comparison by Group
    fig = px.box(
        filtered_df,
        x='GROUP',
        y=[metrics[selected_metric]['pre'], metrics[selected_metric]['post']],
        color='GROUP',
        labels={'value': f"{metrics[selected_metric]['title']} ({metrics[selected_metric]['unit']})", 'variable': 'Test'},
        title=f"Pre vs Post {metrics[selected_metric]['title']}",
        color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'}
    )
    fig.update_layout(
        boxmode='group',
        legend_title_text='Treatment Group',
        yaxis_title=f"{metrics[selected_metric]['title']} ({metrics[selected_metric]['unit']})"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Change Distribution
    fig = px.box(
        filtered_df,
        x='GROUP',
        y=f'{selected_metric}_CHANGE',
        color='GROUP',
        points="all",
        title=f"Improvement in {metrics[selected_metric]['title']}",
        labels={f'{selected_metric}_CHANGE': f"Change ({metrics[selected_metric]['unit']})"},
        color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ===== DETAILED VIEW =====
st.markdown('<div class="section-title">🔍 Detailed Analysis</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Scatter plot with reference line
    fig = px.scatter(
        filtered_df,
        x=metrics[selected_metric]['pre'],
        y=metrics[selected_metric]['post'],
        color='GROUP',
        trendline="ols",
        title=f"Pre vs Post Correlation",
        labels={
            metrics[selected_metric]['pre']: f"Pre-Test ({metrics[selected_metric]['unit']})",
            metrics[selected_metric]['post']: f"Post-Test ({metrics[selected_metric]['unit']})"
        },
        color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'}
    )
    fig.add_shape(
        type="line", line=dict(dash='dash', color='black'),
        x0=filtered_df[metrics[selected_metric]['pre']].min(),
        y0=filtered_df[metrics[selected_metric]['pre']].min(),
        x1=filtered_df[metrics[selected_metric]['pre']].max(),
        y1=filtered_df[metrics[selected_metric]['pre']].max()
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Response Rates
    response_rates = filtered_df.groupby(['GROUP', f'{selected_metric}_IMPROVED']).size().unstack().fillna(0)
    response_rates['Response Rate'] = response_rates[True] / (response_rates[True] + response_rates[False]) * 100
    
    fig = px.bar(
        response_rates.reset_index(),
        x='GROUP',
        y='Response Rate',
        color='GROUP',
        title="Percentage of Patients with Improvement",
        labels={'Response Rate': 'Percentage Improved'},
        color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ===== DATA EXPORT =====
st.markdown('<div class="section-title">💾 Export Data</div>', unsafe_allow_html=True)
st.download_button(
    "Download Filtered Data (CSV)",
    filtered_df.to_csv(index=False).encode('utf-8'),
    "knee_oa_treatment_data.csv",
    "text/csv"
)
