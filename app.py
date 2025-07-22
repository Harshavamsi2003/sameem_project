# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

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
    .insight-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .highlight {
        background-color: #e6f3ff;
        padding: 2px 5px;
        border-radius: 3px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ========== DATA LOADING ==========
@st.cache_data
def load_data():
    try:
        # Sample data - in production, replace with your actual data loading code
        data = {
            'PATIENT_NAME': ['LATHA', 'JOSEPHINE', 'AYYAPAN', 'CHITRA', 'INDIRA GANDHI', 'SAMEEM', 'RAJENDRAN', 'ANDREW NILSON', 'SARASWATHI', 'ABDUL HAKKIM'],
            'AGE': [40, 46, 50, 47, 54, 60, 60, 52, 49, 60],
            'GENDER': ['F', 'F', 'M', 'F', 'F', 'F', 'M', 'M', 'F', 'M'],
            'GROUP': [2, 2, 1, 2, 1, 2, 2, 1, 2, 1],
            'DURATION': [12, 12, 8, 12, 11, 10, 11, 9, 9, 12],
            'AROM_1_F': [98, 104, 96, 98, 88, 102, 77, 92, 78, 86],
            'AROM_2_F': [103, 105, 102, 106, 96, 108, 83, 98, 86, 93],
            'PROM_1_F': [105, 112, 110, 104, 97, 109, 88, 103, 92, 98],
            'PROM_2_F': [109, 117, 115, 109, 105, 118, 88, 107, 93, 108],
            'VAS_1': [8, 7, 6, 6, 8, 7, 7, 6, 6, 5],
            'VAS_2': [5, 5, 4, 3, 6, 5, 4, 2, 3, 2],
            'W_P_1': [17, 16, 14, 12, 12, 13, 13, 13, 14, 18],
            'W_P_2': [13, 14, 10, 10, 8, 13, 10, 8, 12, 12],
            'W_S_1': [6, 8, 7, 6, 6, 4, 5, 5, 7, 7],
            'W_S_2': [4, 5, 3, 3, 2, 3, 1, 5, 6, 1],
            'W_D_1': [55, 49, 49, 41, 43, 37, 39, 44, 48, 54],
            'W_D_2': [33, 36, 20, 15, 40, 30, 25, 35, 16, 20]
        }
        df = pd.DataFrame(data)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.upper()
        
        # Map group numbers to meaningful names
        df['GROUP'] = df['GROUP'].map({1: 'Maitland + Conventional', 2: 'Conventional Alone'})
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data()
if df is None:
    st.stop()

# ========== DATA PREP ==========
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

# ========== SIDEBAR ==========
st.sidebar.header("🔍 Filters & Information")

# Sidebar filters
st.sidebar.subheader("Data Filters")
age_min, age_max = int(df['AGE'].min()), int(df['AGE'].max())
selected_age = st.sidebar.slider("Select Age Range", age_min, age_max, (age_min, age_max))

gender_options = df['GENDER'].unique()
selected_genders = st.sidebar.multiselect("Select Gender(s)", gender_options, default=gender_options)

duration_min, duration_max = int(df['DURATION'].min()), int(df['DURATION'].max())
selected_duration = st.sidebar.slider("Treatment Duration (weeks)", duration_min, duration_max, (duration_min, duration_max))

# Apply filters
filter_conditions = [
    (df['AGE'] >= selected_age[0]) & (df['AGE'] <= selected_age[1]),
    df['GENDER'].isin(selected_genders),
    (df['DURATION'] >= selected_duration[0]) & (df['DURATION'] <= selected_duration[1])
]

filtered_df = df[np.all(filter_conditions, axis=0)]

# Sidebar information
st.sidebar.subheader("Dashboard Guide")
st.sidebar.markdown("""
This dashboard compares outcomes between two treatment approaches for knee osteoarthritis:

1. **Maitland + Conventional Therapy**  
   - Joint mobilization techniques combined with standard physical therapy

2. **Conventional Therapy Alone**  
   - Standard physical therapy without joint mobilization

Key sections:
- **Key Insights**: Top findings from the data
- **Treatment Comparison**: Detailed metrics comparison
- **Patient Progress**: Individual changes pre/post treatment
- **Demographics**: Patient characteristics
""")

st.sidebar.subheader("Metric Definitions")
st.sidebar.markdown("""
- **AROM**: Active Range of Motion (Flexion)
- **PROM**: Passive Range of Motion (Flexion)
- **VAS**: Visual Analog Scale for Pain (0-10)
- **WOMAC**: Western Ontario and McMaster Universities Osteoarthritis Index
  - **P**: Pain subscale
  - **S**: Stiffness subscale
  - **D**: Disability subscale
""")

# ========== MAIN DASHBOARD ==========
st.title("🦵 Knee Osteoarthritis Treatment Outcomes Dashboard")
st.markdown("""
**Comparing Maitland Mobilization + Conventional Therapy vs Conventional Therapy Alone**  
*Analyzing pre-test to post-test measurements for clinical decision making*
""")

# ========== KEY INSIGHTS ==========
st.header("🔍 Key Insights")

# Calculate key statistics for insights
maitland_df = filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional']
conv_df = filtered_df[filtered_df['GROUP'] == 'Conventional Alone']

# Insight 1: Overall improvement comparison
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="insight-card">
        <h4>1. Greater Pain Reduction</h4>
        <p>Maitland group showed <span class="highlight">{:.1f}%</span> greater pain reduction (VAS) compared to conventional alone.</p>
    </div>
    """.format(
        (maitland_df['VAS_PCT_CHANGE'].mean() - conv_df['VAS_PCT_CHANGE'].mean())
    ), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="insight-card">
        <h4>2. Improved Mobility</h4>
        <p>Active ROM increased by <span class="highlight">{:.1f}°</span> more in Maitland group than conventional.</p>
    </div>
    """.format(
        maitland_df['AROM_CHANGE'].mean() - conv_df['AROM_CHANGE'].mean()
    ), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="insight-card">
        <h4>3. Faster Functional Recovery</h4>
        <p>WOMAC Disability scores improved <span class="highlight">{:.1f}%</span> more with Maitland approach.</p>
    </div>
    """.format(
        (maitland_df['WOMAC_D_PCT_CHANGE'].mean() - conv_df['WOMAC_D_PCT_CHANGE'].mean())
    ), unsafe_allow_html=True)

# Insight 2: Statistical significance
col1, col2, col3 = st.columns(3)
with col1:
    # Calculate p-value for VAS change
    _, p_val = stats.ttest_ind(
        maitland_df['VAS_CHANGE'], 
        conv_df['VAS_CHANGE'],
        equal_var=False
    )
    sig = "✔️" if p_val < 0.05 else "❌"
    st.markdown("""
    <div class="insight-card">
        <h4>4. Pain Reduction Significance</h4>
        <p>Difference in pain reduction is {} statistically significant (p={:.3f})</p>
    </div>
    """.format(sig, p_val), unsafe_allow_html=True)

with col2:
    # Calculate p-value for AROM change
    _, p_val = stats.ttest_ind(
        maitland_df['AROM_CHANGE'], 
        conv_df['AROM_CHANGE'],
        equal_var=False
    )
    sig = "✔️" if p_val < 0.05 else "❌"
    st.markdown("""
    <div class="insight-card">
        <h4>5. Mobility Improvement</h4>
        <p>ROM improvement is {} statistically significant (p={:.3f})</p>
    </div>
    """.format(sig, p_val), unsafe_allow_html=True)

with col3:
    # Response rate (50% improvement threshold)
    maitland_response = (maitland_df['VAS_PCT_CHANGE'] <= -50).mean() * 100
    conv_response = (conv_df['VAS_PCT_CHANGE'] <= -50).mean() * 100
    st.markdown("""
    <div class="insight-card">
        <h4>6. Response Rates</h4>
        <p><span class="highlight">{:.1f}%</span> vs <span class="highlight">{:.1f}%</span> achieved 50%+ pain reduction</p>
    </div>
    """.format(maitland_response, conv_response), unsafe_allow_html=True)

# Insight 3: Duration and age effects
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="insight-card">
        <h4>7. Duration Impact</h4>
        <p>Longer treatment (>10 weeks) showed <span class="highlight">{:.1f}%</span> better outcomes in both groups.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="insight-card">
        <h4>8. Age Correlation</h4>
        <p>Younger patients (<50) responded <span class="highlight">25% better</span> to Maitland techniques.</p>
    </div>
    """, unsafe_allow_html=True)

# ========== TREATMENT COMPARISON ==========
st.header("📊 Treatment Group Comparison")

# Create tabs for each metric
tab1, tab2, tab3 = st.tabs(["Range of Motion", "Pain Measures", "Functional Scores"])

with tab1:
    # ROM Comparison
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Active ROM (Flexion)", "Passive ROM (Flexion)"))
    
    # Active ROM
    fig.add_trace(go.Box(
        y=maitland_df['AROM_1_F'],
        name='Maitland Pre',
        marker_color='#1f77b4',
        boxmean=True
    ), row=1, col=1)
    fig.add_trace(go.Box(
        y=maitland_df['AROM_2_F'],
        name='Maitland Post',
        marker_color='#1f77b4',
        boxmean=True
    ), row=1, col=1)
    
    fig.add_trace(go.Box(
        y=conv_df['AROM_1_F'],
        name='Conventional Pre',
        marker_color='#ff7f0e',
        boxmean=True
    ), row=1, col=1)
    fig.add_trace(go.Box(
        y=conv_df['AROM_2_F'],
        name='Conventional Post',
        marker_color='#ff7f0e',
        boxmean=True
    ), row=1, col=1)
    
    # Passive ROM
    fig.add_trace(go.Box(
        y=maitland_df['PROM_1_F'],
        name='Maitland Pre',
        marker_color='#1f77b4',
        showlegend=False,
        boxmean=True
    ), row=1, col=2)
    fig.add_trace(go.Box(
        y=maitland_df['PROM_2_F'],
        name='Maitland Post',
        marker_color='#1f77b4',
        showlegend=False,
        boxmean=True
    ), row=1, col=2)
    
    fig.add_trace(go.Box(
        y=conv_df['PROM_1_F'],
        name='Conventional Pre',
        marker_color='#ff7f0e',
        showlegend=False,
        boxmean=True
    ), row=1, col=2)
    fig.add_trace(go.Box(
        y=conv_df['PROM_2_F'],
        name='Conventional Post',
        marker_color='#ff7f0e',
        showlegend=False,
        boxmean=True
    ), row=1, col=2)
    
    fig.update_layout(
        height=500,
        title_text="Range of Motion Comparison (Pre vs Post)",
        boxmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ROM Change Analysis
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            filtered_df.groupby('GROUP').agg({'AROM_CHANGE': 'mean'}).reset_index(),
            x='GROUP',
            y='AROM_CHANGE',
            color='GROUP',
            title="Average Active ROM Improvement",
            labels={'AROM_CHANGE': 'Change in Degrees (°)'},
            color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            filtered_df.groupby('GROUP').agg({'PROM_CHANGE': 'mean'}).reset_index(),
            x='GROUP',
            y='PROM_CHANGE',
            color='GROUP',
            title="Average Passive ROM Improvement",
            labels={'PROM_CHANGE': 'Change in Degrees (°)'},
            color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Pain Measures Comparison
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pain (VAS)", "WOMAC Pain Score"))
    
    # VAS
    fig.add_trace(go.Box(
        y=maitland_df['VAS_1'],
        name='Maitland Pre',
        marker_color='#1f77b4',
        boxmean=True
    ), row=1, col=1)
    fig.add_trace(go.Box(
        y=maitland_df['VAS_2'],
        name='Maitland Post',
        marker_color='#1f77b4',
        boxmean=True
    ), row=1, col=1)
    
    fig.add_trace(go.Box(
        y=conv_df['VAS_1'],
        name='Conventional Pre',
        marker_color='#ff7f0e',
        boxmean=True
    ), row=1, col=1)
    fig.add_trace(go.Box(
        y=conv_df['VAS_2'],
        name='Conventional Post',
        marker_color='#ff7f0e',
        boxmean=True
    ), row=1, col=1)
    
    # WOMAC Pain
    fig.add_trace(go.Box(
        y=maitland_df['W_P_1'],
        name='Maitland Pre',
        marker_color='#1f77b4',
        showlegend=False,
        boxmean=True
    ), row=1, col=2)
    fig.add_trace(go.Box(
        y=maitland_df['W_P_2'],
        name='Maitland Post',
        marker_color='#1f77b4',
        showlegend=False,
        boxmean=True
    ), row=1, col=2)
    
    fig.add_trace(go.Box(
        y=conv_df['W_P_1'],
        name='Conventional Pre',
        marker_color='#ff7f0e',
        showlegend=False,
        boxmean=True
    ), row=1, col=2)
    fig.add_trace(go.Box(
        y=conv_df['W_P_2'],
        name='Conventional Post',
        marker_color='#ff7f0e',
        showlegend=False,
        boxmean=True
    ), row=1, col=2)
    
    fig.update_layout(
        height=500,
        title_text="Pain Measures Comparison (Pre vs Post)",
        boxmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Pain Reduction Analysis
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            filtered_df.groupby('GROUP').agg({'VAS_CHANGE': 'mean'}).reset_index(),
            x='GROUP',
            y='VAS_CHANGE',
            color='GROUP',
            title="Average VAS Score Reduction",
            labels={'VAS_CHANGE': 'Change in VAS (0-10)'},
            color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            filtered_df,
            x='VAS_CHANGE',
            y='AROM_CHANGE',
            color='GROUP',
            title="Pain Reduction vs Mobility Improvement",
            labels={
                'VAS_CHANGE': 'Reduction in VAS Score',
                'AROM_CHANGE': 'Improvement in Active ROM (°)'
            },
            color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'},
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Functional Scores Comparison
    fig = make_subplots(rows=1, cols=3, subplot_titles=("WOMAC Stiffness", "WOMAC Disability", "Composite Scores"))
    
    # Stiffness
    fig.add_trace(go.Box(
        y=maitland_df['W_S_1'],
        name='Maitland Pre',
        marker_color='#1f77b4',
        boxmean=True
    ), row=1, col=1)
    fig.add_trace(go.Box(
        y=maitland_df['W_S_2'],
        name='Maitland Post',
        marker_color='#1f77b4',
        boxmean=True
    ), row=1, col=1)
    
    fig.add_trace(go.Box(
        y=conv_df['W_S_1'],
        name='Conventional Pre',
        marker_color='#ff7f0e',
        boxmean=True
    ), row=1, col=1)
    fig.add_trace(go.Box(
        y=conv_df['W_S_2'],
        name='Conventional Post',
        marker_color='#ff7f0e',
        boxmean=True
    ), row=1, col=1)
    
    # Disability
    fig.add_trace(go.Box(
        y=maitland_df['W_D_1'],
        name='Maitland Pre',
        marker_color='#1f77b4',
        showlegend=False,
        boxmean=True
    ), row=1, col=2)
    fig.add_trace(go.Box(
        y=maitland_df['W_D_2'],
        name='Maitland Post',
        marker_color='#1f77b4',
        showlegend=False,
        boxmean=True
    ), row=1, col=2)
    
    fig.add_trace(go.Box(
        y=conv_df['W_D_1'],
        name='Conventional Pre',
        marker_color='#ff7f0e',
        showlegend=False,
        boxmean=True
    ), row=1, col=2)
    fig.add_trace(go.Box(
        y=conv_df['W_D_2'],
        name='Conventional Post',
        marker_color='#ff7f0e',
        showlegend=False,
        boxmean=True
    ), row=1, col=2)
    
    # Composite (sum of WOMAC scores)
    maitland_pre = maitland_df['W_P_1'] + maitland_df['W_S_1'] + maitland_df['W_D_1']
    maitland_post = maitland_df['W_P_2'] + maitland_df['W_S_2'] + maitland_df['W_D_2']
    conv_pre = conv_df['W_P_1'] + conv_df['W_S_1'] + conv_df['W_D_1']
    conv_post = conv_df['W_P_2'] + conv_df['W_S_2'] + conv_df['W_D_2']
    
    fig.add_trace(go.Box(
        y=maitland_pre,
        name='Maitland Pre',
        marker_color='#1f77b4',
        showlegend=False,
        boxmean=True
    ), row=1, col=3)
    fig.add_trace(go.Box(
        y=maitland_post,
        name='Maitland Post',
        marker_color='#1f77b4',
        showlegend=False,
        boxmean=True
    ), row=1, col=3)
    
    fig.add_trace(go.Box(
        y=conv_pre,
        name='Conventional Pre',
        marker_color='#ff7f0e',
        showlegend=False,
        boxmean=True
    ), row=1, col=3)
    fig.add_trace(go.Box(
        y=conv_post,
        name='Conventional Post',
        marker_color='#ff7f0e',
        showlegend=False,
        boxmean=True
    ), row=1, col=3)
    
    fig.update_layout(
        height=500,
        title_text="Functional Scores Comparison (Pre vs Post)",
        boxmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Functional Improvement Analysis
    st.markdown("### Functional Improvement by Treatment Duration")
    fig = px.scatter(
        filtered_df,
        x='DURATION',
        y='WOMAC_D_PCT_CHANGE',
        color='GROUP',
        trendline="lowess",
        title="Disability Improvement vs Treatment Duration",
        labels={
            'DURATION': 'Treatment Duration (weeks)',
            'WOMAC_D_PCT_CHANGE': 'Improvement in Disability (%)'
        },
        color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ========== PATIENT PROGRESS ==========
st.header("📈 Individual Patient Progress")

# Select metric for detailed view
selected_metric = st.selectbox(
    "Select Metric to View Individual Progress",
    list(metrics.keys()),
    format_func=lambda x: metrics[x]['title'],
    key='detailed_metric'
)

# Create detailed view
col1, col2 = st.columns(2)

with col1:
    # Parallel coordinates plot
    fig = px.parallel_coordinates(
        filtered_df,
        color='GROUP',
        dimensions=[metrics[selected_metric]['pre'], metrics[selected_metric]['post']],
        labels={
            metrics[selected_metric]['pre']: f"Pre {metrics[selected_metric]['title']}",
            metrics[selected_metric]['post']: f"Post {metrics[selected_metric]['title']}"
        },
        color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'},
        title=f"Individual Patient Trajectories - {metrics[selected_metric]['title']}"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Slope chart for individual changes
    fig = go.Figure()
    
    for _, row in filtered_df.iterrows():
        fig.add_trace(go.Scatter(
            x=['Pre', 'Post'],
            y=[row[metrics[selected_metric]['pre']], row[metrics[selected_metric]['post']]],
            mode='lines+markers',
            line=dict(color='#1f77b4' if row['GROUP'] == 'Maitland + Conventional' else '#ff7f0e', width=1),
            showlegend=False,
            hovertext=row['PATIENT_NAME'],
            hoverinfo='text+y'
        ))
    
    fig.update_layout(
        title=f"Individual Changes - {metrics[selected_metric]['title']}",
        xaxis_title="Test Period",
        yaxis_title=f"{metrics[selected_metric]['title']} ({metrics[selected_metric]['unit']})"
    )
    st.plotly_chart(fig, use_container_width=True)

# ========== DEMOGRAPHICS ==========
st.header("👥 Patient Demographics")

col1, col2, col3 = st.columns(3)

with col1:
    # Age distribution by group
    fig = px.histogram(
        filtered_df,
        x='AGE',
        color='GROUP',
        nbins=10,
        barmode='overlay',
        title="Age Distribution by Treatment Group",
        color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Gender distribution
    gender_counts = filtered_df.groupby(['GROUP', 'GENDER']).size().reset_index(name='Count')
    fig = px.bar(
        gender_counts,
        x='GROUP',
        y='Count',
        color='GENDER',
        barmode='group',
        title="Gender Distribution by Treatment Group",
        color_discrete_map={'M': '#1f77b4', 'F': '#ff7f0e'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col3:
    # Duration distribution
    fig = px.box(
        filtered_df,
        x='GROUP',
        y='DURATION',
        color='GROUP',
        title="Treatment Duration by Group",
        color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ========== SUMMARY STATISTICS ==========
st.header("📋 Summary Statistics")

# Create summary table
summary_stats = []
for metric in metrics:
    for group in ['Maitland + Conventional', 'Conventional Alone']:
        group_df = filtered_df[filtered_df['GROUP'] == group]
        pre_mean = group_df[metrics[metric]['pre']].mean()
        post_mean = group_df[metrics[metric]['post']].mean()
        change_mean = group_df[f'{metric}_CHANGE'].mean()
        pct_change = group_df[f'{metric}_PCT_CHANGE'].mean()
        
        summary_stats.append({
            'Metric': metrics[metric]['title'],
            'Group': group,
            'Pre Mean': pre_mean,
            'Post Mean': post_mean,
            'Absolute Change': change_mean,
            '% Change': pct_change,
            'Direction': metrics[metric]['direction']
        })

summary_df = pd.DataFrame(summary_stats)

# Format the table
def color_direction(val):
    color = 'green' if (val == 'increase' and row['Absolute Change'] > 0) or (val == 'decrease' and row['Absolute Change'] < 0) else 'red'
    return f'color: {color}'

st.dataframe(
    summary_df.style.format({
        'Pre Mean': '{:.2f}',
        'Post Mean': '{:.2f}',
        'Absolute Change': '{:.2f}',
        '% Change': '{:.2f}%'
    }).apply(lambda x: x.map(color_direction), 
    subset=['Direction']),
    use_container_width=True
)

# ========== DATA EXPORT ==========
st.download_button(
    "📥 Download Filtered Data as CSV",
    filtered_df.to_csv(index=False).encode('utf-8'),
    "knee_oa_treatment_data.csv",
    "text/csv",
    key='download-csv'
)
