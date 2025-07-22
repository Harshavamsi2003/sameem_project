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
    page_title="Knee OA Treatment Comparison",
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
    .nav-button {
        width: 100%;
        margin-bottom: 10px;
    }
    .improvement-good {
        color: green;
        font-weight: bold;
    }
    .improvement-bad {
        color: red;
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
st.sidebar.header("🔍 Navigation & Filters")

# Navigation buttons
page = st.sidebar.radio("Go to:", 
                       ["🏠 Overview", 
                        "📊 Treatment Comparison", 
                        "📈 Patient Progress", 
                        "👥 Demographics"],
                       index=0)

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
st.sidebar.header("ℹ️ About This Dashboard")
st.sidebar.markdown("""
This dashboard compares outcomes between two treatment approaches for knee osteoarthritis:

- **Maitland + Conventional Therapy**  
  Joint mobilization techniques combined with standard physical therapy

- **Conventional Therapy Alone**  
  Standard physical therapy without joint mobilization

Use the navigation menu to explore different aspects of the data.
""")

# ========== HELPER FUNCTIONS ==========
def get_improvement_direction(metric, difference):
    """Determine if improvement is good or bad based on metric direction"""
    if metrics[metric]['direction'] == 'increase':
        return "better" if difference > 0 else "worse"
    else:
        return "better" if difference < 0 else "worse"

def get_improvement_class(metric, difference):
    """Get CSS class for improvement direction"""
    if metrics[metric]['direction'] == 'increase':
        return "improvement-good" if difference > 0 else "improvement-bad"
    else:
        return "improvement-good" if difference < 0 else "improvement-bad"

# ========== PAGE CONTENT ==========
if page == "🏠 Overview":
    st.title("🦵 Knee Osteoarthritis Treatment Outcomes")
    st.markdown("""
    **Comparing Maitland Mobilization + Conventional Therapy vs Conventional Therapy Alone**  
    *Analyzing pre-test to post-test measurements for clinical decision making*
    """)
    
    # Key Metrics Overview
    st.header("📌 Key Findings At A Glance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Pain reduction insight
        vas_diff = filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional']['VAS_CHANGE'].mean() - filtered_df[filtered_df['GROUP'] == 'Conventional Alone']['VAS_CHANGE'].mean()
        st.markdown(f"""
        <div class="insight-card">
            <h4>1. Pain Reduction</h4>
            <p>Maitland group showed <span class="highlight">{abs(vas_diff):.1f} points greater</span> reduction in pain (VAS)</p>
            <p class="{get_improvement_class('VAS', vas_diff)}">
                {get_improvement_direction('VAS', vas_diff).capitalize()} improvement
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Functional improvement insight
        womac_diff = filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional']['WOMAC_D_CHANGE'].mean() - filtered_df[filtered_df['GROUP'] == 'Conventional Alone']['WOMAC_D_CHANGE'].mean()
        st.markdown(f"""
        <div class="insight-card">
            <h4>2. Functional Improvement</h4>
            <p><span class="highlight">{abs(womac_diff):.1f} points greater</span> disability improvement with Maitland</p>
            <p class="{get_improvement_class('WOMAC_D', womac_diff)}">
                {get_improvement_direction('WOMAC_D', womac_diff).capitalize()} improvement
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # ROM improvement insight
        arom_diff = filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional']['AROM_CHANGE'].mean() - filtered_df[filtered_df['GROUP'] == 'Conventional Alone']['AROM_CHANGE'].mean()
        st.markdown(f"""
        <div class="insight-card">
            <h4>3. Mobility Gain</h4>
            <p><span class="highlight">{arom_diff:.1f}° greater</span> active ROM improvement with Maitland</p>
            <p class="{get_improvement_class('AROM', arom_diff)}">
                {get_improvement_direction('AROM', arom_diff).capitalize()} improvement
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Treatment Comparison Summary
    st.header("📊 Treatment Comparison Summary")
    
    # Create a summary table of improvements
    summary_data = []
    for metric in metrics:
        maitland_change = filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional'][f'{metric}_CHANGE'].mean()
        conv_change = filtered_df[filtered_df['GROUP'] == 'Conventional Alone'][f'{metric}_CHANGE'].mean()
        diff = maitland_change - conv_change
        
        summary_data.append({
            'Metric': metrics[metric]['title'],
            'Maitland Improvement': maitland_change,
            'Conventional Improvement': conv_change,
            'Difference': diff,
            'Unit': metrics[metric]['unit'],
            'Direction': metrics[metric]['direction']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Display the summary table with conditional formatting
    def color_diff(row):
        if row['Direction'] == 'increase':
            color = 'green' if row['Difference'] > 0 else 'red'
        else:
            color = 'green' if row['Difference'] < 0 else 'red'
        return [''] * (len(row)-1) + [f'color: {color}']
    
    st.dataframe(
        summary_df.style.format({
            'Maitland Improvement': '{:.1f} {Unit}',
            'Conventional Improvement': '{:.1f} {Unit}',
            'Difference': '{:.1f} {Unit}'
        }).apply(color_diff, axis=1),
        use_container_width=True,
        height=(len(summary_df) * 35) + 35
    )
    
    # Visual comparison
    st.subheader("Average Improvement by Treatment Group")
    fig = px.bar(
        summary_df.melt(id_vars=['Metric', 'Unit'], 
                       value_vars=['Maitland Improvement', 'Conventional Improvement'],
                       var_name='Group', value_name='Improvement'),
        x='Metric',
        y='Improvement',
        color='Group',
        barmode='group',
        labels={'Improvement': 'Improvement'},
        color_discrete_map={'Maitland Improvement': '#1f77b4', 'Conventional Improvement': '#ff7f0e'},
        text='Improvement'
    )
    fig.update_traces(texttemplate='%{y:.1f}', textposition='outside')
    fig.update_layout(
        showlegend=True,
        yaxis_title="Improvement",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Treatment Comparison":
    st.title("📊 Treatment Group Comparison")
    
    # Metric selector
    selected_metric = st.selectbox(
        "Select Metric to Compare",
        list(metrics.keys()),
        format_func=lambda x: metrics[x]['title'],
        key='comparison_metric'
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pre-Post Comparison
        fig = go.Figure()
        
        for group in ['Maitland + Conventional', 'Conventional Alone']:
            group_df = filtered_df[filtered_df['GROUP'] == group]
            
            fig.add_trace(go.Box(
                y=group_df[metrics[selected_metric]['pre']],
                name=f'{group} Pre',
                marker_color='#1f77b4' if group == 'Maitland + Conventional' else '#ff7f0e',
                boxmean=True,
                showlegend=True
            ))
            
            fig.add_trace(go.Box(
                y=group_df[metrics[selected_metric]['post']],
                name=f'{group} Post',
                marker_color='#1f77b4' if group == 'Maitland + Conventional' else '#ff7f0e',
                boxmean=True,
                showlegend=True
            ))
        
        fig.update_layout(
            title=f"Pre vs Post {metrics[selected_metric]['title']}",
            yaxis_title=f"{metrics[selected_metric]['title']} ({metrics[selected_metric]['unit']})",
            boxmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Improvement comparison
        fig = px.box(
            filtered_df,
            x='GROUP',
            y=f'{selected_metric}_CHANGE',
            color='GROUP',
            points="all",
            title=f"Improvement in {metrics[selected_metric]['title']}",
            labels={f'{selected_metric}_CHANGE': f"Change in {metrics[selected_metric]['title']} ({metrics[selected_metric]['unit']})"},
            color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'}
        )
        fig.update_layout(
            showlegend=False,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display statistics
        maitland_change = filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional'][f'{selected_metric}_CHANGE'].mean()
        conv_change = filtered_df[filtered_df['GROUP'] == 'Conventional Alone'][f'{selected_metric}_CHANGE'].mean()
        diff = maitland_change - conv_change
        
        direction = get_improvement_direction(selected_metric, diff)
        delta_color = "normal"
        if (metrics[selected_metric]['direction'] == 'increase' and diff > 0) or (metrics[selected_metric]['direction'] == 'decrease' and diff < 0):
            delta_color = "normal"
        else:
            delta_color = "inverse"
        
        st.metric(
            label="Average Improvement Difference",
            value=f"{abs(diff):.1f} {metrics[selected_metric]['unit']}",
            delta=f"Maitland shows {direction} improvement",
            delta_color=delta_color
        )

elif page == "📈 Patient Progress":
    st.title("📈 Individual Patient Progress")
    
    # Select metric for detailed view
    selected_metric = st.selectbox(
        "Select Metric to View Individual Progress",
        list(metrics.keys()),
        format_func=lambda x: metrics[x]['title'],
        key='progress_metric'
    )
    
    # Create detailed view
    col1, col2 = st.columns(2)
    
    with col1:
        # Slope chart for individual changes
        fig = go.Figure()
        
        for _, row in filtered_df.iterrows():
            fig.add_trace(go.Scatter(
                x=['Pre', 'Post'],
                y=[row[metrics[selected_metric]['pre']], row[metrics[selected_metric]['post']]],
                mode='lines+markers',
                line=dict(color='#1f77b4' if row['GROUP'] == 'Maitland + Conventional' else '#ff7f0e', width=1),
                showlegend=False,
                name=row['PATIENT_NAME'],
                hoverinfo='text',
                hovertext=f"{row['PATIENT_NAME']}<br>Group: {row['GROUP']}<br>Age: {row['AGE']}<br>Change: {row[f'{selected_metric}_CHANGE']:.1f} {metrics[selected_metric]['unit']}"
            ))
        
        fig.update_layout(
            title=f"Individual Changes - {metrics[selected_metric]['title']}",
            xaxis_title="Test Period",
            yaxis_title=f"{metrics[selected_metric]['title']} ({metrics[selected_metric]['unit']})",
            hovermode="closest",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Response analysis
        threshold = st.slider(
            "Define response threshold (% improvement)", 
            min_value=0, 
            max_value=100, 
            value=30 if metrics[selected_metric]['direction'] == 'decrease' else 10,
            key='response_threshold'
        )
        
        # Calculate response rates
        if metrics[selected_metric]['direction'] == 'decrease':
            maitland_response = (filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional'][f'{selected_metric}_PCT_CHANGE'] <= -threshold).mean() * 100
            conv_response = (filtered_df[filtered_df['GROUP'] == 'Conventional Alone'][f'{selected_metric}_PCT_CHANGE'] <= -threshold).mean() * 100
        else:
            maitland_response = (filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional'][f'{selected_metric}_PCT_CHANGE'] >= threshold).mean() * 100
            conv_response = (filtered_df[filtered_df['GROUP'] == 'Conventional Alone'][f'{selected_metric}_PCT_CHANGE'] >= threshold).mean() * 100
        
        # Display response rates
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Maitland + Conventional', 'Conventional Alone'],
            y=[maitland_response, conv_response],
            marker_color=['#1f77b4', '#ff7f0e'],
            text=[f"{maitland_response:.1f}%", f"{conv_response:.1f}%"],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"Response Rates ({threshold}% Improvement Threshold)",
            yaxis_title="Percentage of Patients",
            yaxis_range=[0, 100],
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "👥 Demographics":
    st.title("👥 Patient Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig = px.histogram(
            filtered_df,
            x='AGE',
            color='GROUP',
            nbins=10,
            barmode='overlay',
            title="Age Distribution by Treatment Group",
            color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'},
            labels={'AGE': 'Patient Age'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Duration distribution
        fig = px.box(
            filtered_df,
            x='GROUP',
            y='DURATION',
            color='GROUP',
            title="Treatment Duration by Group",
            color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'},
            labels={'DURATION': 'Duration (weeks)'}
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
            color_discrete_map={'M': '#1f77b4', 'F': '#ff7f0e'},
            labels={'Count': 'Number of Patients'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Age vs Improvement
        selected_metric = st.selectbox(
            "Select Metric for Age Correlation",
            list(metrics.keys()),
            format_func=lambda x: metrics[x]['title'],
            key='age_metric'
        )
        
        fig = px.scatter(
            filtered_df,
            x='AGE',
            y=f'{selected_metric}_CHANGE',
            color='GROUP',
            trendline="ols",
            title=f"Age vs Improvement in {metrics[selected_metric]['title']}",
            labels={
                'AGE': 'Patient Age',
                f'{selected_metric}_CHANGE': f"Change in {metrics[selected_metric]['title']} ({metrics[selected_metric]['unit']})"
            },
            color_discrete_map={'Maitland + Conventional': '#1f77b4', 'Conventional Alone': '#ff7f0e'}
        )
        st.plotly_chart(fig, use_container_width=True)

# ========== KEY INSIGHTS ==========
if page != "🏠 Overview":
    st.header("🔍 Key Insights")
    
    # Calculate meaningful insights
    maitland_df = filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional']
    conv_df = filtered_df[filtered_df['GROUP'] == 'Conventional Alone']
    
    # Insight 1: Pain reduction
    vas_diff = maitland_df['VAS_CHANGE'].mean() - conv_df['VAS_CHANGE'].mean()
    
    # Insight 2: ROM improvement
    arom_diff = maitland_df['AROM_CHANGE'].mean() - conv_df['AROM_CHANGE'].mean()
    
    # Insight 3: Response rates
    maitland_response = (maitland_df['VAS_PCT_CHANGE'] <= -30).mean() * 100
    conv_response = (conv_df['VAS_PCT_CHANGE'] <= -30).mean() * 100
    
    # Display insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="insight-card">
            <h4>1. Pain Reduction</h4>
            <p>Maitland group showed <span class="highlight">{abs(vas_diff):.1f} points greater</span> reduction in pain (VAS)</p>
            <p class="{get_improvement_class('VAS', vas_diff)}">
                {get_improvement_direction('VAS', vas_diff).capitalize()} improvement
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="insight-card">
            <h4>2. Mobility Gain</h4>
            <p><span class="highlight">{arom_diff:.1f}° more improvement</span> in active ROM with Maitland approach</p>
            <p class="{get_improvement_class('AROM', arom_diff)}">
                {get_improvement_direction('AROM', arom_diff).capitalize()} improvement
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="insight-card">
            <h4>3. Response Rates</h4>
            <p><span class="highlight">{maitland_response:.0f}% vs {conv_response:.0f}%</span> achieved 30%+ pain reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional insights
    col1, col2 = st.columns(2)
    
    with col1:
        # Age effect
        corr_maitland = maitland_df[['AGE', 'VAS_CHANGE']].corr().iloc[0,1]
        corr_conv = conv_df[['AGE', 'VAS_CHANGE']].corr().iloc[0,1]
        
        st.markdown(f"""
        <div class="insight-card">
            <h4>4. Age Effect</h4>
            <p>Younger patients showed better response (r={corr_maitland:.2f} Maitland, r={corr_conv:.2f} Conventional)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Duration effect
        dur_maitland = maitland_df[['DURATION', 'VAS_CHANGE']].corr().iloc[0,1]
        dur_conv = conv_df[['DURATION', 'VAS_CHANGE']].corr().iloc[0,1]
        
        st.markdown(f"""
        <div class="insight-card">
            <h4>5. Duration Impact</h4>
            <p>Longer treatment correlated with better outcomes (r={dur_maitland:.2f} Maitland, r={dur_conv:.2f} Conventional)</p>
        </div>
        """, unsafe_allow_html=True)
