import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Osteoarthritis Treatment Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    # Read the Excel file
    df = pd.read_excel("RESEARCH_DATA.xlsx")
    
    # Ensure column names match exactly what's in the Excel file
    # Calculate differences between pre and post tests
    df['AROM_DIFF'] = df['AROM_2_F'] - df['AROM_1_F']
    df['PROM_DIFF'] = df['PROM_2_F'] - df['PROM_1_F']
    df['VAS_DIFF'] = df['VAS_2'] - df['VAS_1']
    df['W_P_DIFF'] = df['W_P_2'] - df['W_P_1']
    df['W_S_DIFF'] = df['W_S_2'] - df['W_S_1']
    df['W_D_DIFF'] = df['W_D_2'] - df['W_D_1']
    
    # Map group numbers to meaningful names
    df['GROUP_NAME'] = df['GROUP'].map({1: 'Maitland + Conventional', 2: 'Conventional Alone'})
    
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
group_filter = st.sidebar.multiselect(
    "Select Treatment Group(s)",
    options=df['GROUP_NAME'].unique(),
    default=df['GROUP_NAME'].unique()
)

gender_filter = st.sidebar.multiselect(
    "Select Gender(s)",
    options=df['GENDER'].unique(),
    default=df['GENDER'].unique()
)

# Ensure AGE column exists before using it
if 'AGE' in df.columns:
    age_range = st.sidebar.slider(
        "Select Age Range",
        min_value=int(df['AGE'].min()),
        max_value=int(df['AGE'].max()),
        value=(int(df['AGE'].min()), int(df['AGE'].max()))
    )
else:
    st.sidebar.warning("AGE column not found in data")
    age_range = (0, 100)  # Default range if AGE column is missing

# Ensure DURATION column exists before using it
if 'DURATION' in df.columns:
    duration_range = st.sidebar.slider(
        "Select Duration Range (weeks)",
        min_value=int(df['DURATION'].min()),
        max_value=int(df['DURATION'].max()),
        value=(int(df['DURATION'].min()), int(df['DURATION'].max()))
    )
else:
    st.sidebar.warning("DURATION column not found in data")
    duration_range = (0, 20)  # Default range if DURATION column is missing

# Apply filters
filter_conditions = [
    df['GROUP_NAME'].isin(group_filter),
    df['GENDER'].isin(gender_filter)
]

# Only add age filter if AGE column exists
if 'AGE' in df.columns:
    filter_conditions.append(df['AGE'].between(age_range[0], age_range[1]))

# Only add duration filter if DURATION column exists
if 'DURATION' in df.columns:
    filter_conditions.append(df['DURATION'].between(duration_range[0], duration_range[1]))

# Combine all conditions
if filter_conditions:
    filtered_df = df[pd.concat(filter_conditions, axis=1).all(axis=1)]
else:
    filtered_df = df.copy()

# Main page
st.title("Osteoarthritis Treatment Outcomes Analysis")
st.markdown("""
This dashboard compares the effectiveness of two treatment approaches for osteoarthritis:
- **Group 1**: Maitland mobilization + Conventional therapy
- **Group 2**: Conventional therapy alone
""")

# Key Metrics
st.subheader("Key Metrics Comparison")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Patients", len(filtered_df))
    if 'AGE' in filtered_df.columns:
        st.metric("Average Age", f"{filtered_df['AGE'].mean():.1f} years")
    else:
        st.metric("Average Age", "Data not available")

with col2:
    st.metric("Group 1 Patients", len(filtered_df[filtered_df['GROUP'] == 1]))
    st.metric("Group 2 Patients", len(filtered_df[filtered_df['GROUP'] == 2]))

with col3:
    if 'DURATION' in filtered_df.columns:
        avg_duration = filtered_df['DURATION'].mean()
        st.metric("Average Treatment Duration", f"{avg_duration:.1f} weeks")
    else:
        st.metric("Average Treatment Duration", "Data not available")

# Pre-Post Comparison Tabs
st.subheader("Pre-Test vs Post-Test Comparison")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Active ROM", "Passive ROM", "VAS", 
    "WOMAC Pain", "WOMAC Stiffness", "WOMAC Disability"
])

with tab1:
    st.markdown("**Active Range of Motion (Flexion)**")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pre-Test", "Post-Test"))
    
    for i, group in enumerate(filtered_df['GROUP_NAME'].unique(), 1):
        group_data = filtered_df[filtered_df['GROUP_NAME'] == group]
        fig.add_trace(
            go.Box(y=group_data['AROM_1_F'], name=f"{group} Pre", marker_color='blue' if i == 1 else 'red'),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=group_data['AROM_2_F'], name=f"{group} Post", marker_color='lightblue' if i == 1 else 'pink'),
            row=1, col=2
        )
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Improvement analysis
    st.markdown("**Improvement in Active ROM**")
    fig_diff = px.box(
        filtered_df, 
        x='GROUP_NAME', 
        y='AROM_DIFF', 
        color='GROUP_NAME',
        labels={'AROM_DIFF': 'Improvement in Active ROM (degrees)', 'GROUP_NAME': 'Treatment Group'}
    )
    st.plotly_chart(fig_diff, use_container_width=True)

with tab2:
    st.markdown("**Passive Range of Motion (Flexion)**")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pre-Test", "Post-Test"))
    
    for i, group in enumerate(filtered_df['GROUP_NAME'].unique(), 1):
        group_data = filtered_df[filtered_df['GROUP_NAME'] == group]
        fig.add_trace(
            go.Box(y=group_data['PROM_1_F'], name=f"{group} Pre", marker_color='blue' if i == 1 else 'red'),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=group_data['PROM_2_F'], name=f"{group} Post", marker_color='lightblue' if i == 1 else 'pink'),
            row=1, col=2
        )
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**Improvement in Passive ROM**")
    fig_diff = px.box(
        filtered_df, 
        x='GROUP_NAME', 
        y='PROM_DIFF', 
        color='GROUP_NAME',
        labels={'PROM_DIFF': 'Improvement in Passive ROM (degrees)', 'GROUP_NAME': 'Treatment Group'}
    )
    st.plotly_chart(fig_diff, use_container_width=True)

with tab3:
    st.markdown("**Visual Analog Scale (Pain)**")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pre-Test", "Post-Test"))
    
    for i, group in enumerate(filtered_df['GROUP_NAME'].unique(), 1):
        group_data = filtered_df[filtered_df['GROUP_NAME'] == group]
        fig.add_trace(
            go.Box(y=group_data['VAS_1'], name=f"{group} Pre", marker_color='blue' if i == 1 else 'red'),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=group_data['VAS_2'], name=f"{group} Post", marker_color='lightblue' if i == 1 else 'pink'),
            row=1, col=2
        )
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**Reduction in VAS Score**")
    fig_diff = px.box(
        filtered_df, 
        x='GROUP_NAME', 
        y='VAS_DIFF', 
        color='GROUP_NAME',
        labels={'VAS_DIFF': 'Reduction in VAS Score', 'GROUP_NAME': 'Treatment Group'}
    )
    st.plotly_chart(fig_diff, use_container_width=True)

with tab4:
    st.markdown("**WOMAC Pain Score**")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pre-Test", "Post-Test"))
    
    for i, group in enumerate(filtered_df['GROUP_NAME'].unique(), 1):
        group_data = filtered_df[filtered_df['GROUP_NAME'] == group]
        fig.add_trace(
            go.Box(y=group_data['W_P_1'], name=f"{group} Pre", marker_color='blue' if i == 1 else 'red'),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=group_data['W_P_2'], name=f"{group} Post", marker_color='lightblue' if i == 1 else 'pink'),
            row=1, col=2
        )
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**Reduction in WOMAC Pain Score**")
    fig_diff = px.box(
        filtered_df, 
        x='GROUP_NAME', 
        y='W_P_DIFF', 
        color='GROUP_NAME',
        labels={'W_P_DIFF': 'Reduction in WOMAC Pain Score', 'GROUP_NAME': 'Treatment Group'}
    )
    st.plotly_chart(fig_diff, use_container_width=True)

with tab5:
    st.markdown("**WOMAC Stiffness Score**")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pre-Test", "Post-Test"))
    
    for i, group in enumerate(filtered_df['GROUP_NAME'].unique(), 1):
        group_data = filtered_df[filtered_df['GROUP_NAME'] == group]
        fig.add_trace(
            go.Box(y=group_data['W_S_1'], name=f"{group} Pre", marker_color='blue' if i == 1 else 'red'),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=group_data['W_S_2'], name=f"{group} Post", marker_color='lightblue' if i == 1 else 'pink'),
            row=1, col=2
        )
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**Reduction in WOMAC Stiffness Score**")
    fig_diff = px.box(
        filtered_df, 
        x='GROUP_NAME', 
        y='W_S_DIFF', 
        color='GROUP_NAME',
        labels={'W_S_DIFF': 'Reduction in WOMAC Stiffness Score', 'GROUP_NAME': 'Treatment Group'}
    )
    st.plotly_chart(fig_diff, use_container_width=True)

with tab6:
    st.markdown("**WOMAC Disability Score**")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pre-Test", "Post-Test"))
    
    for i, group in enumerate(filtered_df['GROUP_NAME'].unique(), 1):
        group_data = filtered_df[filtered_df['GROUP_NAME'] == group]
        fig.add_trace(
            go.Box(y=group_data['W_D_1'], name=f"{group} Pre", marker_color='blue' if i == 1 else 'red'),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=group_data['W_D_2'], name=f"{group} Post", marker_color='lightblue' if i == 1 else 'pink'),
            row=1, col=2
        )
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**Reduction in WOMAC Disability Score**")
    fig_diff = px.box(
        filtered_df, 
        x='GROUP_NAME', 
        y='W_D_DIFF', 
        color='GROUP_NAME',
        labels={'W_D_DIFF': 'Reduction in WOMAC Disability Score', 'GROUP_NAME': 'Treatment Group'}
    )
    st.plotly_chart(fig_diff, use_container_width=True)

# Summary Statistics
st.subheader("Summary Statistics by Group")
if not filtered_df.empty:
    summary_stats = filtered_df.groupby('GROUP_NAME').agg({
        'AROM_DIFF': ['mean', 'std', 'min', 'max'],
        'PROM_DIFF': ['mean', 'std', 'min', 'max'],
        'VAS_DIFF': ['mean', 'std', 'min', 'max'],
        'W_P_DIFF': ['mean', 'std', 'min', 'max'],
        'W_S_DIFF': ['mean', 'std', 'min', 'max'],
        'W_D_DIFF': ['mean', 'std', 'min', 'max']
    }).round(2)

    st.dataframe(summary_stats.style.background_gradient(cmap='Blues'))
else:
    st.warning("No data available after filtering")

# Correlation Analysis
st.subheader("Correlation Between Variables")
corr_cols = []
if 'AGE' in filtered_df.columns:
    corr_cols.append('AGE')
if 'DURATION' in filtered_df.columns:
    corr_cols.append('DURATION')
    
corr_cols.extend([
    'AROM_DIFF', 'PROM_DIFF', 'VAS_DIFF', 
    'W_P_DIFF', 'W_S_DIFF', 'W_D_DIFF'
])

if len(corr_cols) > 0:
    corr_matrix = filtered_df[corr_cols].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        range_color=[-1, 1]
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Not enough columns available for correlation analysis")

# Raw Data
st.subheader("Raw Data")
st.dataframe(filtered_df)
