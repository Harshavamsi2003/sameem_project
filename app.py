import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Set page config
st.set_page_config(
    page_title="Physical Therapy Research Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_excel("RESEARCH_DATA.xlsx")

df = load_data()

# Rename groups for better display
df['GROUP'] = df['GROUP'].map({1: 'Maitland + Conventional', 2: 'Conventional Alone'})

# Calculate differences
df['AROM_DIFF'] = df['AROM_2_F'] - df['AROM_1_F']
df['PROM_DIFF'] = df['PROM_2_F'] - df['PROM_1_F']
df['VAS_DIFF'] = df['VAS_2'] - df['VAS_1']
df['W_P_DIFF'] = df['W_P_2'] - df['W_P_1']
df['W_S_DIFF'] = df['W_S_2'] - df['W_S_1']
df['W_D_DIFF'] = df['W_D_2'] - df['W_D_1']

# Sidebar filters
st.sidebar.header("Filters")
group_filter = st.sidebar.multiselect(
    "Select Treatment Group(s)",
    options=df['GROUP'].unique(),
    default=df['GROUP'].unique()
)

age_range = st.sidebar.slider(
    "Select Age Range",
    min_value=int(df['AGE'].min()),
    max_value=int(df['AGE'].max()),
    value=(int(df['AGE'].min()), int(df['AGE'].max()))
)

gender_filter = st.sidebar.multiselect(
    "Select Gender(s)",
    options=df['GENDER'].unique(),
    default=df['GENDER'].unique()
)

# Apply filters
filtered_df = df[
    (df['GROUP'].isin(group_filter)) &
    (df['AGE'].between(age_range[0], age_range[1])) &
    (df['GENDER'].isin(gender_filter))
]

# Main content
st.title("Physical Therapy Treatment Outcomes Dashboard")
st.markdown("""
    Comparing the effectiveness of **Maitland + Conventional Therapy** vs **Conventional Therapy Alone**  
    across various patient metrics.
""")

# Key metrics
st.subheader("Key Metrics Comparison")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Patients", len(filtered_df))
    
with col2:
    st.metric("Average Age", f"{filtered_df['AGE'].mean():.1f} years")
    
with col3:
    gender_dist = filtered_df['GENDER'].value_counts(normalize=True) * 100
    st.metric("Gender Distribution", 
              f"{gender_dist.get('F', 0):.1f}% Female, {gender_dist.get('M', 0):.1f}% Male")

# Group comparison tabs
tab1, tab2, tab3 = st.tabs(["Range of Motion", "Pain & Disability", "Statistical Tests"])

with tab1:
    st.subheader("Range of Motion Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            filtered_df,
            x='GROUP',
            y='AROM_DIFF',
            color='GROUP',
            title='Active ROM Improvement (AROM_2 - AROM_1)',
            labels={'AROM_DIFF': 'Improvement (degrees)', 'GROUP': 'Treatment Group'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.box(
            filtered_df,
            x='GROUP',
            y='PROM_DIFF',
            color='GROUP',
            title='Passive ROM Improvement (PROM_2 - PROM_1)',
            labels={'PROM_DIFF': 'Improvement (degrees)', 'GROUP': 'Treatment Group'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            filtered_df,
            x='AROM_1_F',
            y='AROM_2_F',
            color='GROUP',
            trendline="ols",
            title='Pre vs Post Active ROM',
            labels={'AROM_1_F': 'Pre-treatment AROM', 'AROM_2_F': 'Post-treatment AROM'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.scatter(
            filtered_df,
            x='PROM_1_F',
            y='PROM_2_F',
            color='GROUP',
            trendline="ols",
            title='Pre vs Post Passive ROM',
            labels={'PROM_1_F': 'Pre-treatment PROM', 'PROM_2_F': 'Post-treatment PROM'}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Pain, Stiffness & Disability Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            filtered_df,
            x='GROUP',
            y='VAS_DIFF',
            color='GROUP',
            title='VAS Score Change (Lower = Better)',
            labels={'VAS_DIFF': 'Change in VAS Score', 'GROUP': 'Treatment Group'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.box(
            filtered_df,
            x='GROUP',
            y='W_P_DIFF',
            color='GROUP',
            title='WOMAC Pain Score Change (Lower = Better)',
            labels={'W_P_DIFF': 'Change in Pain Score', 'GROUP': 'Treatment Group'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            filtered_df,
            x='GROUP',
            y='W_S_DIFF',
            color='GROUP',
            title='WOMAC Stiffness Score Change (Lower = Better)',
            labels={'W_S_DIFF': 'Change in Stiffness Score', 'GROUP': 'Treatment Group'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.box(
            filtered_df,
            x='GROUP',
            y='W_D_DIFF',
            color='GROUP',
            title='WOMAC Disability Score Change (Lower = Better)',
            labels={'W_D_DIFF': 'Change in Disability Score', 'GROUP': 'Treatment Group'}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Statistical Significance Testing")
    
    if len(group_filter) == 2:
        st.write("""
        **Mann-Whitney U Test Results** (non-parametric test for group differences):
        """)
        
        metrics = {
            'AROM Improvement': 'AROM_DIFF',
            'PROM Improvement': 'PROM_DIFF',
            'VAS Change': 'VAS_DIFF',
            'WOMAC Pain Change': 'W_P_DIFF',
            'WOMAC Stiffness Change': 'W_S_DIFF',
            'WOMAC Disability Change': 'W_D_DIFF'
        }
        
        results = []
        for name, col in metrics.items():
            group1 = filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional'][col]
            group2 = filtered_df[filtered_df['GROUP'] == 'Conventional Alone'][col]
            stat, p = stats.mannwhitneyu(group1, group2)
            results.append({
                'Metric': name,
                'U Statistic': stat,
                'p-value': p,
                'Significant': p < 0.05
            })
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.apply(
            lambda x: ['background: lightgreen' if x['Significant'] else '' for i in x], 
            axis=1
        ))
        
        st.caption("""
        Green highlights indicate statistically significant differences (p < 0.05) between treatment groups.
        """)
    else:
        st.warning("Please select both treatment groups to compare statistical significance.")

# Age and Duration Analysis
st.subheader("Age and Treatment Duration Relationships")

col1, col2 = st.columns(2)

with col1:
    fig = px.scatter(
        filtered_df,
        x='AGE',
        y='AROM_DIFF',
        color='GROUP',
        trendline="ols",
        title='Age vs AROM Improvement',
        labels={'AGE': 'Patient Age', 'AROM_DIFF': 'AROM Improvement'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        filtered_df,
        x='DURATION',
        y='VAS_DIFF',
        color='GROUP',
        trendline="ols",
        title='Treatment Duration vs VAS Change',
        labels={'DURATION': 'Treatment Duration (weeks)', 'VAS_DIFF': 'VAS Score Change'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Raw data view
st.subheader("Raw Data View")
st.dataframe(filtered_df, use_container_width=True)

# Download button
st.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='filtered_research_data.csv',
    mime='text/csv'
)
