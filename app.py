import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Knee OA Treatment Analytics",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_excel("RESEARCH_DATA.xlsx")

df = load_data()

# Data cleaning and preparation
df['GROUP'] = df['GROUP'].map({1: 'Maitland + Conventional', 2: 'Conventional Alone'})

# Define metrics
metrics = {
    'AROM': {
        'pre': ['AROM_1_F', 'AROM_1_E'],
        'post': ['AROM_2_F', 'AROM_2_E'],
        'title': 'Active Range of Motion',
        'y_label': 'Degrees'
    },
    'PROM': {
        'pre': ['PROM_1_F', 'PROM_1_E'],
        'post': ['PROM_2_F', 'PROM_2_E'],
        'title': 'Passive Range of Motion',
        'y_label': 'Degrees'
    },
    'VAS': {
        'pre': ['VAS_1'],
        'post': ['VAS_2'],
        'title': 'Visual Analogue Scale (Pain)',
        'y_label': 'Score (0-10)'
    },
    'WOMAC_P': {
        'pre': ['W_P_1'],
        'post': ['W_P_2'],
        'title': 'WOMAC Pain',
        'y_label': 'Score'
    },
    'WOMAC_S': {
        'pre': ['W_S_1'],
        'post': ['W_S_2'],
        'title': 'WOMAC Stiffness',
        'y_label': 'Score'
    },
    'WOMAC_D': {
        'pre': ['W_D_1'],
        'post': ['W_D_2'],
        'title': 'WOMAC Disability',
        'y_label': 'Score'
    }
}

# Calculate improvement percentages
for metric, cols in metrics.items():
    pre_col = cols['pre'][0]  # Taking first column if multiple
    post_col = cols['post'][0]
    df[f'{metric}_improvement'] = ((df[post_col] - df[pre_col]) / df[pre_col]) * 100

# Sidebar filters
st.sidebar.header("Filters")
selected_metric = st.sidebar.selectbox(
    "Select Metric to Analyze",
    list(metrics.keys()),
    format_func=lambda x: metrics[x]['title']
)

age_range = st.sidebar.slider(
    "Select Age Range",
    min_value=int(df['AGE'].min()),
    max_value=int(df['AGE'].max()),
    value=(int(df['AGE'].min()), int(df['AGE'].max()))
)

gender_filter = st.sidebar.multiselect(
    "Filter by Gender",
    options=df['GENDER'].unique(),
    default=df['GENDER'].unique()
)

duration_filter = st.sidebar.slider(
    "Treatment Duration (weeks)",
    min_value=int(df['DURATION'].min()),
    max_value=int(df['DURATION'].max()),
    value=(int(df['DURATION'].min()), int(df['DURATION'].max()))
)

# Apply filters
filtered_df = df[
    (df['AGE'] >= age_range[0]) & 
    (df['AGE'] <= age_range[1]) &
    (df['GENDER'].isin(gender_filter)) &
    (df['DURATION'] >= duration_filter[0]) &
    (df['DURATION'] <= duration_filter[1])
]

# Main content
st.title("Knee Osteoarthritis Treatment Outcomes Dashboard")
st.markdown("""
Comparing outcomes between:
- **Group 1**: Maitland mobilization + Conventional therapy
- **Group 2**: Conventional therapy alone
""")

# Key metrics overview
st.subheader("Overall Treatment Effectiveness")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Patients", len(filtered_df))
    
with col2:
    avg_improvement = filtered_df[f'{selected_metric}_improvement'].mean()
    st.metric(
        f"Avg {metrics[selected_metric]['title']} Improvement", 
        f"{avg_improvement:.1f}%",
        delta=f"{(avg_improvement - df[f'{selected_metric}_improvement'].mean()):.1f}% vs unfiltered"
    )

with col3:
    group_comparison = filtered_df.groupby('GROUP')[f'{selected_metric}_improvement'].mean()
    st.metric(
        "Group Difference", 
        f"{(group_comparison['Maitland + Conventional'] - group_comparison['Conventional Alone']):.1f}%",
        delta="Maitland vs Conventional"
    )

# Visualization section
st.subheader(f"{metrics[selected_metric]['title']} Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["Pre-Post Comparison", "Improvement by Group", "Demographic Analysis", "Statistical Tests"])

with tab1:
    # Pre-post comparison plot
    fig = go.Figure()
    
    for group in filtered_df['GROUP'].unique():
        group_df = filtered_df[filtered_df['GROUP'] == group]
        
        # Add pre-test data
        fig.add_trace(go.Violin(
            x=['Pre-Test'] * len(group_df),
            y=group_df[metrics[selected_metric]['pre'][0]],
            name=f'{group} (Pre)',
            box_visible=True,
            meanline_visible=True,
            line_color='blue' if group == 'Maitland + Conventional' else 'red'
        ))
        
        # Add post-test data
        fig.add_trace(go.Violin(
            x=['Post-Test'] * len(group_df),
            y=group_df[metrics[selected_metric]['post'][0]],
            name=f'{group} (Post)',
            box_visible=True,
            meanline_visible=True,
            line_color='lightblue' if group == 'Maitland + Conventional' else 'pink'
        ))
    
    fig.update_layout(
        title=f"Pre-Test vs Post-Test {metrics[selected_metric]['title']}",
        yaxis_title=metrics[selected_metric]['y_label'],
        violinmode='group',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Improvement by group
    fig = px.box(
        filtered_df, 
        x='GROUP', 
        y=f'{selected_metric}_improvement',
        color='GROUP',
        points="all",
        title=f"Percentage Improvement in {metrics[selected_metric]['title']} by Treatment Group",
        labels={f'{selected_metric}_improvement': 'Improvement (%)', 'GROUP': 'Treatment Group'},
        color_discrete_map={'Maitland + Conventional': 'blue', 'Conventional Alone': 'red'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Improvement vs duration
    fig = px.scatter(
        filtered_df,
        x='DURATION',
        y=f'{selected_metric}_improvement',
        color='GROUP',
        trendline="lowess",
        title=f"Improvement vs Treatment Duration",
        labels={'DURATION': 'Duration (weeks)', f'{selected_metric}_improvement': 'Improvement (%)'},
        color_discrete_map={'Maitland + Conventional': 'blue', 'Conventional Alone': 'red'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Age vs improvement
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            filtered_df,
            x='AGE',
            y=f'{selected_metric}_improvement',
            color='GROUP',
            trendline="lowess",
            title=f"Improvement by Age",
            labels={'AGE': 'Age (years)', f'{selected_metric}_improvement': 'Improvement (%)'},
            color_discrete_map={'Maitland + Conventional': 'blue', 'Conventional Alone': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            filtered_df,
            x='GENDER',
            y=f'{selected_metric}_improvement',
            color='GROUP',
            title=f"Improvement by Gender",
            labels={'GENDER': 'Gender', f'{selected_metric}_improvement': 'Improvement (%)'},
            color_discrete_map={'Maitland + Conventional': 'blue', 'Conventional Alone': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Statistical Comparison Between Groups")
    
    # Prepare data for statistical tests
    group1 = filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional']
    group2 = filtered_df[filtered_df['GROUP'] == 'Conventional Alone']
    
    # Normality test
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Normality Tests (Shapiro-Wilk)**")
        _, p1 = stats.shapiro(group1[f'{selected_metric}_improvement'])
        _, p2 = stats.shapiro(group2[f'{selected_metric}_improvement'])
        
        st.write(f"- Maitland group p-value: {p1:.4f} {'(normal)' if p1 > 0.05 else '(not normal)'}")
        st.write(f"- Conventional group p-value: {p2:.4f} {'(normal)' if p2 > 0.05 else '(not normal)'}")
    
    with col2:
        st.markdown("**Variance Equality (Levene's Test)**")
        _, p_var = stats.levene(
            group1[f'{selected_metric}_improvement'],
            group2[f'{selected_metric}_improvement']
        )
        st.write(f"p-value: {p_var:.4f} {'(equal variances)' if p_var > 0.05 else '(unequal variances)'}")
    
    # Appropriate statistical test
    if p1 > 0.05 and p2 > 0.05:  # Both normal
        st.markdown("**Independent Samples t-test**")
        t_stat, p_val = stats.ttest_ind(
            group1[f'{selected_metric}_improvement'],
            group2[f'{selected_metric}_improvement'],
            equal_var=(p_var > 0.05)
        )
        test_used = "t-test"
    else:
        st.markdown("**Mann-Whitney U Test**")
        u_stat, p_val = stats.mannwhitneyu(
            group1[f'{selected_metric}_improvement'],
            group2[f'{selected_metric}_improvement']
        )
        test_used = "Mann-Whitney U"
    
    st.write(f"- Test used: {test_used}")
    st.write(f"- p-value: {p_val:.4f}")
    st.write(f"- Significant difference: {'YES' if p_val < 0.05 else 'NO'}")
    
    # Effect size
    cohen_d = (group1[f'{selected_metric}_improvement'].mean() - group2[f'{selected_metric}_improvement'].mean()) / np.sqrt(
        (group1[f'{selected_metric}_improvement'].std()**2 + group2[f'{selected_metric}_improvement'].std()**2) / 2
    )
    st.markdown("**Effect Size (Cohen's d)**")
    st.write(f"d = {cohen_d:.2f}")
    st.write("Interpretation:")
    st.write("- 0.2: Small effect")
    st.write("- 0.5: Medium effect")
    st.write("- 0.8: Large effect")

# Raw data view
st.subheader("Filtered Data")
st.dataframe(filtered_df, use_container_width=True)

# Download button
st.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='filtered_treatment_data.csv',
    mime='text/csv'
)
