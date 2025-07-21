import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set page config
st.set_page_config(
    page_title="Maitland vs Conventional Therapy for Knee OA",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_excel("RESEARCH_DATA.xlsx")

df = load_data()

# Data preparation
df['GROUP_NAME'] = df['GROUP'].map({1: 'Maitland + Conventional', 2: 'Conventional Alone'})

# Calculate composite scores and differences
df['AROM_PRE'] = df['AROM_1_F'] + df['AROM_1_E']
df['AROM_POST'] = df['AROM_2_F'] + df['AROM_2_E']
df['AROM_DIFF'] = df['AROM_POST'] - df['AROM_PRE']

df['PROM_PRE'] = df['PROM_1_F'] + df['PROM_1_E']
df['PROM_POST'] = df['PROM_2_F'] + df['PROM_2_E']
df['PROM_DIFF'] = df['PROM_POST'] - df['PROM_PRE']

df['VAS_DIFF'] = df['VAS_2'] - df['VAS_1']  # Note: Lower is better for VAS
df['WOMAC_P_DIFF'] = df['W_P_2'] - df['W_P_1']  # Lower is better
df['WOMAC_S_DIFF'] = df['W_S_2'] - df['W_S_1']  # Lower is better
df['WOMAC_D_DIFF'] = df['W_D_2'] - df['W_D_1']  # Lower is better

# Define metrics for comparison
metrics = {
    'AROM': {
        'pre': 'AROM_PRE',
        'post': 'AROM_POST',
        'diff': 'AROM_DIFF',
        'title': 'Active ROM',
        'unit': 'degrees',
        'improvement': 'increase'
    },
    'PROM': {
        'pre': 'PROM_PRE',
        'post': 'PROM_POST',
        'diff': 'PROM_DIFF',
        'title': 'Passive ROM',
        'unit': 'degrees',
        'improvement': 'increase'
    },
    'VAS': {
        'pre': 'VAS_1',
        'post': 'VAS_2',
        'diff': 'VAS_DIFF',
        'title': 'Pain (VAS)',
        'unit': 'points',
        'improvement': 'decrease'
    },
    'WOMAC_P': {
        'pre': 'W_P_1',
        'post': 'W_P_2',
        'diff': 'WOMAC_P_DIFF',
        'title': 'WOMAC Pain',
        'unit': 'points',
        'improvement': 'decrease'
    },
    'WOMAC_S': {
        'pre': 'W_S_1',
        'post': 'W_S_2',
        'diff': 'WOMAC_S_DIFF',
        'title': 'WOMAC Stiffness',
        'unit': 'points',
        'improvement': 'decrease'
    },
    'WOMAC_D': {
        'pre': 'W_D_1',
        'post': 'W_D_2',
        'diff': 'WOMAC_D_DIFF',
        'title': 'WOMAC Disability',
        'unit': 'points',
        'improvement': 'decrease'
    }
}

# Sidebar filters
st.sidebar.header("Filters")
age_range = st.sidebar.slider(
    "Age Range",
    min_value=int(df['AGE'].min()),
    max_value=int(df['AGE'].max()),
    value=(int(df['AGE'].min()), int(df['AGE'].max()))
)
gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df['GENDER'].unique(),
    default=df['GENDER'].unique()
)
group_filter = st.sidebar.multiselect(
    "Treatment Group",
    options=df['GROUP_NAME'].unique(),
    default=df['GROUP_NAME'].unique()
)

# Apply filters
filtered_df = df[
    (df['AGE'].between(age_range[0], age_range[1])) &
    (df['GENDER'].isin(gender_filter)) &
    (df['GROUP_NAME'].isin(group_filter))
]

# Main dashboard
st.title("Effectiveness of Maitland Mobilization + Conventional Therapy vs Conventional Therapy Alone")
st.markdown("""
**Comparative Analysis of Pain, Range of Motion, and Functional Ability in Knee Osteoarthritis Patients**
""")

# Summary statistics
st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", len(filtered_df))
col2.metric("Average Age", f"{filtered_df['AGE'].mean():.1f} years")
col3.metric("Gender Distribution", 
           f"{len(filtered_df[filtered_df['GENDER']=='F'])} Female, {len(filtered_df[filtered_df['GENDER']=='M'])} Male")

# Group comparison header
st.subheader("Group-wise Comparison of Treatment Effectiveness")

# Metric selector
selected_metric = st.selectbox(
    "Select Metric to Compare:",
    options=list(metrics.keys()),
    format_func=lambda x: metrics[x]['title'],
    key='metric_selector'
)

metric_info = metrics[selected_metric]

# Group comparison visualization
col1, col2 = st.columns(2)

with col1:
    # Pre-Post comparison within groups
    st.markdown(f"**{metric_info['title']} Changes Within Groups**")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for line plot
    plot_data = filtered_df.melt(
        id_vars=['GROUP_NAME', 'PATIENT_NAME'],
        value_vars=[metric_info['pre'], metric_info['post']],
        var_name='Timepoint',
        value_name='Score'
    )
    plot_data['Timepoint'] = plot_data['Timepoint'].replace({
        metric_info['pre']: 'Pre-Treatment',
        metric_info['post']: 'Post-Treatment'
    })
    
    sns.lineplot(
        data=plot_data,
        x='Timepoint',
        y='Score',
        hue='GROUP_NAME',
        style='GROUP_NAME',
        markers=True,
        dashes=False,
        ci=95,
        palette='Set2'
    )
    plt.title(f'{metric_info["title"]} Changes\n(Mean with 95% CI)')
    plt.ylabel(f'{metric_info["title"]} ({metric_info["unit"]})')
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    # Between-group improvement comparison
    st.markdown(f"**Improvement Comparison Between Groups**")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate percentage improvement
    if metric_info['improvement'] == 'increase':
        filtered_df['IMPROVEMENT_PCT'] = (filtered_df[metric_info['diff']] / filtered_df[metric_info['pre']]) * 100
    else:
        filtered_df['IMPROVEMENT_PCT'] = (-filtered_df[metric_info['diff']] / filtered_df[metric_info['pre']]) * 100
    
    sns.barplot(
        data=filtered_df,
        x='GROUP_NAME',
        y='IMPROVEMENT_PCT',
        palette='Set2',
        ci=95,
        capsize=0.1
    )
    plt.title(f'Percentage Improvement in {metric_info["title"]}')
    plt.ylabel(f'% Improvement')
    plt.grid(True, axis='y', alpha=0.3)
    st.pyplot(fig)

# Statistical comparison
st.subheader("Statistical Comparison Between Groups")

# Calculate statistics
group1 = filtered_df[filtered_df['GROUP_NAME'] == 'Maitland + Conventional']
group2 = filtered_df[filtered_df['GROUP_NAME'] == 'Conventional Alone']

# Create metrics columns
stat_col1, stat_col2, stat_col3 = st.columns(3)

with stat_col1:
    st.metric(
        f"Mean Improvement - Maitland Group",
        f"{group1[metric_info['diff']].mean():.2f} {metric_info['unit']}",
        delta=f"{group1['IMPROVEMENT_PCT'].mean():.1f}% change"
    )

with stat_col2:
    st.metric(
        f"Mean Improvement - Conventional Group",
        f"{group2[metric_info['diff']].mean():.2f} {metric_info['unit']}",
        delta=f"{group2['IMPROVEMENT_PCT'].mean():.1f}% change"
    )

with stat_col3:
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(
        group1[metric_info['diff']],
        group2[metric_info['diff']],
        equal_var=False
    )
    
    # Calculate effect size
    pooled_std = np.sqrt((group1[metric_info['diff']].std()**2 + group2[metric_info['diff']].std()**2) / 2)
    cohens_d = (group1[metric_info['diff']].mean() - group2[metric_info['diff']].mean()) / pooled_std
    
    st.metric(
        "Difference Between Groups",
        f"{(group1[metric_info['diff']].mean() - group2[metric_info['diff']].mean()):.2f} {metric_info['unit']}",
        delta=f"p = {p_value:.4f} | d = {cohens_d:.2f}",
        delta_color="normal" if p_value < 0.05 else "off"
    )

# All metrics comparison
st.subheader("Comparison Across All Outcome Measures")

# Create a grid of small multiples
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (metric_key, m_info) in enumerate(metrics.items()):
    ax = axes[idx]
    
    # Plot grouped bar chart
    sns.barplot(
        data=filtered_df,
        x='GROUP_NAME',
        y=m_info['diff'],
        palette='Set2',
        ci=95,
        capsize=0.1,
        ax=ax
    )
    
    # Format based on improvement direction
    if m_info['improvement'] == 'increase':
        ax.set_title(f'Δ{m_info["title"]} (Higher=Better)')
    else:
        ax.set_title(f'Δ{m_info["title"]} (Lower=Better)')
    
    ax.set_ylabel(f'Change ({m_info["unit"]})')
    ax.set_xlabel('')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add significance asterisk if p < 0.05
    t_stat, p_val = stats.ttest_ind(
        filtered_df[filtered_df['GROUP_NAME'] == 'Maitland + Conventional'][m_info['diff']],
        filtered_df[filtered_df['GROUP_NAME'] == 'Conventional Alone'][m_info['diff']],
        equal_var=False
    )
    if p_val < 0.05:
        ax.text(0.5, 0.9, '*', transform=ax.transAxes, 
               fontsize=20, ha='center', color='red')

plt.tight_layout()
st.pyplot(fig)

# Individual patient trajectories
st.subheader("Individual Patient Trajectories")

# Select random sample if too many patients
if len(filtered_df) > 15:
    sample_df = filtered_df.sample(15, random_state=42)
else:
    sample_df = filtered_df

fig, ax = plt.subplots(figsize=(12, 6))

for _, row in sample_df.iterrows():
    ax.plot(
        ['Pre', 'Post'],
        [row[metric_info['pre']], row[metric_info['post']]],
        marker='o',
        linestyle='-',
        label=f"{row['PATIENT_NAME']} ({row['GROUP_NAME']})",
        alpha=0.7
    )

plt.title(f'Individual {metric_info["title"]} Changes')
plt.ylabel(f'{metric_info["title"]} ({metric_info["unit"]})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
st.pyplot(fig)
