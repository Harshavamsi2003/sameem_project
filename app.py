import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Knee OA Treatment Comparison",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== DATA LOADING ==========
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("RESEARCH_DATA.xlsx")
        df.columns = df.columns.str.strip().str.upper()  # Standardize column names
        
        # Ensure required columns exist
        required_cols = ['GROUP', 'AROM_1_F', 'AROM_2_F', 'PROM_1_F', 'PROM_2_F',
                        'VAS_1', 'VAS_2', 'W_P_1', 'W_P_2', 'W_S_1', 'W_S_2', 'W_D_1', 'W_D_2']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
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
    'AROM': {'pre': 'AROM_1_F', 'post': 'AROM_2_F', 'title': 'Active ROM (Flexion)', 'unit': '°'},
    'PROM': {'pre': 'PROM_1_F', 'post': 'PROM_2_F', 'title': 'Passive ROM (Flexion)', 'unit': '°'},
    'VAS': {'pre': 'VAS_1', 'post': 'VAS_2', 'title': 'Pain (VAS)', 'unit': '0-10'},
    'WOMAC_P': {'pre': 'W_P_1', 'post': 'W_P_2', 'title': 'WOMAC Pain', 'unit': 'Score'},
    'WOMAC_S': {'pre': 'W_S_1', 'post': 'W_S_2', 'title': 'WOMAC Stiffness', 'unit': 'Score'},
    'WOMAC_D': {'pre': 'W_D_1', 'post': 'W_D_2', 'title': 'WOMAC Disability', 'unit': 'Score'}
}

# Calculate improvement
for metric in metrics:
    pre, post = metrics[metric]['pre'], metrics[metric]['post']
    df[f'{metric}_improvement'] = ((df[post] - df[pre]) / df[pre].replace(0, 0.001)) * 100

# ========== SIDEBAR FILTERS ==========
st.sidebar.header("🔍 Filters")
selected_metric = st.sidebar.selectbox(
    "Select Metric", 
    list(metrics.keys()),
    format_func=lambda x: metrics[x]['title']
)

# Age filter (if column exists)
if 'AGE' in df.columns:
    age_min, age_max = int(df['AGE'].min()), int(df['AGE'].max())
    age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))
else:
    age_range = (0, 100)  # Default if no age data

# Gender filter (if column exists)
gender_options = df['GENDER'].unique() if 'GENDER' in df.columns else []
gender_filter = st.sidebar.multiselect("Gender", gender_options, default=gender_options)

# Duration filter
duration_min, duration_max = int(df['DURATION'].min()), int(df['DURATION'].max())
duration_range = st.sidebar.slider("Treatment Duration (weeks)", duration_min, duration_max, (duration_min, duration_max))

# Apply filters
filter_conditions = [
    (df['DURATION'] >= duration_range[0]) & (df['DURATION'] <= duration_range[1])
]

if 'AGE' in df.columns:
    filter_conditions.append((df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1]))

if 'GENDER' in df.columns and gender_filter:
    filter_conditions.append(df['GENDER'].isin(gender_filter))

filtered_df = df[np.all(filter_conditions, axis=0)] if filter_conditions else df

# ========== MAIN DASHBOARD ==========
st.title("🧬 Knee Osteoarthritis Treatment Comparison")
st.markdown("""
- **Group 1**: Maitland Mobilization + Conventional Therapy  
- **Group 2**: Conventional Therapy Alone  
""")

# Key Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", len(filtered_df))
avg_improve = filtered_df[f'{selected_metric}_improvement'].mean()
col2.metric(f"Avg Improvement ({metrics[selected_metric]['title']})", f"{avg_improve:.1f}%")

group_diff = (
    filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional'][f'{selected_metric}_improvement'].mean() -
    filtered_df[filtered_df['GROUP'] == 'Conventional Alone'][f'{selected_metric}_improvement'].mean()
)
col3.metric("Difference (Maitland vs Conventional)", f"{group_diff:.1f}%")

# ========== VISUALIZATIONS ==========
tab1, tab2, tab3 = st.tabs(["📊 Pre-Post Comparison", "📈 Improvement Analysis", "📉 Statistical Tests"])

with tab1:
    # Pre-Post Violin Plot
    fig = go.Figure()
    for group in filtered_df['GROUP'].unique():
        group_data = filtered_df[filtered_df['GROUP'] == group]
        fig.add_trace(go.Violin(
            x=['Pre-Test'] * len(group_data),
            y=group_data[metrics[selected_metric]['pre']],
            name=f'{group} (Pre)',
            box_visible=True,
            line_color='blue' if group == 'Maitland + Conventional' else 'red'
        ))
        fig.add_trace(go.Violin(
            x=['Post-Test'] * len(group_data),
            y=group_data[metrics[selected_metric]['post']],
            name=f'{group} (Post)',
            box_visible=True,
            line_color='lightblue' if group == 'Maitland + Conventional' else 'pink'
        ))
    fig.update_layout(title=f"Pre vs Post {metrics[selected_metric]['title']}", height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Improvement by Group
    fig = px.box(
        filtered_df, 
        x='GROUP', 
        y=f'{selected_metric}_improvement',
        color='GROUP',
        title=f"Improvement in {metrics[selected_metric]['title']}",
        labels={'GROUP': 'Treatment Group', f'{selected_metric}_improvement': 'Improvement (%)'},
        color_discrete_map={'Maitland + Conventional': 'blue', 'Conventional Alone': 'red'}
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Statistical Tests
    st.subheader("🔬 Hypothesis Testing")
    group1 = filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional']
    group2 = filtered_df[filtered_df['GROUP'] == 'Conventional Alone']
    
    # Normality Check
    _, p1 = stats.shapiro(group1[f'{selected_metric}_improvement'])
    _, p2 = stats.shapiro(group2[f'{selected_metric}_improvement'])
    
    if p1 > 0.05 and p2 > 0.05:
        # t-test if normal
        t_stat, p_val = stats.ttest_ind(
            group1[f'{selected_metric}_improvement'],
            group2[f'{selected_metric}_improvement']
        )
        test_used = "Independent t-test"
    else:
        # Mann-Whitney if non-normal
        u_stat, p_val = stats.mannwhitneyu(
            group1[f'{selected_metric}_improvement'],
            group2[f'{selected_metric}_improvement']
        )
        test_used = "Mann-Whitney U Test"
    
    st.write(f"**Test Used:** {test_used}")
    st.write(f"**p-value:** {p_val:.4f} {'(Significant)' if p_val < 0.05 else '(Not Significant)'}")
    
    # Effect Size
    cohen_d = (group1[f'{selected_metric}_improvement'].mean() - group2[f'{selected_metric}_improvement'].mean()) / np.sqrt(
        (group1[f'{selected_metric}_improvement'].std()**2 + group2[f'{selected_metric}_improvement'].std()**2) / 2
    )
    st.write(f"**Effect Size (Cohen's d):** {cohen_d:.2f}")

# ========== DATA EXPORT ==========
st.download_button(
    "📥 Download Filtered Data (CSV)",
    filtered_df.to_csv(index=False).encode('utf-8'),
    "knee_oa_filtered_data.csv",
    "text/csv"
)
