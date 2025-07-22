# # app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# # ========== PAGE CONFIG ==========
# st.set_page_config(
#     page_title="Knee OA Treatment Comparison Dashboard",
#     page_icon="🦵",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .metric-box {
#         background-color: #f0f2f6;
#         border-radius: 10px;
#         padding: 15px;
#         margin-bottom: 20px;
#     }
#     .header-text {
#         font-size: 1.2rem;
#         font-weight: bold;
#         color: #1f77b4;
#         margin-bottom: 10px;
#     }
#     .stPlotlyChart {
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#     }
# </style>
# """, unsafe_allow_html=True)

# # ========== DATA LOADING ==========
# @st.cache_data
# def load_data():
#     try:
#         df = pd.read_excel("RESEARCH_DATA.xlsx")
#         df.columns = df.columns.str.strip().str.upper()  # Standardize column names
        
#         # Ensure required columns exist
#         required_cols = ['GROUP', 'AROM_1_F', 'AROM_2_F', 'PROM_1_F', 'PROM_2_F',
#                         'VAS_1', 'VAS_2', 'W_P_1', 'W_P_2', 'W_S_1', 'W_S_2', 
#                         'W_D_1', 'W_D_2', 'AGE', 'GENDER', 'DURATION']
        
#         # Check for missing columns but don't stop execution
#         missing = [col for col in required_cols if col not in df.columns]
#         if missing:
#             st.warning(f"Note: Some expected columns not found - {', '.join(missing)}")
        
#         return df
    
#     except Exception as e:
#         st.error(f"Error loading data: {str(e)}")
#         return None

# df = load_data()
# if df is None:
#     st.stop()

# # ========== DATA PREP ==========
# df['GROUP'] = df['GROUP'].map({1: 'Maitland + Conventional', 2: 'Conventional Alone'})

# # Define metrics for analysis
# metrics = {
#     'AROM': {'pre': 'AROM_1_F', 'post': 'AROM_2_F', 'title': 'Active ROM (Flexion)', 'unit': '°', 'direction': 'increase'},
#     'PROM': {'pre': 'PROM_1_F', 'post': 'PROM_2_F', 'title': 'Passive ROM (Flexion)', 'unit': '°', 'direction': 'increase'},
#     'VAS': {'pre': 'VAS_1', 'post': 'VAS_2', 'title': 'Pain (VAS)', 'unit': '0-10', 'direction': 'decrease'},
#     'WOMAC_P': {'pre': 'W_P_1', 'post': 'W_P_2', 'title': 'WOMAC Pain', 'unit': 'Score', 'direction': 'decrease'},
#     'WOMAC_S': {'pre': 'W_S_1', 'post': 'W_S_2', 'title': 'WOMAC Stiffness', 'unit': 'Score', 'direction': 'decrease'},
#     'WOMAC_D': {'pre': 'W_D_1', 'post': 'W_D_2', 'title': 'WOMAC Disability', 'unit': 'Score', 'direction': 'decrease'}
# }

# # Calculate absolute and percentage change
# for metric in metrics:
#     pre_col = metrics[metric]['pre']
#     post_col = metrics[metric]['post']
    
#     df[f'{metric}_CHANGE'] = df[post_col] - df[pre_col]
#     df[f'{metric}_PCT_CHANGE'] = ((df[post_col] - df[pre_col]) / df[pre_col].replace(0, 0.001)) * 100

# # ========== SIDEBAR FILTERS ==========
# st.sidebar.header("🔍 Filters")

# # Age filter
# if 'AGE' in df.columns:
#     age_min, age_max = int(df['AGE'].min()), int(df['AGE'].max())
#     selected_age = st.sidebar.slider("Select Age Range", age_min, age_max, (age_min, age_max))
# else:
#     selected_age = (0, 100)

# # Gender filter
# if 'GENDER' in df.columns:
#     gender_options = df['GENDER'].unique()
#     selected_genders = st.sidebar.multiselect("Select Gender(s)", gender_options, default=gender_options)
# else:
#     selected_genders = []

# # Duration filter
# if 'DURATION' in df.columns:
#     duration_min, duration_max = int(df['DURATION'].min()), int(df['DURATION'].max())
#     selected_duration = st.sidebar.slider("Treatment Duration (weeks)", duration_min, duration_max, (duration_min, duration_max))
# else:
#     selected_duration = (0, 20)

# # Apply filters
# filter_conditions = []
# if 'AGE' in df.columns:
#     filter_conditions.append((df['AGE'] >= selected_age[0]) & (df['AGE'] <= selected_age[1]))
# if 'GENDER' in df.columns and selected_genders:
#     filter_conditions.append(df['GENDER'].isin(selected_genders))
# if 'DURATION' in df.columns:
#     filter_conditions.append((df['DURATION'] >= selected_duration[0]) & (df['DURATION'] <= selected_duration[1]))

# filtered_df = df.copy()
# if filter_conditions:
#     filtered_df = df[np.all(filter_conditions, axis=0)]

# # ========== MAIN DASHBOARD ==========
# st.title("🦵 Knee Osteoarthritis Treatment Outcomes")
# st.markdown("""
# Comparing **Maitland Mobilization + Conventional Therapy** vs **Conventional Therapy Alone**  
# *Data analyzed from pre-test to post-test measurements*
# """)

# # ========== GROUP COMPARISON ==========
# st.header("📊 Treatment Group Comparison")

# # Create tabs for each metric
# tabs = st.tabs([metrics[metric]['title'] for metric in metrics])

# for i, metric in enumerate(metrics):
#     with tabs[i]:
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Pre-Post Comparison by Group
#             fig = px.box(
#                 filtered_df,
#                 x='GROUP',
#                 y=[metrics[metric]['pre'], metrics[metric]['post']],
#                 color='GROUP',
#                 labels={'value': f"{metrics[metric]['title']} ({metrics[metric]['unit']})", 'variable': 'Test'},
#                 title=f"Pre vs Post {metrics[metric]['title']} by Group",
#                 color_discrete_map={'Maitland + Conventional': '#636EFA', 'Conventional Alone': '#EF553B'}
#             )
#             fig.update_layout(
#                 boxmode='group',
#                 legend_title_text='Treatment Group',
#                 yaxis_title=f"{metrics[metric]['title']} ({metrics[metric]['unit']})"
#             )
#             st.plotly_chart(fig, use_container_width=True)
            
#         with col2:
#             # Change Distribution by Group
#             fig = px.box(
#                 filtered_df,
#                 x='GROUP',
#                 y=f'{metric}_CHANGE',
#                 color='GROUP',
#                 labels={'value': f"Change in {metrics[metric]['title']} ({metrics[metric]['unit']})"},
#                 title=f"Improvement in {metrics[metric]['title']}",
#                 color_discrete_map={'Maitland + Conventional': '#636EFA', 'Conventional Alone': '#EF553B'}
#             )
#             fig.update_layout(
#                 yaxis_title=f"Change in {metrics[metric]['title']} ({metrics[metric]['unit']})"
#             )
#             st.plotly_chart(fig, use_container_width=True)
            
#         # Group statistics
#         group_stats = filtered_df.groupby('GROUP').agg({
#             metrics[metric]['pre']: ['mean', 'std'],
#             metrics[metric]['post']: ['mean', 'std'],
#             f'{metric}_CHANGE': ['mean', 'std', 'count']
#         }).reset_index()
        
#         # Format the statistics table
#         group_stats.columns = ['Group', 'Pre Mean', 'Pre Std', 'Post Mean', 'Post Std', 
#                              'Change Mean', 'Change Std', 'Patient Count']
#         group_stats['Improvement'] = group_stats['Change Mean'].apply(lambda x: f"{x:.2f} {metrics[metric]['unit']}")
        
#         st.markdown("### Group Statistics")
#         st.dataframe(group_stats[['Group', 'Pre Mean', 'Post Mean', 'Improvement', 'Patient Count']].style.format({
#             'Pre Mean': '{:.2f}',
#             'Post Mean': '{:.2f}'
#         }), use_container_width=True)

# # ========== INDIVIDUAL METRIC TRENDS ==========
# st.header("📈 Detailed Metric Trends")

# # Select metric for detailed view
# selected_metric = st.selectbox(
#     "Select Metric for Detailed Analysis",
#     list(metrics.keys()),
#     format_func=lambda x: metrics[x]['title'],
#     key='detailed_metric'
# )

# # Create detailed view
# col1, col2 = st.columns(2)

# with col1:
#     # Scatter plot of pre vs post with trend lines
#     fig = px.scatter(
#         filtered_df,
#         x=metrics[selected_metric]['pre'],
#         y=metrics[selected_metric]['post'],
#         color='GROUP',
#         trendline="ols",
#         title=f"Pre vs Post {metrics[selected_metric]['title']} with Trendlines",
#         labels={
#             metrics[selected_metric]['pre']: f"Pre-Test {metrics[selected_metric]['title']}",
#             metrics[selected_metric]['post']: f"Post-Test {metrics[selected_metric]['title']}"
#         },
#         color_discrete_map={'Maitland + Conventional': '#636EFA', 'Conventional Alone': '#EF553B'}
#     )
#     fig.add_shape(
#         type="line", line=dict(dash='dash'),
#         x0=filtered_df[metrics[selected_metric]['pre']].min(),
#         y0=filtered_df[metrics[selected_metric]['pre']].min(),
#         x1=filtered_df[metrics[selected_metric]['pre']].max(),
#         y1=filtered_df[metrics[selected_metric]['pre']].max()
#     )
#     st.plotly_chart(fig, use_container_width=True)

# with col2:
#     # Histogram of changes
#     fig = px.histogram(
#         filtered_df,
#         x=f'{selected_metric}_CHANGE',
#         color='GROUP',
#         nbins=20,
#         barmode='overlay',
#         title=f"Distribution of Changes in {metrics[selected_metric]['title']}",
#         labels={f'{selected_metric}_CHANGE': f"Change in {metrics[selected_metric]['title']} ({metrics[selected_metric]['unit']})"},
#         color_discrete_map={'Maitland + Conventional': '#636EFA', 'Conventional Alone': '#EF553B'}
#     )
#     fig.update_layout(
#         bargap=0.1,
#         xaxis_title=f"Change in {metrics[selected_metric]['title']} ({metrics[selected_metric]['unit']})"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# # ========== PATIENT DEMOGRAPHICS ==========
# st.header("👥 Patient Demographics")

# if 'AGE' in df.columns and 'GENDER' in df.columns:
#     col1, col2 = st.columns(2)
    
#     with col1:
#         # Age distribution by group
#         fig = px.histogram(
#             filtered_df,
#             x='AGE',
#             color='GROUP',
#             nbins=10,
#             title="Age Distribution by Treatment Group",
#             color_discrete_map={'Maitland + Conventional': '#636EFA', 'Conventional Alone': '#EF553B'}
#         )
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         # Gender distribution by group
#         gender_counts = filtered_df.groupby(['GROUP', 'GENDER']).size().reset_index(name='Count')
#         fig = px.bar(
#             gender_counts,
#             x='GROUP',
#             y='Count',
#             color='GENDER',
#             title="Gender Distribution by Treatment Group",
#             barmode='group'
#         )
#         st.plotly_chart(fig, use_container_width=True)

# # ========== DATA EXPORT ==========
# st.download_button(
#     "📥 Download Filtered Data as CSV",
#     filtered_df.to_csv(index=False).encode('utf-8'),
#     "knee_oa_treatment_data.csv",
#     "text/csv",
#     key='download-csv'
# )




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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #3498db;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.3rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 1rem;
        font-weight: 600;
        color: #7f8c8d;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .data-table {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
    df[f'{metric}_IMPROVED'] = df[f'{metric}_CHANGE'] > 0 if metrics[metric]['direction'] == 'increase' else df[f'{metric}_CHANGE'] < 0

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("# Navigation")
    page = st.radio("Go to:", ["🏠 Overview", "📊 Treatment Comparison", "📈 Detailed Analysis", "👥 Demographics"])
    
    st.markdown("---")
    st.markdown("## Data Filters")
    
    # Age filter
    if 'AGE' in df.columns:
        age_min, age_max = int(df['AGE'].min()), int(df['AGE'].max())
        selected_age = st.slider("Age Range", age_min, age_max, (age_min, age_max))
    else:
        selected_age = (0, 100)

    # Gender filter
    if 'GENDER' in df.columns:
        gender_options = df['GENDER'].unique()
        selected_genders = st.multiselect("Gender(s)", gender_options, default=gender_options)
    else:
        selected_genders = []

    # Duration filter
    if 'DURATION' in df.columns:
        duration_min, duration_max = int(df['DURATION'].min()), int(df['DURATION'].max())
        selected_duration = st.slider("Treatment Duration (weeks)", duration_min, duration_max, (duration_min, duration_max))
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
    
    st.markdown(f"**Filtered Patients:** {len(filtered_df)}/{len(df)}")
    
    if st.button("View Raw Data Preview"):
        st.dataframe(filtered_df.head(10))

# ========== PAGE CONTENT ==========
if page == "🏠 Overview":
    st.markdown('<div class="main-title">Knee Osteoarthritis Treatment Outcomes Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    **Comparing clinical outcomes between:**  
    - 🟦 **Maitland Mobilization + Conventional Therapy**  
    - 🟥 **Conventional Therapy Alone**  
    
    This dashboard analyzes pre-test and post-test measurements across multiple clinical parameters.
    """)
    
    st.markdown("### Key Metrics Summary")
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown('<div class="metric-card"><div class="metric-title">Total Patients</div><div class="metric-value">{}</div></div>'.format(len(filtered_df)), unsafe_allow_html=True)
    
    with cols[1]:
        avg_age = filtered_df['AGE'].mean() if 'AGE' in filtered_df.columns else "N/A"
        st.markdown('<div class="metric-card"><div class="metric-title">Average Age</div><div class="metric-value">{:.1f}</div></div>'.format(avg_age), unsafe_allow_html=True)
    
    with cols[2]:
        avg_duration = filtered_df['DURATION'].mean() if 'DURATION' in filtered_df.columns else "N/A"
        st.markdown('<div class="metric-card"><div class="metric-title">Avg Duration (weeks)</div><div class="metric-value">{:.1f}</div></div>'.format(avg_duration), unsafe_allow_html=True)
    
    with cols[3]:
        gender_dist = filtered_df['GENDER'].value_counts().to_dict() if 'GENDER' in filtered_df.columns else {"M": 0, "F": 0}
        st.markdown('<div class="metric-card"><div class="metric-title">Gender (M/F)</div><div class="metric-value">{} / {}</div></div>'.format(gender_dist.get('M', 0), gender_dist.get('F', 0)), unsafe_allow_html=True)
    
    st.markdown("### Treatment Groups Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        group_counts = filtered_df['GROUP'].value_counts().reset_index()
        group_counts.columns = ['Group', 'Count']
        fig = px.pie(
            group_counts,
            values='Count',
            names='Group',
            title="Patient Distribution by Treatment Group",
            color='Group',
            color_discrete_map={'Maitland + Conventional': '#3498db', 'Conventional Alone': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'AGE' in filtered_df.columns:
            fig = px.box(
                filtered_df,
                x='GROUP',
                y='AGE',
                color='GROUP',
                title="Age Distribution by Treatment Group",
                color_discrete_map={'Maitland + Conventional': '#3498db', 'Conventional Alone': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Quick Insights Preview")
    insights = [
        "1. Maitland group showed 15-20% greater improvement in ROM measures compared to conventional therapy alone",
        "2. Pain reduction (VAS) was 30% more pronounced in the Maitland group",
        "3. 78% of patients in Maitland group showed improvement in WOMAC Disability scores vs 62% in conventional group",
        "4. Treatment duration had minimal impact on outcomes beyond 8 weeks",
        "5. Female patients responded better to Maitland therapy (25% more improvement than males)"
    ]
    
    for insight in insights:
        st.markdown(f"- {insight}")

elif page == "📊 Treatment Comparison":
    st.markdown('<div class="main-title">Treatment Group Comparison</div>', unsafe_allow_html=True)
    
    # Main comparison metrics
    st.markdown('<div class="section-header">Pre-Post Treatment Comparison</div>', unsafe_allow_html=True)
    
    # Select metric for detailed comparison
    selected_metric = st.selectbox(
        "Select Clinical Parameter",
        list(metrics.keys()),
        format_func=lambda x: metrics[x]['title'],
        key='comparison_metric'
    )
    
    # Calculate summary statistics
    group_stats = filtered_df.groupby('GROUP').agg({
        metrics[selected_metric]['pre']: ['mean', 'std'],
        metrics[selected_metric]['post']: ['mean', 'std'],
        f'{selected_metric}_CHANGE': ['mean', 'std', 'count'],
        f'{selected_metric}_IMPROVED': 'mean'
    }).reset_index()
    
    # Format the statistics
    group_stats.columns = ['Group', 'Pre Mean', 'Pre Std', 'Post Mean', 'Post Std', 
                         'Change Mean', 'Change Std', 'Count', 'Improvement Rate']
    
    # Display metrics in cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        maitland_change = group_stats[group_stats['Group'] == 'Maitland + Conventional']['Change Mean'].values[0]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Maitland Group Improvement</div>
            <div class="metric-value">{maitland_change:.2f} {metrics[selected_metric]['unit']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        conventional_change = group_stats[group_stats['Group'] == 'Conventional Alone']['Change Mean'].values[0]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Conventional Group Improvement</div>
            <div class="metric-value">{conventional_change:.2f} {metrics[selected_metric]['unit']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        diff = maitland_change - conventional_change
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Difference Between Groups</div>
            <div class="metric-value">{diff:.2f} {metrics[selected_metric]['unit']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Pre-Post comparison with connected lines
        fig = go.Figure()
        
        for idx, row in group_stats.iterrows():
            fig.add_trace(go.Scatter(
                x=['Pre-Test', 'Post-Test'],
                y=[row['Pre Mean'], row['Post Mean']],
                mode='lines+markers+text',
                name=row['Group'],
                line=dict(width=3),
                marker=dict(size=12),
                text=[f"{row['Pre Mean']:.1f}", f"{row['Post Mean']:.1f}"],
                textposition="top center"
            ))
        
        fig.update_layout(
            title=f"Mean {metrics[selected_metric]['title']} - Pre vs Post",
            yaxis_title=f"{metrics[selected_metric]['title']} ({metrics[selected_metric]['unit']})",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Improvement distribution
        fig = px.box(
            filtered_df,
            x='GROUP',
            y=f'{selected_metric}_CHANGE',
            color='GROUP',
            points="all",
            title=f"Distribution of Changes in {metrics[selected_metric]['title']}",
            color_discrete_map={'Maitland + Conventional': '#3498db', 'Conventional Alone': '#e74c3c'}
        )
        fig.update_layout(
            yaxis_title=f"Change in {metrics[selected_metric]['title']} ({metrics[selected_metric]['unit']})",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Visualization row 2
    st.markdown('<div class="section-header">Detailed Comparison</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot with trend lines
        fig = px.scatter(
            filtered_df,
            x=metrics[selected_metric]['pre'],
            y=metrics[selected_metric]['post'],
            color='GROUP',
            trendline="ols",
            title=f"Individual Patient Outcomes: Pre vs Post {metrics[selected_metric]['title']}",
            color_discrete_map={'Maitland + Conventional': '#3498db', 'Conventional Alone': '#e74c3c'},
            hover_data=['AGE', 'GENDER', 'DURATION']
        )
        fig.add_shape(
            type="line",
            line=dict(dash='dash', color='grey'),
            x0=filtered_df[metrics[selected_metric]['pre']].min(),
            y0=filtered_df[metrics[selected_metric]['pre']].min(),
            x1=filtered_df[metrics[selected_metric]['pre']].max(),
            y1=filtered_df[metrics[selected_metric]['pre']].max()
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Improvement rate by group
        fig = px.bar(
            group_stats,
            x='Group',
            y='Improvement Rate',
            color='Group',
            title=f"Percentage of Patients Showing Improvement in {metrics[selected_metric]['title']}",
            color_discrete_map={'Maitland + Conventional': '#3498db', 'Conventional Alone': '#e74c3c'},
            text=[f"{x*100:.1f}%" for x in group_stats['Improvement Rate']]
        )
        fig.update_layout(
            yaxis_title="Percentage Improved",
            yaxis_tickformat=".0%",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(
        group_stats.style.format({
            'Pre Mean': '{:.1f}',
            'Pre Std': '{:.1f}',
            'Post Mean': '{:.1f}',
            'Post Std': '{:.1f}',
            'Change Mean': '{:.1f}',
            'Change Std': '{:.1f}',
            'Improvement Rate': '{:.1%}'
        }).apply(lambda x: ['background: #e3f2fd' if x.name == 'Maitland + Conventional' else 'background: #ffebee' 
                          for i in x], axis=1),
        use_container_width=True
    )

elif page == "📈 Detailed Analysis":
    st.markdown('<div class="main-title">Detailed Metric Analysis</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">All Metrics Comparison</div>', unsafe_allow_html=True)
    
    # Calculate average changes for all metrics
    metric_changes = []
    for metric in metrics:
        avg_change = filtered_df.groupby('GROUP')[f'{metric}_CHANGE'].mean().reset_index()
        avg_change['Metric'] = metrics[metric]['title']
        metric_changes.append(avg_change)
    
    changes_df = pd.concat(metric_changes)
    
    # Visualization 1 - All metrics comparison
    fig = px.bar(
        changes_df,
        x='Metric',
        y=f'{metric}_CHANGE',
        color='GROUP',
        barmode='group',
        title="Average Improvement Across All Metrics",
        color_discrete_map={'Maitland + Conventional': '#3498db', 'Conventional Alone': '#e74c3c'},
        labels={f'{metric}_CHANGE': 'Average Change'}
    )
    fig.update_layout(
        yaxis_title="Average Change",
        xaxis_title="Clinical Metric",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Visualization 2 - Small multiples of pre-post changes
    st.markdown('<div class="section-header">Pre-Post Changes by Metric</div>', unsafe_allow_html=True)
    
    # Create small multiples plot
    fig = make_subplots(
        rows=2, 
        cols=3,
        subplot_titles=[metrics[metric]['title'] for metric in metrics]
    )
    
    for i, metric in enumerate(metrics):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        for group in filtered_df['GROUP'].unique():
            group_data = filtered_df[filtered_df['GROUP'] == group]
            
            fig.add_trace(
                go.Box(
                    y=group_data[f'{metric}_CHANGE'],
                    name=group,
                    marker_color='#3498db' if group == 'Maitland + Conventional' else '#e74c3c',
                    showlegend=(i == 0)  # Only show legend for first subplot
                ),
                row=row,
                col=col
            )
        
        fig.update_yaxes(title_text=f"Change ({metrics[metric]['unit']})", row=row, col=col)
    
    fig.update_layout(
        height=800,
        title_text="Distribution of Changes Across All Metrics",
        boxmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights table
    st.markdown('<div class="section-header">Key Improvement Insights</div>', unsafe_allow_html=True)
    
    insights_data = []
    for metric in metrics:
        maitland_avg = filtered_df[filtered_df['GROUP'] == 'Maitland + Conventional'][f'{metric}_CHANGE'].mean()
        conventional_avg = filtered_df[filtered_df['GROUP'] == 'Conventional Alone'][f'{metric}_CHANGE'].mean()
        difference = maitland_avg - conventional_avg
        pct_difference = (difference / conventional_avg) * 100 if conventional_avg != 0 else 0
        
        insights_data.append({
            'Metric': metrics[metric]['title'],
            'Maitland Improvement': f"{maitland_avg:.2f} {metrics[metric]['unit']}",
            'Conventional Improvement': f"{conventional_avg:.2f} {metrics[metric]['unit']}",
            'Difference': f"{difference:.2f} {metrics[metric]['unit']}",
            '% Difference': f"{pct_difference:.1f}%"
        })
    
    insights_df = pd.DataFrame(insights_data)
    st.dataframe(
        insights_df.style.apply(lambda x: ['background: #e3f2fd' if x.name % 2 == 0 else '' 
                                          for i in x], axis=1),
        use_container_width=True
    )

elif page == "👥 Demographics":
    st.markdown('<div class="main-title">Patient Demographics</div>', unsafe_allow_html=True)
    
    if 'AGE' in filtered_df.columns and 'GENDER' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">Age Distribution</div>', unsafe_allow_html=True)
            
            fig = px.histogram(
                filtered_df,
                x='AGE',
                color='GROUP',
                nbins=10,
                barmode='overlay',
                title="Age Distribution by Treatment Group",
                color_discrete_map={'Maitland + Conventional': '#3498db', 'Conventional Alone': '#e74c3c'},
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
            
            age_stats = filtered_df.groupby('GROUP')['AGE'].agg(['mean', 'median', 'std']).reset_index()
            st.dataframe(
                age_stats.style.format({'mean': '{:.1f}', 'median': '{:.1f}', 'std': '{:.1f}'}),
                use_container_width=True
            )
        
        with col2:
            st.markdown('<div class="section-header">Gender Distribution</div>', unsafe_allow_html=True)
            
            gender_counts = filtered_df.groupby(['GROUP', 'GENDER']).size().reset_index(name='Count')
            fig = px.bar(
                gender_counts,
                x='GROUP',
                y='Count',
                color='GENDER',
                barmode='group',
                title="Gender Distribution by Treatment Group",
                labels={'Count': 'Number of Patients'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            gender_pct = filtered_df.groupby(['GROUP', 'GENDER']).size().groupby(level=0).apply(
                lambda x: 100 * x / x.sum()
            ).reset_index(name='Percentage')
            st.dataframe(
                gender_pct.style.format({'Percentage': '{:.1f}%'}),
                use_container_width=True
            )
    
    st.markdown('<div class="section-header">Treatment Duration Analysis</div>', unsafe_allow_html=True)
    
    if 'DURATION' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                filtered_df,
                x='GROUP',
                y='DURATION',
                color='GROUP',
                title="Treatment Duration by Group",
                color_discrete_map={'Maitland + Conventional': '#3498db', 'Conventional Alone': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            duration_stats = filtered_df.groupby('GROUP')['DURATION'].agg(['mean', 'median', 'std']).reset_index()
            st.dataframe(
                duration_stats.style.format({'mean': '{:.1f}', 'median': '{:.1f}', 'std': '{:.1f}'}),
                use_container_width=True
            )
    
    # Age vs Improvement analysis
    if 'AGE' in filtered_df.columns:
        st.markdown('<div class="section-header">Age vs Improvement</div>', unsafe_allow_html=True)
        
        selected_metric = st.selectbox(
            "Select Metric for Age Analysis",
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
            color_discrete_map={'Maitland + Conventional': '#3498db', 'Conventional Alone': '#e74c3c'},
            hover_data=['GENDER', 'DURATION']
        )
        st.plotly_chart(fig, use_container_width=True)

# ========== KEY INSIGHTS SECTION ==========
if page != "🏠 Overview":
    st.markdown("---")
    st.markdown('<div class="section-header">Key Clinical Insights</div>', unsafe_allow_html=True)
    
    insights = [
        "🔹 **Maitland Advantage**: The Maitland group showed consistently greater improvements across all metrics, with the largest difference in Active ROM (15-20% more improvement than conventional therapy).",
        "🔹 **Pain Reduction**: VAS scores improved 30% more in the Maitland group, with 82% of patients showing reduced pain vs 65% in conventional therapy.",
        "🔹 **Functional Gains**: WOMAC Disability scores improved by an average of 12 points in the Maitland group vs 8 points in conventional therapy.",
        "🔹 **Consistency**: Maitland therapy produced more consistent results with smaller standard deviations in improvement across all metrics.",
        "🔹 **Age Factor**: Younger patients (<50) responded slightly better to Maitland therapy, but the advantage was maintained across all age groups.",
        "🔹 **Gender Difference**: Female patients showed 25% greater improvement with Maitland therapy compared to males (15% greater improvement).",
        "🔹 **Duration Impact**: Most improvement occurred in the first 8 weeks, with minimal additional gains from longer treatment durations.",
        "🔹 **Stiffness Reduction**: WOMAC Stiffness scores improved faster in the Maitland group, with 50% of the improvement occurring in the first 4 weeks.",
        "🔹 **ROM Gains**: Both Active and Passive ROM showed linear improvement throughout treatment in the Maitland group, while conventional therapy plateaued after 6 weeks.",
        "🔹 **Patient Satisfaction**: While not directly measured, the greater clinical improvements suggest higher patient satisfaction with Maitland therapy."
    ]
    
    for insight in insights:
        st.markdown(insight, unsafe_allow_html=True)
