import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. DATA GENERATION (CSV reading helpers) ---

# Cached helper that only reads local file paths. NOTE: this function does
# NOT include any Streamlit widgets so it is safe to cache.
@st.cache_data
def _read_local_csvs(voters_path: str, history_path: str):
    df_voters = pd.read_csv(voters_path)
    df_history_raw = pd.read_csv(history_path)

    # Normalize column name if present
    if 'Ward_No' in df_history_raw.columns and 'ward_no' not in df_history_raw.columns:
        df_history_raw = df_history_raw.rename(columns={'Ward_No': 'ward_no'})

    return df_voters, df_history_raw


def load_data():
    """
    Load voter and history CSVs. Widgets (file uploaders) are executed
    at top-level here (not inside a cached function) to avoid
    Streamlit CachedWidgetWarning when deployed.

    Behavior:
    - Try to read the local absolute paths using the cached helper.
    - If local files are not available, show `st.file_uploader` widgets
      (outside cache) and read uploaded files directly (no caching).
    """
    # Prefer local relative files (place CSVs next to this script). If not found,
    # show Streamlit upload widgets so the user can upload files through the app.
    local_voters_abs = r"C:\Users\LENOVO\Desktop\deploy\voters_filtereddata - Sheet1 (1).csv"
    local_history_abs = r"C:\Users\LENOVO\Desktop\deploy\voters_data - Sheet2.csv"

    try:
        # Use the cached reader for local files only
        df_voters, df_history_raw = _read_local_csvs(local_voters_abs, local_history_abs)
    except Exception:
        # Local files not available — perform uploads at top-level (not cached)
        st.warning('Local CSV files not found. Please upload them below or place them next to this script.')
        uploaded_voters = st.file_uploader('Upload `voters_filtereddata - Sheet1 (1).csv`', type=['csv'], key='voters')
        uploaded_history = st.file_uploader('Upload `voters_data - Sheet2.csv`', type=['csv'], key='history')

        if uploaded_voters is None or uploaded_history is None:
            # Halt execution until files are provided
            st.stop()

        df_voters = pd.read_csv(uploaded_voters)
        df_history_raw = pd.read_csv(uploaded_history)

        # Normalize column name if present
        if 'Ward_No' in df_history_raw.columns and 'ward_no' not in df_history_raw.columns:
            df_history_raw = df_history_raw.rename(columns={'Ward_No': 'ward_no'})

    return df_voters, df_history_raw

df_voters, df_history_raw = load_data()

# --- Basic validation: ensure expected columns exist to avoid runtime errors ---
missing_voter_cols = {'ward_no', 'voter_age', 'Gender'} - set(df_voters.columns)
missing_history_cols = {'ward_no', 'total_votes_per_person', 'name', 'party', 'total_votes_cast'} - set(df_history_raw.columns)

if missing_voter_cols:
    st.error(f"Missing columns in voters CSV: {list(missing_voter_cols)}. Please check the file.")
    st.stop()

if missing_history_cols:
    st.error(f"Missing columns in history CSV: {list(missing_history_cols)}. Please check the file.")
    st.stop()

# --- 2. DATA TRANSFORMATION (The "Senior Analyst" Logic) ---

def process_history(df):
    """
    Turns Raw Candidate Data into Ward Summary (Winner/Runner-up)
    """
    ward_summaries = []

    # Group by Ward to find Winner and Runner-up
    for ward in df['ward_no'].unique():
        ward_data = df[df['ward_no'] == ward]

        # Sort candidates by votes (Highest first)
        sorted_candidates = ward_data.sort_values(by='total_votes_per_person', ascending=False)

        # Get Winner (Rank 1) and Runner Up (Rank 2)
        winner = sorted_candidates.iloc[0]
        runner_up = sorted_candidates.iloc[1] if len(sorted_candidates) > 1 else None

        summary = {
            'ward_no': ward,
            'Winner_Name': winner['name'],
            'Winner_Party': winner['party'],
            'Winner_Votes': winner['total_votes_per_person'],
            'RunnerUp_Name': runner_up['name'] if runner_up is not None else "N/A",
            'RunnerUp_Party': runner_up['party'] if runner_up is not None else "N/A",
            'RunnerUp_Votes': runner_up['total_votes_per_person'] if runner_up is not None else 0,
            'Total_Votes_Cast': winner['total_votes_cast']
        }

        # Calculate Margin
        summary['Win_Margin'] = summary['Winner_Votes'] - summary['RunnerUp_Votes']
        ward_summaries.append(summary)

    return pd.DataFrame(ward_summaries)

# Apply the transformation
df_history_clean = process_history(df_history_raw)

# --- 3. FILTER LOGIC (Age Bins) ---
def create_age_bins(age):
    if 18 <= age <= 28: return '18-28'
    elif 29 <= age <= 39: return '29-39'
    elif 40 <= age <= 59: return '40-59'
    else: return '60+'

df_voters['Age_Group'] = df_voters['voter_age'].apply(create_age_bins)

# --- 4. DASHBOARD UI ---
st.set_page_config(layout="wide")
st.title("🗳️ Election Strategy Dashboard: Historical vs Current")

# Sidebar
st.sidebar.header("Target Filters")
selected_age = st.sidebar.multiselect("Age Group", ['18-28', '29-39', '40-59', '60+'], default=['18-28'])
selected_gender = st.sidebar.multiselect("Gender", df_voters['Gender'].unique(), default=df_voters['Gender'].unique())
# Ward filter: allow user to select specific ward numbers (default = all)
ward_options = sorted(df_voters['ward_no'].unique()) if 'ward_no' in df_voters.columns else []
selected_wards = st.sidebar.multiselect("Ward Number", ward_options, default=ward_options)
# Slider to choose how many top wards to display in the bar chart
max_wards = min(50, len(ward_options)) if ward_options else 10
top_n = st.sidebar.slider("Top N wards to show (bar chart)", min_value=1, max_value=max_wards, value=min(10, max_wards))
# Party filter (based on historical winners)
party_options = sorted(df_history_clean['Winner_Party'].dropna().unique()) if 'Winner_Party' in df_history_clean.columns else []
default_party = ['con.'] if 'con.' in [p.lower() for p in party_options] else party_options
# Normalize case: keep display values as-is but ensure default matches case-insensitively
selected_parties = st.sidebar.multiselect("Winner Party", party_options, default=default_party)


# Filter Data
filtered_voters = df_voters[
    (df_voters['Age_Group'].isin(selected_age)) &
    (df_voters['Gender'].isin(selected_gender))
]
# Apply ward selection if provided
if selected_wards:
    filtered_voters = filtered_voters[filtered_voters['ward_no'].isin(selected_wards)]

# Aggregate filtered voters by Ward
voter_counts = filtered_voters.groupby('ward_no').size().reset_index(name='Voters_In_Ward')

# Show totals and per-ward counts (respecting current filters)
st.markdown("---")
st.subheader("Voter Counts by Ward")
# Total filtered voters
st.write(f"Total voters matching filters: {len(filtered_voters)}")

# If exactly one ward is selected, show a highlighted info box with its count
if selected_wards and len(selected_wards) == 1:
    ward_val = selected_wards[0]
    ward_row = voter_counts[voter_counts['ward_no'] == ward_val]
    ward_count = int(ward_row['Voters_In_Ward'].squeeze()) if not ward_row.empty else 0
    st.info(f"Selected Ward {ward_val}: {ward_count} voters (after applied filters)")

# Show table: selected wards if any, otherwise top 10 wards by count
if selected_wards:
    display_counts = voter_counts[voter_counts['ward_no'].isin(selected_wards)].sort_values('Voters_In_Ward', ascending=False)
else:
    display_counts = voter_counts.sort_values('Voters_In_Ward', ascending=False).head(10)

st.dataframe(display_counts.reset_index(drop=True), use_container_width=True)

# --- VOTERS SUMMARY BAR CHART (df_voters details) ---
# Single bar chart showing counts by Gender, stacked by Age_Group.
st.markdown("---")
st.subheader("Voters Detail: Gender × Age Group")

# Prepare counts from the filtered voters (respects age, gender, ward filters)
if filtered_voters.empty:
    st.info("No voters match the selected filters — adjust Age/Gender/Ward filters to see the chart.")
else:
    # Group by Gender and Age_Group and compute counts
    counts = (
        filtered_voters
        .groupby(['Gender', 'Age_Group'], dropna=False)
        .size()
        .reset_index(name='Count')
    )

    # If Age_Group has NaNs, fill with 'Unknown'
    counts['Age_Group'] = counts['Age_Group'].fillna('Unknown')

    # Create a stacked bar chart: x=Gender, y=Count, color=Age_Group
    voters_bar = px.bar(
        counts,
        x='Gender',
        y='Count',
        color='Age_Group',
        title=f"Voter Counts by Gender (filtered by Age/Ward)",
        labels={'Count': 'Number of Voters', 'Gender': 'Gender', 'Age_Group': 'Age Group'},
        category_orders={'Age_Group': ['18-28', '29-39', '40-59', '60+']}
    )
    voters_bar.update_layout(barmode='stack')
    voters_bar.update_traces(textposition='outside', text=counts.groupby('Gender')['Count'].sum().values)
    voters_bar.update_xaxes(type='category')
    st.plotly_chart(voters_bar, use_container_width=True)




# MERGE History (Clean) + Current Voters
final_df = pd.merge(df_history_clean, voter_counts, on='ward_no', how='left')
final_df['Voters_In_Ward'] = final_df['Voters_In_Ward'].fillna(0)

# Calculate Impact Factor: (Target Voters / Previous Margin)
# If this is > 1.0, the target group is larger than the margin of victory!
def _compute_impact(row):
    m = row.get('Win_Margin')
    t = row.get('Voters_In_Ward', 0)
    try:
        if m is None or pd.isna(m) or m == 0:
            return np.inf if t > 0 else 0.0
        return t / m
    except Exception:
        return np.nan

final_df['Impact_Score'] = final_df.apply(_compute_impact, axis=1)

# Apply party filter (historical winner) if user selected parties
if selected_parties:
    final_df = final_df[final_df['Winner_Party'].isin(selected_parties)]

# --- 5. VISUALIZATION ---

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Voter Distribution")

    # Pie chart: distribution of voters in ward (filtered) by Winner_Party
    party_pie_df = final_df.groupby('Winner_Party', dropna=False)['Voters_In_Ward'].sum().reset_index()
    party_pie_df = party_pie_df.sort_values('Voters_In_Ward', ascending=False)
    pie_title = f"Voters in Ward (filtered) by Winning Party ({', '.join(map(str, selected_age))})"
    pie_fig = px.pie(party_pie_df, values='Voters_In_Ward', names='Winner_Party', title=pie_title,
                     hover_data=['Voters_In_Ward'])
    st.plotly_chart(pie_fig, use_container_width=True)

    # Bar chart: Top-N wards by Target_Voter_Count (colored by historical winner party)
    st.markdown("---")
    st.subheader(f"Top {top_n} Wards by Voters in Ward")
    bar_df = final_df[['ward_no', 'Voters_In_Ward', 'Winner_Party']].fillna({'Voters_In_Ward': 0, 'Winner_Party': 'Unknown'})
    bar_df = bar_df.sort_values('Voters_In_Ward', ascending=False).head(top_n)
    bar_fig = px.bar(
        bar_df,
        x='ward_no',
        y='Voters_In_Ward',
        color='Winner_Party',
        title=f"Top {top_n} Wards (Voters in Ward)",
        labels={'ward_no': 'Ward Number', 'Voters_In_Ward': 'Voters in Ward'},
        hover_data=['Winner_Party', 'Voters_In_Ward']
    )
    st.plotly_chart(bar_fig, use_container_width=True)

with col2:
    st.subheader("💡 Key Insights")
    # Identify Swing Wards
    swing_wards = final_df[final_df['Voters_In_Ward'] > final_df['Win_Margin']]
    st.markdown(f"*{len(swing_wards)} Wards* have more target voters than the previous winning margin.")

    if not swing_wards.empty:
        st.warning(f"Critical Wards: {list(swing_wards['ward_no'])}")
        st.markdown("In these wards, your selected demographic alone could flip the result.")

    # Con party summary across all wards (independent of sidebar party filter)
    con_df_all = pd.merge(df_history_clean, voter_counts, on='ward_no', how='left')
    con_df_all['Voters_In_Ward'] = con_df_all['Voters_In_Ward'].fillna(0)
    con_mask = con_df_all['Winner_Party'].astype(str).str.lower().isin(['con', 'con.', 'conservative'])
    con_total_voters = int(con_df_all.loc[con_mask, 'Voters_In_Ward'].sum()) if not con_df_all[con_mask].empty else 0
    con_total_winner_votes = int(con_df_all.loc[con_mask, 'Winner_Votes'].sum()) if ('Winner_Votes' in con_df_all.columns and not con_df_all[con_mask].empty) else 0
    st.markdown(f"**Con party summary (all wards):** {con_total_voters} voters available; historical con winners received {con_total_winner_votes} votes total.")
    if con_total_voters > 0:
        st.markdown(f"That is {con_total_winner_votes / con_total_voters:.1%} winner-votes / available-voters (historical).")

# Data Table
st.subheader("Detailed Ward Breakdown")
# The Pandas Styler.background_gradient requires matplotlib. Detect it
# and fall back to an unstyled dataframe with a helpful message if missing.
try:
    import matplotlib  # type: ignore
    _have_mpl = True
except Exception:
    _have_mpl = False

_cols = ['ward_no', 'Winner_Party', 'Win_Margin', 'Voters_In_Ward', 'Winner_Votes', 'Winner_Name', 'RunnerUp_Name', 'RunnerUp_Votes', 'Impact_Score']
if _have_mpl:
    st.dataframe(
        final_df[_cols].style.background_gradient(subset=['Impact_Score'], cmap="Reds"),
        use_container_width=True,
    )
else:
    st.warning('`matplotlib` not installed — run `pip install matplotlib` to enable Impact_Score styling.')
    st.dataframe(final_df[_cols], use_container_width=True)
