import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Business Customer Intelligence",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    font-size: 17px;
    color: #666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.action-card {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    border: 1px solid #e5e7eb;
    margin-bottom: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

.action-high {
    border-left: 6px solid #d9534f;
}

.action-medium {
    border-left: 6px solid #f0ad4e;
}

.action-low {
    border-left: 6px solid #5bc0de;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# BUSINESS CONFIGURATION
# ============================================================

business_config = {

    "Retail Store": {
        "file": "Data/retail_customers.csv",
        "features": [
            "Age",
            "Annual_Income",
            "Purchase_Frequency",
            "Average_Order_Value",
            "Total_Spending",
            "Recency_Days"
        ],
        "frequency_column": "Purchase_Frequency",
        "spending_column": "Total_Spending",
        "recency_column": "Recency_Days"
    },

    "E-Commerce": {
        "file": "Data/ecommerce_customers.csv",
        "features": [
            "Age",
            "Annual_Income",
            "Orders_Count",
            "Average_Order_Value",
            "Total_Spending",
            "Website_Visits",
            "Cart_Abandonment"
        ],
        "frequency_column": "Orders_Count",
        "spending_column": "Total_Spending",
        "recency_column": None
    },

    "Restaurant": {
        "file": "Data/restaurant_customers.csv",
        "features": [
            "Age",
            "Annual_Income",
            "Visit_Frequency",
            "Average_Bill",
            "Total_Spending",
            "Online_Orders",
            "Recency_Days"
        ],
        "frequency_column": "Visit_Frequency",
        "spending_column": "Total_Spending",
        "recency_column": "Recency_Days"
    }
}


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">📊 Business Customer Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'AI-powered customer segmentation, customer value analysis and business action recommendations'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Dashboard Controls")

business_type = st.sidebar.selectbox(
    "Select Business Type",
    list(business_config.keys())
)

config = business_config[business_type]

st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "Choose Data Source",
    ["Use Sample Dataset", "Upload CSV"]
)


# ============================================================
# LOAD DATA
# ============================================================

if data_source == "Use Sample Dataset":

    try:
        df = pd.read_csv(config["file"])

    except Exception as e:

        st.error(
            f"Could not load dataset: {e}"
        )

        st.stop()

else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload a CSV file from the sidebar to continue."
        )

        st.stop()

    try:

        df = pd.read_csv(uploaded_file)

    except Exception as e:

        st.error(
            f"Could not read uploaded CSV: {e}"
        )

        st.stop()


# ============================================================
# DATA VALIDATION
# ============================================================

required_columns = [
    "Customer_ID"
] + config["features"]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    st.error(
        "❌ Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.info(
        "Please upload a CSV containing the required columns."
    )

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

df = df.copy()

df = df.drop_duplicates(
    subset=["Customer_ID"]
)


for col in config["features"]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


before_rows = len(df)

df = df.dropna(
    subset=config["features"]
)


removed_rows = (
    before_rows - len(df)
)


if len(df) < 3:

    st.error(
        "Not enough valid customer records for clustering."
    )

    st.stop()


# ============================================================
# PREPARE FEATURES
# ============================================================

features = config["features"]

X = df[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# SILHOUETTE ANALYSIS
# ============================================================

max_k = min(
    6,
    len(df) - 1
)

k_values = list(
    range(2, max_k + 1)
)

silhouette_results = []


for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_results.append({
        "K": k,
        "Silhouette Score": round(
            score,
            4
        )
    })


silhouette_df = pd.DataFrame(
    silhouette_results
)


best_index = silhouette_df[
    "Silhouette Score"
].idxmax()


best_k = int(
    silhouette_df.loc[
        best_index,
        "K"
    ]
)


best_silhouette = float(
    silhouette_df.loc[
        best_index,
        "Silhouette Score"
    ]
)


# ============================================================
# FINAL K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(
    X_scaled
)


# ============================================================
# AUTOMATIC SEGMENT NAMING
# ============================================================

segment_summary = df.groupby(
    "Cluster"
).agg({

    config["spending_column"]: "mean",

    config["frequency_column"]: "mean"

}).reset_index()


spending_rank = (
    segment_summary
    .sort_values(
        config["spending_column"],
        ascending=False
    )
    ["Cluster"]
    .tolist()
)


frequency_rank = (
    segment_summary
    .sort_values(
        config["frequency_column"],
        ascending=False
    )
    ["Cluster"]
    .tolist()
)


segment_names = {}


for cluster in segment_summary["Cluster"]:

    if cluster == spending_rank[0]:

        segment_names[cluster] = (
            "High-Value Customers"
        )

    elif cluster == frequency_rank[0]:

        segment_names[cluster] = (
            "Regular / Growing Customers"
        )

    else:

        segment_names[cluster] = (
            "Low-Engagement Customers"
        )


# ============================================================
# AT-RISK IDENTIFICATION
# ============================================================

if config["recency_column"]:

    recency_threshold = df[
        config["recency_column"]
    ].median()

    high_income_threshold = df[
        "Annual_Income"
    ].median()

    at_risk_mask = (

        df[
            config["recency_column"]
        ] > recency_threshold

    ) & (

        df["Annual_Income"]
        >= high_income_threshold

    )

else:

    cart_threshold = df[
        "Cart_Abandonment"
    ].median()

    at_risk_mask = (
        df["Cart_Abandonment"]
        >= cart_threshold
    )


# ============================================================
# CUSTOMER VALUE SCORE
# ============================================================

def minmax_score(series):

    min_value = series.min()

    max_value = series.max()

    if max_value == min_value:

        return pd.Series(
            50,
            index=series.index
        )

    return (
        (series - min_value)
        /
        (max_value - min_value)
    ) * 100


spending_score = minmax_score(
    df[
        config["spending_column"]
    ]
)


frequency_score = minmax_score(
    df[
        config["frequency_column"]
    ]
)


if config["recency_column"]:

    recency_score = (
        100
        -
        minmax_score(
            df[
                config["recency_column"]
            ]
        )
    )

else:

    recency_score = minmax_score(
        df["Website_Visits"]
    )


if business_type == "E-Commerce":

    engagement_score = (
        100
        -
        minmax_score(
            df["Cart_Abandonment"]
        )
    )

elif business_type == "Restaurant":

    engagement_score = minmax_score(
        df["Online_Orders"]
    )

else:

    engagement_score = frequency_score


df["Customer_Value_Score"] = (

    spending_score * 0.40

    +

    frequency_score * 0.30

    +

    recency_score * 0.20

    +

    engagement_score * 0.10
)


def value_category(score):

    if score >= 75:

        return "Premium"

    elif score >= 50:

        return "Valuable"

    elif score >= 25:

        return "Developing"

    else:

        return "Low Value"


df["Value_Category"] = (
    df[
        "Customer_Value_Score"
    ].apply(value_category)
)


df["Segment"] = (
    df["Cluster"].map(
        segment_names
    )
)


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.metric(
    "Customers",
    len(df)
)

st.sidebar.metric(
    "Best K",
    best_k
)

st.sidebar.metric(
    "Silhouette Score",
    f"{best_silhouette:.3f}"
)


# ============================================================
# BUSINESS OVERVIEW
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📌 Business Overview</div>',
    unsafe_allow_html=True
)


total_customers = len(df)


total_spending = df[
    config["spending_column"]
].sum()


average_spending = df[
    config["spending_column"]
].mean()


premium_count = (
    df["Value_Category"]
    ==
    "Premium"
).sum()


at_risk_count = (
    at_risk_mask.sum()
)


average_value_score = df[
    "Customer_Value_Score"
].mean()


col1, col2, col3, col4, col5, col6 = st.columns(6)


with col1:

    st.metric(
        "👥 Customers",
        total_customers
    )


with col2:

    st.metric(
        "💰 Total Spending",
        f"₹{total_spending:,.0f}"
    )


with col3:

    st.metric(
        "🛒 Avg Spending",
        f"₹{average_spending:,.0f}"
    )


with col4:

    st.metric(
        "🏆 Premium",
        premium_count
    )


with col5:

    st.metric(
        "⚠️ At Risk",
        at_risk_count
    )


with col6:

    st.metric(
        "⭐ Avg Value Score",
        f"{average_value_score:.1f}"
    )


# ============================================================
# BUSINESS ACTION CENTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🎯 Business Action Center</div>',
    unsafe_allow_html=True
)

st.write(
    "Turn customer analytics into practical business decisions."
)


regular_count = (
    df["Segment"]
    ==
    "Regular / Growing Customers"
).sum()


low_engagement_count = (
    df["Segment"]
    ==
    "Low-Engagement Customers"
).sum()


action_col1, action_col2, action_col3, action_col4 = st.columns(4)


with action_col1:

    st.metric(
        "🏆 Premium Customers",
        premium_count
    )


with action_col2:

    st.metric(
        "⚠️ At-Risk Customers",
        at_risk_count
    )


with action_col3:

    st.metric(
        "📈 Growth Opportunities",
        regular_count
    )


with action_col4:

    st.metric(
        "🔄 Low Engagement",
        low_engagement_count
    )


# ============================================================
# ACTION DATASETS
# ============================================================

premium_customers = df[
    df["Value_Category"]
    ==
    "Premium"
].copy()


at_risk_customers = df[
    at_risk_mask
].copy()


growth_customers = df[
    (
        df["Value_Category"].isin(
            [
                "Valuable",
                "Developing"
            ]
        )
    )
    &
    (
        df[
            config["frequency_column"]
        ]
        >=
        df[
            config["frequency_column"]
        ].median()
    )
].copy()


low_engagement_customers = df[
    df["Value_Category"]
    ==
    "Low Value"
].copy()


# ============================================================
# PREMIUM CUSTOMER STRATEGY
# ============================================================

st.markdown("### 🏆 Premium Customer Strategy")

premium_value = premium_customers[config["spending_column"]].sum()

st.markdown(
    f'<div class="action-card action-high"><h4>🏆 Premium Customers</h4><p><b>{premium_count}</b> premium customers generate approximately <b>₹{premium_value:,.0f}</b> in total spending.</p><p><b>Recommended Action:</b> Focus on retention, loyalty rewards, exclusive offers, early access and personalized recommendations.</p></div>',
    unsafe_allow_html=True
)


# ============================================================
# AT-RISK CUSTOMER STRATEGY
# ============================================================

st.markdown("### ⚠️ At-Risk Customer Strategy")

at_risk_value = at_risk_customers[config["spending_column"]].sum()

if at_risk_count > 0:
    st.markdown(
        f'<div class="action-card action-high"><h4>⚠️ At-Risk Customers</h4><p><b>{at_risk_count}</b> customers have been identified as potential retention risks.</p><p>Their recorded spending is approximately <b>₹{at_risk_value:,.0f}</b>.</p><p><b>Recommended Action:</b> Send personalized retention offers, loyalty incentives and targeted follow-up campaigns.</p></div>',
        unsafe_allow_html=True
    )
else:
    st.success("✅ No major at-risk customers were detected.")


# ============================================================
# GROWTH OPPORTUNITY
# ============================================================

st.markdown("### 📈 Growth Opportunity Strategy")

st.markdown(
    f'<div class="action-card action-medium"><h4>📈 Growth Opportunities</h4><p><b>{len(growth_customers)}</b> customers show potential for increased engagement or spending.</p><p><b>Recommended Action:</b> Use upselling, cross-selling, product bundles, membership benefits and personalized recommendations to increase customer value.</p></div>',
    unsafe_allow_html=True
)


# ============================================================
# LOW ENGAGEMENT
# ============================================================

st.markdown("### 🔄 Low-Engagement Strategy")

st.markdown(
    f'<div class="action-card action-low"><h4>🔄 Low-Engagement Customers</h4><p><b>{low_engagement_count}</b> customers currently show relatively low customer value.</p><p><b>Recommended Action:</b> Use re-engagement campaigns, introductory discounts, reminders and personalized promotions.</p></div>',
    unsafe_allow_html=True
)


# ============================================================
# RECOMMENDED BUSINESS ACTIONS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🚨 Recommended Business Actions</div>',
    unsafe_allow_html=True
)


actions = []


if at_risk_count > 0:

    actions.append({
        "Priority": "HIGH",
        "Business Problem":
            "Customer retention risk",
        "Recommended Action":
            "Contact at-risk customers with personalized retention offers."
    })


if premium_count > 0:

    actions.append({
        "Priority": "HIGH",
        "Business Problem":
            "Protect high-value customers",
        "Recommended Action":
            "Provide loyalty rewards, exclusive benefits and personalized offers."
    })


if len(growth_customers) > 0:

    actions.append({
        "Priority": "MEDIUM",
        "Business Problem":
            "Untapped customer value",
        "Recommended Action":
            "Use upselling, cross-selling and product recommendations."
    })


if low_engagement_count > 0:

    actions.append({
        "Priority": "MEDIUM",
        "Business Problem":
            "Low customer engagement",
        "Recommended Action":
            "Launch targeted re-engagement campaigns."
    })


for action in actions:
    if action["Priority"] == "HIGH":
        css_class = "action-high"
        icon = "🔴"
    elif action["Priority"] == "MEDIUM":
        css_class = "action-medium"
        icon = "🟠"
    else:
        css_class = "action-low"
        icon = "🔵"

    st.markdown(
        f'<div class="action-card {css_class}"><h4>{icon} {action["Priority"]} PRIORITY</h4><p><b>Problem:</b> {action["Business Problem"]}</p><p><b>Action:</b> {action["Recommended Action"]}</p></div>',
        unsafe_allow_html=True
    )


# ============================================================
# ML MODEL ANALYSIS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🤖 ML Model Analysis</div>',
    unsafe_allow_html=True
)


ml_col1, ml_col2, ml_col3 = st.columns(3)


with ml_col1:

    st.metric(
        "Algorithm",
        "K-Means"
    )


with ml_col2:

    st.metric(
        "Best Number of Clusters",
        best_k
    )


with ml_col3:

    st.metric(
        "Silhouette Score",
        f"{best_silhouette:.3f}"
    )


st.markdown(
    "### 📊 Cluster Quality Comparison"
)


st.dataframe(
    silhouette_df,
    use_container_width=True,
    hide_index=True
)


fig, ax = plt.subplots()


ax.plot(
    silhouette_df["K"],
    silhouette_df["Silhouette Score"],
    marker="o"
)


ax.set_xlabel(
    "Number of Clusters (K)"
)

ax.set_ylabel(
    "Silhouette Score"
)

ax.set_title(
    "Silhouette Score by Number of Clusters"
)

ax.grid(True)


st.pyplot(fig)

plt.close(fig)


# ============================================================
# STEP 8 - MODEL COMPARISON
# ============================================================

st.markdown("---")
st.markdown(
    '<div class="section-title">🤖 Model Comparison: K-Means vs Hierarchical Clustering</div>',
    unsafe_allow_html=True
)
st.write(
    "Compare two clustering approaches using the same standardized customer features. "
    "The higher Silhouette Score indicates the better-separated clustering result."
)

comparison_rows = []

# K-Means uses the automatically selected K from the earlier analysis.
kmeans_compare = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)
kmeans_labels_compare = kmeans_compare.fit_predict(X_scaled)
kmeans_score_compare = silhouette_score(X_scaled, kmeans_labels_compare)

# Hierarchical/Agglomerative clustering uses the same K for a fair comparison.
hierarchical_model = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward"
)
hierarchical_labels = hierarchical_model.fit_predict(X_scaled)
hierarchical_score = silhouette_score(X_scaled, hierarchical_labels)

comparison_rows.append({
    "Algorithm": "K-Means",
    "Clusters (K)": best_k,
    "Silhouette Score": round(kmeans_score_compare, 4)
})
comparison_rows.append({
    "Algorithm": "Hierarchical Clustering",
    "Clusters (K)": best_k,
    "Silhouette Score": round(hierarchical_score, 4)
})

model_comparison_df = pd.DataFrame(comparison_rows)
better_model = model_comparison_df.loc[
    model_comparison_df["Silhouette Score"].idxmax(), "Algorithm"
]
better_score = model_comparison_df["Silhouette Score"].max()

comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
with comparison_col1:
    st.metric("🥇 Best Model", better_model)
with comparison_col2:
    st.metric("📊 Best Silhouette", f"{better_score:.3f}")
with comparison_col3:
    st.metric("🔢 Clusters Compared", best_k)

st.dataframe(
    model_comparison_df,
    use_container_width=True,
    hide_index=True
)

fig, ax = plt.subplots()
ax.bar(
    model_comparison_df["Algorithm"],
    model_comparison_df["Silhouette Score"]
)
ax.set_title("Clustering Model Performance")
ax.set_xlabel("Algorithm")
ax.set_ylabel("Silhouette Score")
ax.set_ylim(0, 1)
ax.grid(True, axis="y")
st.pyplot(fig)
plt.close(fig)

if better_model == "K-Means":
    st.success(
        f"🏆 K-Means performed better on this dataset with a Silhouette Score of {better_score:.3f}. "
        "The dashboard therefore continues to use the K-Means segmentation results."
    )
else:
    st.info(
        f"🏆 Hierarchical Clustering achieved the higher Silhouette Score ({better_score:.3f}). "
        "K-Means remains the dashboard's primary model because it supports the existing business segmentation workflow."
    )


# ============================================================
# CUSTOMER VALUE DISTRIBUTION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📊 Customer Value Distribution</div>',
    unsafe_allow_html=True
)


st.write(
    "Distribution of customers based on their Customer Value Score."
)


value_counts = (
    df["Value_Category"]
    .value_counts()
    .reindex(
        [
            "Premium",
            "Valuable",
            "Developing",
            "Low Value"
        ],
        fill_value=0
    )
)


value_col1, value_col2 = st.columns(2)


with value_col1:

    fig, ax = plt.subplots()


    value_counts.plot(
        kind="bar",
        ax=ax
    )


    ax.set_title(
        "Customers by Value Category"
    )

    ax.set_xlabel(
        "Customer Value Category"
    )

    ax.set_ylabel(
        "Number of Customers"
    )


    plt.xticks(
        rotation=25,
        ha="right"
    )


    st.pyplot(fig)

    plt.close(fig)


with value_col2:

    fig, ax = plt.subplots()


    value_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )


    ax.set_title(
        "Customer Value Distribution"
    )

    ax.set_ylabel("")


    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# STEP 7 PART 2
# SEGMENT PERFORMANCE ANALYSIS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📈 Segment Performance Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "Compare customer segments based on customer count, "
    "spending, frequency and overall customer value."
)


# ------------------------------------------------------------
# CREATE SEGMENT PERFORMANCE TABLE
# ------------------------------------------------------------

segment_performance = (
    df.groupby("Segment")
    .agg(
        Customer_Count=(
            "Customer_ID",
            "count"
        ),

        Total_Spending=(
            config["spending_column"],
            "sum"
        ),

        Average_Spending=(
            config["spending_column"],
            "mean"
        ),

        Average_Frequency=(
            config["frequency_column"],
            "mean"
        ),

        Average_Value_Score=(
            "Customer_Value_Score",
            "mean"
        )
    )
    .reset_index()
)


segment_performance[
    "Total_Spending"
] = segment_performance[
    "Total_Spending"
].round(0)


segment_performance[
    "Average_Spending"
] = segment_performance[
    "Average_Spending"
].round(0)


segment_performance[
    "Average_Frequency"
] = segment_performance[
    "Average_Frequency"
].round(2)


segment_performance[
    "Average_Value_Score"
] = segment_performance[
    "Average_Value_Score"
].round(2)


# ------------------------------------------------------------
# PERFORMANCE TABLE
# ------------------------------------------------------------

st.markdown(
    "### 📋 Segment Performance Table"
)


display_segment_performance = (
    segment_performance.rename(
        columns={
            "Segment":
                "Customer Segment",

            "Customer_Count":
                "Customer Count",

            "Total_Spending":
                "Total Spending",

            "Average_Spending":
                "Average Spending",

            "Average_Frequency":
                "Average Frequency",

            "Average_Value_Score":
                "Average Value Score"
        }
    )
)


st.dataframe(
    display_segment_performance,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# SEGMENT KPI CARDS
# ------------------------------------------------------------

st.markdown(
    "### 🏆 Segment Highlights"
)


best_spending_row = segment_performance.loc[
    segment_performance[
        "Average_Spending"
    ].idxmax()
]


best_frequency_row = segment_performance.loc[
    segment_performance[
        "Average_Frequency"
    ].idxmax()
]


best_value_row = segment_performance.loc[
    segment_performance[
        "Average_Value_Score"
    ].idxmax()
]


highlight_col1, highlight_col2, highlight_col3 = st.columns(3)


with highlight_col1:

    st.metric(
        "💰 Highest Avg Spending",
        best_spending_row[
            "Segment"
        ]
    )

    st.caption(
        f"₹{best_spending_row['Average_Spending']:,.0f} average"
    )


with highlight_col2:

    st.metric(
        "🔄 Highest Frequency",
        best_frequency_row[
            "Segment"
        ]
    )

    st.caption(
        f"{best_frequency_row['Average_Frequency']:.2f} average"
    )


with highlight_col3:

    st.metric(
        "⭐ Highest Value Score",
        best_value_row[
            "Segment"
        ]
    )

    st.caption(
        f"{best_value_row['Average_Value_Score']:.2f}/100"
    )


# ------------------------------------------------------------
# TOTAL SPENDING BY SEGMENT
# ------------------------------------------------------------

st.markdown(
    "### 💰 Total Spending by Segment"
)


fig, ax = plt.subplots()


segment_performance_sorted = (
    segment_performance
    .sort_values(
        "Total_Spending",
        ascending=False
    )
)


ax.bar(
    segment_performance_sorted[
        "Segment"
    ],
    segment_performance_sorted[
        "Total_Spending"
    ]
)


ax.set_title(
    "Total Spending by Customer Segment"
)

ax.set_xlabel(
    "Customer Segment"
)

ax.set_ylabel(
    "Total Spending"
)


plt.xticks(
    rotation=25,
    ha="right"
)


st.pyplot(fig)

plt.close(fig)


# ------------------------------------------------------------
# AVERAGE SPENDING BY SEGMENT
# ------------------------------------------------------------

st.markdown(
    "### 🛒 Average Spending Comparison"
)


fig, ax = plt.subplots()


average_spending_segment = (
    segment_performance
    .sort_values(
        "Average_Spending",
        ascending=False
    )
)


ax.bar(
    average_spending_segment[
        "Segment"
    ],
    average_spending_segment[
        "Average_Spending"
    ]
)


ax.set_title(
    "Average Spending by Segment"
)

ax.set_xlabel(
    "Customer Segment"
)

ax.set_ylabel(
    "Average Spending"
)


plt.xticks(
    rotation=25,
    ha="right"
)


st.pyplot(fig)

plt.close(fig)


# ------------------------------------------------------------
# FREQUENCY BY SEGMENT
# ------------------------------------------------------------

st.markdown(
    "### 🔄 Average Purchase / Visit Frequency"
)


fig, ax = plt.subplots()


frequency_segment = (
    segment_performance
    .sort_values(
        "Average_Frequency",
        ascending=False
    )
)


ax.bar(
    frequency_segment[
        "Segment"
    ],
    frequency_segment[
        "Average_Frequency"
    ]
)


ax.set_title(
    "Average Purchase / Visit Frequency by Segment"
)

ax.set_xlabel(
    "Customer Segment"
)

ax.set_ylabel(
    "Average Frequency"
)


plt.xticks(
    rotation=25,
    ha="right"
)


st.pyplot(fig)

plt.close(fig)


# ------------------------------------------------------------
# VALUE SCORE BY SEGMENT
# ------------------------------------------------------------

st.markdown(
    "### ⭐ Average Customer Value Score"
)


fig, ax = plt.subplots()


value_segment = (
    segment_performance
    .sort_values(
        "Average_Value_Score",
        ascending=False
    )
)


ax.bar(
    value_segment[
        "Segment"
    ],
    value_segment[
        "Average_Value_Score"
    ]
)


ax.set_title(
    "Average Customer Value Score by Segment"
)

ax.set_xlabel(
    "Customer Segment"
)

ax.set_ylabel(
    "Value Score (0-100)"
)


plt.xticks(
    rotation=25,
    ha="right"
)


st.pyplot(fig)

plt.close(fig)


# ============================================================
# STEP 7 PART 3
# AT-RISK CUSTOMER ANALYSIS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">⚠️ At-Risk Customer Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "Identify customers who may need immediate retention or re-engagement actions."
)

# Create a detailed at-risk customer table
if at_risk_count > 0:
    risk_df = at_risk_customers.copy()

    risk_reasons = []
    risk_actions = []
    risk_scores = []

    if config["recency_column"]:
        for _, row in risk_df.iterrows():
            recency = row[config["recency_column"]]
            income = row["Annual_Income"]
            reasons = []
            actions_for_customer = []
            score = 0

            if recency > recency_threshold:
                reasons.append(
                    f"Inactive for {recency:.0f} days"
                )
                actions_for_customer.append("Send a personalized re-engagement offer")
                score += 60

            if income >= high_income_threshold:
                reasons.append("High-income customer")
                actions_for_customer.append("Offer loyalty benefits or premium incentives")
                score += 40

            risk_reasons.append(" + ".join(reasons))
            risk_actions.append("; ".join(actions_for_customer))
            risk_scores.append(min(score, 100))

    else:
        for _, row in risk_df.iterrows():
            cart_value = row["Cart_Abandonment"]
            reasons = [
                f"High cart abandonment ({cart_value:.0f})"
            ]
            actions_for_customer = [
                "Send cart recovery reminder",
                "Provide a personalized checkout incentive"
            ]

            # Give higher risk to customers with greater abandonment.
            if cart_threshold > 0:
                score = min(
                    100,
                    (cart_value / cart_threshold) * 60
                )
            else:
                score = 60

            risk_reasons.append(" + ".join(reasons))
            risk_actions.append("; ".join(actions_for_customer))
            risk_scores.append(round(score, 1))

    risk_df["Risk Score"] = risk_scores
    risk_df["Risk Reason"] = risk_reasons
    risk_df["Recommended Action"] = risk_actions

    risk_df = risk_df.sort_values(
        ["Risk Score", "Customer_Value_Score"],
        ascending=False
    )

    # Summary cards
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

    with risk_col1:
        st.metric(
            "⚠️ At-Risk Customers",
            len(risk_df)
        )

    with risk_col2:
        st.metric(
            "💰 Spending at Risk",
            f"₹{risk_df[config['spending_column']].sum():,.0f}"
        )

    with risk_col3:
        st.metric(
            "⭐ Avg Value Score",
            f"{risk_df['Customer_Value_Score'].mean():.1f}"
        )

    with risk_col4:
        st.metric(
            "🚨 Highest Risk Score",
            f"{risk_df['Risk Score'].max():.0f}/100"
        )

    st.markdown("### 👥 Customers Requiring Attention")

    risk_display_columns = [
        "Customer_ID",
        config["spending_column"],
        config["frequency_column"],
        "Customer_Value_Score",
        "Value_Category",
        "Risk Score",
        "Risk Reason",
        "Recommended Action"
    ]

    if config["recency_column"]:
        risk_display_columns.insert(
            3,
            config["recency_column"]
        )
    else:
        risk_display_columns.insert(
            3,
            "Cart_Abandonment"
        )

    risk_display = risk_df[risk_display_columns].copy()

    risk_display = risk_display.rename(
        columns={
            "Customer_ID": "Customer ID",
            config["spending_column"]: "Total Spending",
            config["frequency_column"]: "Frequency",
            config["recency_column"] if config["recency_column"] else "Cart_Abandonment":
                "Recency / Cart Abandonment",
            "Customer_Value_Score": "Value Score",
            "Value_Category": "Value Category"
        }
    )

    st.dataframe(
        risk_display,
        use_container_width=True,
        hide_index=True
    )

    # Highest-risk customer spotlight
    highest_risk_customer = risk_df.iloc[0]

    st.markdown("### 🚨 Highest-Risk Customer Spotlight")

    spotlight_col1, spotlight_col2 = st.columns(2)

    with spotlight_col1:
        st.metric(
            "Customer",
            str(highest_risk_customer["Customer_ID"])
        )
        st.metric(
            "Risk Score",
            f"{highest_risk_customer['Risk Score']:.0f}/100"
        )

    with spotlight_col2:
        st.metric(
            "Customer Value Score",
            f"{highest_risk_customer['Customer_Value_Score']:.1f}/100"
        )
        st.metric(
            "Total Spending",
            f"₹{highest_risk_customer[config['spending_column']]:,.0f}"
        )

    st.warning(
        f"**Why this customer needs attention:** "
        f"{highest_risk_customer['Risk Reason']}"
    )

    st.success(
        f"**Recommended next action:** "
        f"{highest_risk_customer['Recommended Action']}"
    )

    # Risk score chart
    st.markdown("### 📊 Risk Score by Customer")

    risk_chart_df = risk_df.head(10).sort_values(
        "Risk Score",
        ascending=True
    )

    fig, ax = plt.subplots()
    ax.barh(
        risk_chart_df["Customer_ID"].astype(str),
        risk_chart_df["Risk Score"]
    )
    ax.set_xlabel("Risk Score (0-100)")
    ax.set_ylabel("Customer")
    ax.set_title("Top At-Risk Customers")
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x")
    st.pyplot(fig)
    plt.close(fig)

else:
    st.success(
        "✅ No customers currently meet the configured at-risk criteria."
    )


# ============================================================
# SEGMENT DISTRIBUTION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">👥 Segment Distribution</div>',
    unsafe_allow_html=True
)


segment_counts = (
    df["Segment"]
    .value_counts()
)


chart_col1, chart_col2 = st.columns(2)


with chart_col1:

    fig, ax = plt.subplots()


    segment_counts.plot(
        kind="bar",
        ax=ax
    )


    ax.set_title(
        "Customers by Segment"
    )

    ax.set_xlabel(
        "Customer Segment"
    )

    ax.set_ylabel(
        "Number of Customers"
    )


    plt.xticks(
        rotation=25,
        ha="right"
    )


    st.pyplot(fig)

    plt.close(fig)


with chart_col2:

    fig, ax = plt.subplots()


    segment_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )


    ax.set_ylabel("")

    ax.set_title(
        "Segment Percentage"
    )


    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# AVERAGE SPENDING BY SEGMENT
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">💰 Average Spending by Segment</div>',
    unsafe_allow_html=True
)


segment_spending = (
    df.groupby("Segment")[
        config["spending_column"]
    ]
    .mean()
    .sort_values(
        ascending=False
    )
)


fig, ax = plt.subplots()


segment_spending.plot(
    kind="bar",
    ax=ax
)


ax.set_title(
    "Average Spending by Customer Segment"
)

ax.set_xlabel(
    "Customer Segment"
)

ax.set_ylabel(
    "Average Spending"
)


plt.xticks(
    rotation=25,
    ha="right"
)


st.pyplot(fig)

plt.close(fig)


# ============================================================
# CUSTOMER BEHAVIOR
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📈 Customer Behavior Analysis</div>',
    unsafe_allow_html=True
)


behavior_col1, behavior_col2 = st.columns(2)


with behavior_col1:

    fig, ax = plt.subplots()


    for segment in df["Segment"].unique():

        subset = df[
            df["Segment"] == segment
        ]


        ax.scatter(
            subset["Annual_Income"],
            subset[
                config["spending_column"]
            ],
            label=segment
        )


    ax.set_xlabel(
        "Annual Income"
    )

    ax.set_ylabel(
        "Total Spending"
    )

    ax.set_title(
        "Income vs Spending"
    )

    ax.legend()

    ax.grid(True)


    st.pyplot(fig)

    plt.close(fig)


with behavior_col2:

    fig, ax = plt.subplots()


    ax.scatter(
        df[
            config["frequency_column"]
        ],
        df[
            config["spending_column"]
        ]
    )


    ax.set_xlabel(
        "Purchase / Visit Frequency"
    )

    ax.set_ylabel(
        "Total Spending"
    )

    ax.set_title(
        "Frequency vs Spending"
    )

    ax.grid(True)


    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">💡 Business Insights</div>',
    unsafe_allow_html=True
)


highest_spending_segment = (
    df.groupby("Segment")[
        config["spending_column"]
    ]
    .mean()
    .idxmax()
)


highest_frequency_segment = (
    df.groupby("Segment")[
        config["frequency_column"]
    ]
    .mean()
    .idxmax()
)


largest_segment = (
    df["Segment"]
    .value_counts()
    .idxmax()
)


st.success(
    f"💰 Highest-spending segment: "
    f"**{highest_spending_segment}**"
)


st.info(
    f"🛒 Highest-frequency segment: "
    f"**{highest_frequency_segment}**"
)


st.warning(
    f"👥 Largest customer segment: "
    f"**{largest_segment}**"
)


# ============================================================
# INDIVIDUAL CUSTOMER INTELLIGENCE
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🔍 Individual Customer Intelligence</div>',
    unsafe_allow_html=True
)


customer_options = df[
    "Customer_ID"
].tolist()


selected_customer = st.selectbox(
    "Select a Customer",
    customer_options
)


customer_row = df[
    df["Customer_ID"]
    ==
    selected_customer
].iloc[0]


customer_col1, customer_col2, customer_col3, customer_col4 = st.columns(4)


with customer_col1:

    st.metric(
        "Customer",
        customer_row[
            "Customer_ID"
        ]
    )


with customer_col2:

    st.metric(
        "Segment",
        customer_row[
            "Segment"
        ]
    )


with customer_col3:

    st.metric(
        "Value Score",
        f"{customer_row['Customer_Value_Score']:.1f}"
    )


with customer_col4:

    st.metric(
        "Value Category",
        customer_row[
            "Value_Category"
        ]
    )


st.markdown(
    "### 📋 Customer Profile"
)


profile_data = {
    "Attribute": [],
    "Value": []
}


for column in df.columns:

    if column in [
        "Cluster",
        "Segment",
        "Customer_Value_Score",
        "Value_Category"
    ]:

        continue


    profile_data[
        "Attribute"
    ].append(
        column
    )


    profile_data[
        "Value"
    ].append(
        customer_row[column]
    )


profile_df = pd.DataFrame(
    profile_data
)


st.dataframe(
    profile_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CUSTOMER RECOMMENDATION
# ============================================================

st.markdown(
    "### 🎯 Personalized Recommendation"
)


if selected_customer in set(
    at_risk_customers[
        "Customer_ID"
    ]
):

    recommendation = (
        "⚠️ This customer may be at risk. "
        "Consider personalized retention offers, "
        "loyalty rewards and direct follow-up."
    )


elif customer_row[
    "Value_Category"
] == "Premium":

    recommendation = (
        "🏆 Premium customer. "
        "Focus on retention, exclusive benefits, "
        "VIP treatment and personalized recommendations."
    )


elif customer_row[
    "Value_Category"
] == "Valuable":

    recommendation = (
        "📈 Valuable customer. "
        "Use upselling and cross-selling strategies "
        "to increase customer lifetime value."
    )


elif customer_row[
    "Value_Category"
] == "Developing":

    recommendation = (
        "🌱 Developing customer. "
        "Encourage repeat purchases/visits using "
        "targeted promotions and product recommendations."
    )


else:

    recommendation = (
        "🔄 Low-value customer. "
        "Use re-engagement campaigns and personalized "
        "offers to increase activity."
    )


st.info(
    recommendation
)


# ============================================================
# CUSTOMER SEARCH
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🔎 Customer Search & Filtering</div>',
    unsafe_allow_html=True
)


search_text = st.text_input(
    "Search Customer ID"
)


filtered_df = df.copy()


if search_text:

    filtered_df = filtered_df[
        filtered_df[
            "Customer_ID"
        ]
        .astype(str)
        .str.contains(
            search_text,
            case=False,
            na=False
        )
    ]


segment_filter = st.multiselect(
    "Filter by Segment",
    options=sorted(
        df["Segment"].unique()
    )
)


if segment_filter:

    filtered_df = filtered_df[
        filtered_df[
            "Segment"
        ].isin(
            segment_filter
        )
    ]


st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# TOP CUSTOMERS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🏆 Top Customers</div>',
    unsafe_allow_html=True
)


top_customers = (
    df.sort_values(
        config["spending_column"],
        ascending=False
    )
    .head(10)
)


st.dataframe(
    top_customers[
        [
            "Customer_ID",
            "Segment",
            config["spending_column"],
            config["frequency_column"],
            "Customer_Value_Score",
            "Value_Category"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# STEP 7 PART 4 - CUSTOMER OPPORTUNITY & RECOMMENDATION ENGINE
# ============================================================

st.markdown("---")
st.markdown(
    '<div class="section-title">💡 Customer Opportunity & Recommendation Engine</div>',
    unsafe_allow_html=True
)
st.write(
    "Convert customer value and behavior into a practical next-best business action."
)

def build_opportunity(row):
    category = row["Value_Category"]
    is_risk = bool(at_risk_mask.loc[row.name])

    if is_risk:
        opportunity = "Retention / Win-back"
        if business_type == "E-Commerce":
            action = "Send cart-recovery and personalized discount offers."
        elif business_type == "Restaurant":
            action = "Send a personalized comeback offer and loyalty reward."
        else:
            action = "Send a personalized retention offer and loyalty benefit."
        priority = "HIGH"
    elif category == "Premium":
        opportunity = "Loyalty & Retention"
        if business_type == "E-Commerce":
            action = "Offer VIP benefits, early access and premium product recommendations."
        elif business_type == "Restaurant":
            action = "Offer VIP dining benefits, loyalty rewards and exclusive experiences."
        else:
            action = "Offer loyalty rewards, exclusive benefits and personalized recommendations."
        priority = "HIGH"
    elif category == "Valuable":
        opportunity = "Upsell / Cross-sell"
        if business_type == "E-Commerce":
            action = "Recommend complementary products and higher-value bundles."
        elif business_type == "Restaurant":
            action = "Promote add-ons, combos and premium menu options."
        else:
            action = "Recommend complementary products, bundles and upgrades."
        priority = "MEDIUM"
    elif category == "Developing":
        opportunity = "Nurture & Grow"
        action = "Encourage repeat purchases or visits with targeted promotions."
        priority = "MEDIUM"
    else:
        opportunity = "Re-engagement"
        action = "Use reminders, introductory offers and personalized promotions."
        priority = "LOW"

    return opportunity, priority, action

opportunity_rows = []
for idx, row in df.iterrows():
    opportunity, priority, action = build_opportunity(row)
    opportunity_rows.append({
        "Customer ID": row["Customer_ID"],
        "Segment": row["Segment"],
        "Value Score": round(row["Customer_Value_Score"], 1),
        "Value Category": row["Value_Category"],
        "Opportunity": opportunity,
        "Priority": priority,
        "Recommended Action": action
    })

opportunity_df = pd.DataFrame(opportunity_rows)

priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
opportunity_df["_priority_order"] = opportunity_df["Priority"].map(priority_order)
opportunity_df = opportunity_df.sort_values(
    ["_priority_order", "Value Score"],
    ascending=[True, False]
).drop(columns=["_priority_order"])

opp_high = (opportunity_df["Priority"] == "HIGH").sum()
opp_medium = (opportunity_df["Priority"] == "MEDIUM").sum()
opp_low = (opportunity_df["Priority"] == "LOW").sum()

opp_col1, opp_col2, opp_col3 = st.columns(3)
with opp_col1:
    st.metric("🔴 High Priority", opp_high)
with opp_col2:
    st.metric("🟠 Medium Priority", opp_medium)
with opp_col3:
    st.metric("🔵 Low Priority", opp_low)

st.markdown("### 🎯 Next-Best Action for Customers")
st.dataframe(
    opportunity_df,
    use_container_width=True,
    hide_index=True
)

opportunity_counts = opportunity_df["Opportunity"].value_counts()
fig, ax = plt.subplots()
opportunity_counts.plot(kind="bar", ax=ax)
ax.set_title("Customer Opportunities")
ax.set_xlabel("Opportunity Type")
ax.set_ylabel("Number of Customers")
plt.xticks(rotation=25, ha="right")
st.pyplot(fig)
plt.close(fig)

st.info(
    "💡 Business logic: at-risk customers receive retention actions first; "
    "Premium customers receive loyalty actions; Valuable customers receive upsell/cross-sell actions; "
    "Developing customers receive nurturing actions; Low Value customers receive re-engagement actions."
)


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">⬇️ Export Results</div>',
    unsafe_allow_html=True
)


csv_data = df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="📥 Download Customer Intelligence Results",
    data=csv_data,
    file_name="customer_intelligence_results.csv",
    mime="text/csv"
)


# ============================================================
# STEP 9 - UPGRADED INTELLIGENT BUSINESS ASSISTANT
# ============================================================

st.markdown("---")
st.markdown(
    '<div class="section-title">🧠 Intelligent Business Assistant</div>',
    unsafe_allow_html=True
)
st.write(
    "Ask natural-language questions about customers, segments, risk, value, opportunities, "
    "models and recommended business actions."
)

question = st.text_input(
    "Ask your question",
    placeholder="Example: Which customers should I retain first?"
)

if question:
    q = question.lower().strip()
    answer = None

    # Specific customer questions
    for customer_id in df["Customer_ID"].astype(str):
        if customer_id.lower() in q:
            row = df[df["Customer_ID"].astype(str) == customer_id].iloc[0]
            customer_is_risk = bool(at_risk_mask.loc[row.name])
            opportunity_row = opportunity_df[
                opportunity_df["Customer ID"].astype(str) == customer_id
            ].iloc[0]
            risk_text = "at risk" if customer_is_risk else "not currently flagged as at risk"
            answer = (
                f"Customer **{customer_id}** belongs to **{row['Segment']}**. "
                f"Their spending is **₹{row[config['spending_column']]:,.0f}**, "
                f"Customer Value Score is **{row['Customer_Value_Score']:.1f}/100**, "
                f"and they are **{risk_text}**. "
                f"Next-best action: **{opportunity_row['Recommended Action']}**"
            )
            break

    # At-risk / retention
    if answer is None and any(term in q for term in [
        "at risk", "at-risk", "risk", "retain", "retention"
    ]):
        if at_risk_count > 0:
            top_risk_id = str(risk_df.iloc[0]["Customer_ID"]) if "risk_df" in globals() else str(at_risk_customers.iloc[0]["Customer_ID"])
            answer = (
                f"There are **{at_risk_count} at-risk customers**. "
                f"The first customer to prioritize is **{top_risk_id}**. "
                "Use personalized retention offers, loyalty incentives and direct follow-up."
            )
        else:
            answer = "No customers currently meet the configured at-risk criteria."

    # Opportunity / recommendation
    if answer is None and any(term in q for term in [
        "opportunity", "next best", "recommendation", "recommend", "what should", "action"
    ]):
        high_priority = opportunity_df[opportunity_df["Priority"] == "HIGH"]
        if len(high_priority) > 0:
            top = high_priority.iloc[0]
            answer = (
                f"Your highest-priority opportunity is **{top['Opportunity']}** for customer "
                f"**{top['Customer ID']}**. Recommended action: **{top['Recommended Action']}**"
            )
        else:
            answer = "Focus on increasing engagement through targeted promotions and personalized recommendations."

    # Premium
    if answer is None and any(term in q for term in ["premium", "high value", "high-value"]):
        answer = f"There are **{premium_count} premium customers**. Prioritize retention and loyalty benefits for them."

    # Highest spending
    if answer is None and any(term in q for term in ["highest spending", "most spending", "spend the most"]):
        answer = f"The **{highest_spending_segment}** segment has the highest average spending."

    # Highest frequency
    if answer is None and any(term in q for term in ["highest frequency", "most frequent"]):
        answer = f"The **{highest_frequency_segment}** segment has the highest average purchase/visit frequency."

    # Largest segment
    if answer is None and any(term in q for term in ["largest segment", "biggest segment"]):
        answer = f"The largest segment is **{largest_segment}**."

    # Segment explanation
    if answer is None and "segment" in q:
        segment_name = None
        for candidate in df["Segment"].unique():
            if candidate.lower() in q:
                segment_name = candidate
                break
        if segment_name:
            seg = segment_performance[segment_performance["Segment"] == segment_name].iloc[0]
            answer = (
                f"**{segment_name}** has **{int(seg['Customer_Count'])} customers**, "
                f"total spending of **₹{seg['Total_Spending']:,.0f}**, "
                f"average spending of **₹{seg['Average_Spending']:,.0f}**, "
                f"and an average value score of **{seg['Average_Value_Score']:.1f}/100**."
            )

    # Model comparison
    if answer is None and any(term in q for term in [
        "model", "k-means", "kmeans", "hierarchical", "silhouette"
    ]):
        answer = (
            f"Model comparison: **K-Means = {kmeans_score_compare:.3f}** and "
            f"**Hierarchical Clustering = {hierarchical_score:.3f}** Silhouette Score. "
            f"The better score on this dataset is from **{better_model}**."
        )

    # Customer count
    if answer is None and any(term in q for term in [
        "how many customers", "number of customers", "total customers"
    ]):
        answer = f"The dataset contains **{total_customers} customers**."

    # Spending
    if answer is None and any(term in q for term in ["total spending", "total revenue"]):
        answer = f"Total recorded customer spending is **₹{total_spending:,.0f}**."

    if answer is None and any(term in q for term in ["average spending", "avg spending"]):
        answer = f"Average customer spending is **₹{average_spending:,.0f}**."

    # Value score
    if answer is None and any(term in q for term in ["value score", "customer score"]):
        answer = f"The average Customer Value Score is **{average_value_score:.1f}/100**."

    # Best K
    if answer is None and any(term in q for term in ["best k", "number of clusters"]):
        answer = (
            f"The model selected **K = {best_k}** using the highest Silhouette Score of **{best_silhouette:.3f}**."
        )

    # Growth
    if answer is None and any(term in q for term in ["growth", "upsell", "cross-sell"]):
        answer = (
            f"There are **{len(growth_customers)} growth opportunity customers**. "
            "Use upselling, cross-selling, bundles and personalized recommendations."
        )

    # Low engagement
    if answer is None and any(term in q for term in ["low engagement", "inactive", "low value"]):
        answer = (
            f"There are **{low_engagement_count} low-engagement customers**. "
            "Use re-engagement campaigns and targeted offers."
        )

    # Summary
    if answer is None and any(term in q for term in ["summary", "overview", "business insight"]):
        answer = (
            f"The business has **{total_customers} customers** and total spending of **₹{total_spending:,.0f}**. "
            f"The selected K-Means configuration is **K={best_k}** with Silhouette Score **{best_silhouette:.3f}**. "
            f"Average Customer Value Score is **{average_value_score:.1f}/100**. "
            f"There are **{at_risk_count} at-risk customers** and **{opp_high} high-priority opportunities**."
        )

    # Default
    if answer is None:
        answer = (
            "I can answer questions such as:\n\n"
            "• Which customers are at risk?\n"
            "• Which customer should I retain first?\n"
            "• What is the next-best action?\n"
            "• Which segment has the highest spending?\n"
            "• Which model performed better?\n"
            "• What is the best K?\n"
            "• How many premium customers are there?\n"
            "• What is the business summary?"
        )

    st.info(answer)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    "**Business Customer Intelligence Platform**\n\n"
    "Machine Learning • Customer Segmentation • Customer Value Analysis • "
    "Business Intelligence • Action Recommendations"
)