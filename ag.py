# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(
    page_title="Global Superstore: Order Priority Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data(path):
    df = pd.read_excel(path, engine="openpyxl")
    # Create Lead Time
    df["Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days
    return df

# ---------------------------------------------------------
# 1) Load dataset
# ---------------------------------------------------------
df = load_data("Global Superstore.xlsx")

# ---------------------------------------------------------
# 2) Sidebar: Interactive Filters
# ---------------------------------------------------------
st.sidebar.header("🔎 Filter Your Dataset")

# 2a) Date range filter (Order Date)
min_date = df["Order Date"].min()
max_date = df["Order Date"].max()
start_date, end_date = st.sidebar.date_input(
    "Order Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

# 2b) Region filter
regions = sorted(df["Region"].unique())
selected_regions = st.sidebar.multiselect("Select Region(s)", options=regions, default=regions)

# 2c) Category filter
categories = sorted(df["Category"].unique())
selected_categories = st.sidebar.multiselect("Select Category(ies)", options=categories, default=categories)

# 2d) Ship Mode filter
ship_modes = sorted(df["Ship Mode"].unique())
selected_shipmodes = st.sidebar.multiselect("Select Ship Mode(s)", options=ship_modes, default=ship_modes)

# 2e) Segment filter (added for segment-specific plots)
segments = sorted(df["Segment"].unique())
selected_segments = st.sidebar.multiselect("Select Segment(s)", options=segments, default=segments)

# Apply filters
mask = (
    (df["Order Date"] >= pd.to_datetime(start_date))
    & (df["Order Date"] <= pd.to_datetime(end_date))
    & (df["Region"].isin(selected_regions))
    & (df["Category"].isin(selected_categories))
    & (df["Ship Mode"].isin(selected_shipmodes))
    & (df["Segment"].isin(selected_segments))
)
filtered = df[mask].copy()

st.sidebar.markdown(f"**📊 Records after filtering:** {filtered.shape[0]:,}")

# ---------------------------------------------------------
# 3) Main Content
# ---------------------------------------------------------
st.title("Global Superstore – Order Priority Dashboard")
st.markdown(
    """
    Use the controls on the left to filter by Date, Region, Category, Ship Mode, and Segment.  
    Below you’ll find multiple tabs and charts—everything from scatterplots to pairplots, histograms,  
    and model‐preview placeholders—to help you understand which features drive “Order Priority.”
    """
)

# ---------------------------------------------------------
# 3a) Summary metrics
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Orders", f"{filtered.shape[0]:,}")
with col2:
    st.metric("Distinct Customers", f"{filtered['Customer ID'].nunique():,}")
with col3:
    st.metric("Distinct Products", f"{filtered['Product ID'].nunique():,}")
with col4:
    st.metric("Avg. Lead Time (days)", f"{filtered['Lead Time'].mean():.1f}")

st.markdown("---")

# ---------------------------------------------------------
# 3b) Tabs for different visualizations
# ---------------------------------------------------------
tabs = st.tabs([
    "1) Overview", 
    "2) Scatter & Boxplots", 
    "3) Histograms (Before vs. After)", 
    "4) Pairplot", 
    "5) Segment Analysis", 
    "6) Model Preview", 
    "7) Model Performance"
])

# ------------------------
# Tab 1: Overview
# ------------------------
with tabs[0]:
    st.subheader("📈 Distribution of Order Priority")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    priority_order = ["Low", "Medium", "High", "Critical"]
    counts = filtered["Order Priority"].value_counts().reindex(priority_order, fill_value=0)
    counts.plot(kind="bar", ax=ax1, color="#ff8c00")
    ax1.set_xlabel("Order Priority")
    ax1.set_ylabel("Number of Orders")
    ax1.set_title("How Many Orders by Priority")
    for p in ax1.patches:
        ax1.annotate(
            f"{int(p.get_height()):,}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    st.pyplot(fig1)

    st.subheader("📊 Box Plots: Sales, Profit, and Lead Time by Priority")
    subcols = st.columns(3)
    with subcols[0]:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        filtered.boxplot(
            column="Sales",
            by="Order Priority",
            showcaps=True,
            boxprops=dict(color="#ff8c00"),
            whiskerprops=dict(color="#ff8c00"),
            medianprops=dict(color="red"),
            patch_artist=True,
            showfliers=False,
            ax=ax2,
        )
        ax2.set_title("Sales by Priority")
        ax2.set_xlabel("")
        ax2.set_ylabel("Sales")
        plt.suptitle("")
        st.pyplot(fig2)

    with subcols[1]:
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        filtered.boxplot(
            column="Profit",
            by="Order Priority",
            showcaps=True,
            boxprops=dict(color="#ff8c00"),
            whiskerprops=dict(color="#ff8c00"),
            medianprops=dict(color="red"),
            patch_artist=True,
            showfliers=False,
            ax=ax3,
        )
        ax3.set_title("Profit by Priority")
        ax3.set_xlabel("")
        ax3.set_ylabel("Profit")
        plt.suptitle("")
        st.pyplot(fig3)

    with subcols[2]:
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        filtered.boxplot(
            column="Lead Time",
            by="Order Priority",
            showcaps=True,
            boxprops=dict(color="#ff8c00"),
            whiskerprops=dict(color="#ff8c00"),
            medianprops=dict(color="red"),
            patch_artist=True,
            showfliers=False,
            ax=ax4,
        )
        ax4.set_title("Lead Time by Priority")
        ax4.set_xlabel("")
        ax4.set_ylabel("Days")
        plt.suptitle("")
        st.pyplot(fig4)

    st.markdown("---")

    st.subheader("🌐 Priority Distribution Across Regions")
    region_priority = (
        filtered.groupby(["Region", "Order Priority"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=priority_order)
    )
    fig5, ax5 = plt.subplots(figsize=(8, 3.5))
    region_priority.plot(
        kind="bar", stacked=True, ax=ax5,
        color=["#ffa500", "#ff8c53", "#e03131", "#d63384"]
    )
    ax5.set_xlabel("Region")
    ax5.set_ylabel("Number of Orders")
    ax5.set_title("Stacked Bar: Priority by Region")
    ax5.legend(title="Priority", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig5)

# ------------------------
# Tab 2: Scatter & Boxplots
# ------------------------
with tabs[1]:
    st.subheader("📌 Scatter: Sales vs. Profit (colored by Priority)")
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=filtered,
        x="Sales",
        y="Profit",
        hue="Order Priority",
        palette={"Low": "#ff8c00", "Medium": "#ff8c53", "High": "#e03131", "Critical": "#d63384"},
        alpha=0.6,
        ax=ax6,
    )
    ax6.set_title("Sales vs Profit, colored by Priority")
    ax6.set_xlabel("Sales")
    ax6.set_ylabel("Profit")
    ax6.legend(title="Priority", bbox_to_anchor=(1.0, 1.0))
    st.pyplot(fig6)

    st.markdown("---")
    st.subheader("🗂 Box Plots for Additional Categorical Variables")
    extra_cols = ["Ship Mode", "Segment"]
    for col in extra_cols:
        st.markdown(f"**Profit by {col}**")
        fig_, ax_ = plt.subplots(figsize=(6, 3))
        sns.boxplot(
            data=filtered,
            x=col,
            y="Profit",
            palette="Oranges",
            showfliers=False,
            ax=ax_,
        )
        ax_.set_xlabel(col)
        ax_.set_ylabel("Profit")
        ax_.set_title(f"Profit by {col}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_)
        st.markdown("---")

# ------------------------
# Tab 3: Histograms (Before vs. After)
# ------------------------
with tabs[2]:
    st.subheader("📊 Compare Distributions: Before Filtering vs After Filtering")
    numeric_cols = ["Sales", "Profit", "Quantity", "Discount", "Lead Time"]
    selected_numeric = st.selectbox("Select numeric column for histogram", numeric_cols, index=0)

    # Histogram on full dataset
    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(10, 4))
    # Before filtering
    df[selected_numeric].hist(bins=30, alpha=0.5, color="#ff8c00", ax=ax7a)
    ax7a.set_title(f"Full Dataset: {selected_numeric}")
    ax7a.set_xlabel(selected_numeric)
    ax7a.set_ylabel("Count")

    # After filtering
    filtered[selected_numeric].hist(bins=30, alpha=0.5, color="#e03131", ax=ax7b)
    ax7b.set_title(f"Filtered Dataset: {selected_numeric}")
    ax7b.set_xlabel(selected_numeric)
    ax7b.set_ylabel("Count")

    plt.tight_layout()
    st.pyplot(fig7)

# ------------------------
# Tab 4: Pairplot
# ------------------------
with tabs[3]:
    st.subheader("📐 Pairplot of Numeric Features (colored by Priority)")
    st.markdown(
        "This pairplot uses Seaborn to show joint distributions and pairwise "
        "relationships among numeric columns. Depending on data size, it may take a few seconds."
    )
    numeric_for_pair = ["Sales", "Profit", "Quantity", "Discount", "Lead Time"]
    with st.spinner("Generating pairplot... (this can be slow)"):
        pair_df = filtered[numeric_for_pair + ["Order Priority"]].dropna()
        # Sample down if too large
        if pair_df.shape[0] > 2000:
            pair_df = pair_df.sample(2000, random_state=42)
        pp = sns.pairplot(
            pair_df,
            hue="Order Priority",
            vars=numeric_for_pair,
            corner=True,
            palette={"Low": "#ff8c00", "Medium": "#ff8c53", "High": "#e03131", "Critical": "#d63384"},
            plot_kws={"alpha": 0.5, "s": 20},
        )
        st.pyplot(pp)

# ------------------------
# Tab 5: Segment Analysis
# ------------------------
with tabs[4]:
    st.subheader("📋 Segment‐Level Insights")
    st.markdown("1. Stacked Bar: Priority counts by Segment\n2. Boxplots: Sales & Profit by Segment & Priority")

    # 5a) Stacked bar: Priority by Segment
    seg_priority = (
        filtered.groupby(["Segment", "Order Priority"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=priority_order)
    )
    fig8, ax8 = plt.subplots(figsize=(6, 3.5))
    seg_priority.plot(
        kind="bar",
        stacked=True,
        ax=ax8,
        color=["#ffa500", "#ff8c53", "#e03131", "#d63384"],
    )
    ax8.set_xlabel("Segment")
    ax8.set_ylabel("Number of Orders")
    ax8.set_title("Stacked Bar: Priority by Segment")
    ax8.legend(title="Priority", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    st.pyplot(fig8)

    st.markdown("---")

    # 5b) Boxplots: Sales by Segment, hue=Priority
    st.markdown("**Boxplot: Sales by Segment (split by Priority)**")
    fig9, ax9 = plt.subplots(figsize=(6, 3.5))
    sns.boxplot(
        data=filtered,
        x="Segment",
        y="Sales",
        hue="Order Priority",
        palette={"Low": "#ff8c00", "Medium": "#ff8c53", "High": "#e03131", "Critical": "#d63384"},
        showfliers=False,
        ax=ax9,
    )
    ax9.set_xlabel("Segment")
    ax9.set_ylabel("Sales")
    ax9.set_title("Sales by Segment & Priority")
    ax9.legend(title="Priority", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    st.pyplot(fig9)

    st.markdown("---")
    st.markdown(
        "🎯 You can similarly create boxplots for Profit or Quantity by Segment & Priority "
        "if you wish to drill deeper."
    )

# ------------------------
# Tab 6: Model Preview
# ------------------------
with tabs[5]:
    st.subheader("🤖 Model Preview (Upload a Trained Model & Input Features)")
    st.markdown(
        """
        **How it works:**  
        1. Upload your trained neural‐network file (e.g., Keras `.h5`, Pickle‐saved SKLearn pipeline).  
        2. Enter hypothetical feature values below (e.g., Sales, Profit, Quantity, Discount, Lead Time, Region, etc.).  
        3. Click **“Predict Priority”** to see what priority the model assigns.
        """
    )

    model_file = st.file_uploader("🔄 Upload Trained Model (H5 or Pickle)", type=["h5", "pkl"])
    if model_file:
        st.success("Model uploaded ✅")
        # Example: if Keras .h5, load using tensorflow.keras.models.load_model
        # if pickle, load sklearn-like pipeline
        try:
            from tensorflow.keras.models import load_model

            model = load_model(model_file)
            st.info("Detected Keras model. Please fill in all numeric and categorical fields below.")
            # Collect numeric features
            sales_in = st.number_input("Sales", value=100.0, step=1.0)
            profit_in = st.number_input("Profit", value=10.0, step=1.0)
            quantity_in = st.number_input("Quantity", value=1, step=1)
            discount_in = st.number_input("Discount", value=0.1, step=0.01, format="%.2f")
            leadtime_in = st.number_input("Lead Time (days)", value=2, step=1)
            # Collect categorical (use same encoding as during training)
            region_in = st.selectbox("Region", options=regions, index=0)
            shipmode_in = st.selectbox("Ship Mode", options=ship_modes, index=0)
            category_in = st.selectbox("Category", options=categories, index=0)
            segment_in = st.selectbox("Segment", options=segments, index=0)

            if st.button("Predict Priority"):
                # NOTE: You must preprocess these exactly as during training!
                # Here is a placeholder that simply packs into array and calls model.predict:
                input_df = pd.DataFrame([{
                    "Sales": sales_in,
                    "Profit": profit_in,
                    "Quantity": quantity_in,
                    "Discount": discount_in,
                    "Lead Time": leadtime_in,
                    # Categorical fields must be one-hot or label-encoded the same way
                    # For demonstration, we skip proper encoding
                    "Region": region_in,
                    "Ship Mode": shipmode_in,
                    "Category": category_in,
                    "Segment": segment_in,
                }])
                # You need to apply the exact preprocessing pipeline used during training.
                # E.g. one-hot encode Region/category/Segment, scale numeric columns, etc.
                st.warning("⚠️ This is a placeholder. You must insert your own preprocessing code.")
                # Example dummy output:
                st.metric("Predicted Priority", "Medium")

        except Exception as e:
            import pickle

            model = pickle.load(model_file)
            st.info("Loaded a Pickle‐serialized model. Please fill features below.")
            # (Duplicate inputs)
            sales_in = st.number_input("Sales", value=100.0, step=1.0)
            profit_in = st.number_input("Profit", value=10.0, step=1.0)
            quantity_in = st.number_input("Quantity", value=1, step=1)
            discount_in = st.number_input("Discount", value=0.1, step=0.01, format="%.2f")
            leadtime_in = st.number_input("Lead Time (days)", value=2, step=1)
            region_in = st.selectbox("Region", options=regions, index=0)
            shipmode_in = st.selectbox("Ship Mode", options=ship_modes, index=0)
            category_in = st.selectbox("Category", options=categories, index=0)
            segment_in = st.selectbox("Segment", options=segments, index=0)

            if st.button("Predict Priority"):
                # Again, this is a placeholder:
                st.warning("⚠️ You need to preprocess identically to your training pipeline.")
                st.metric("Predicted Priority", "Low")

    else:
        st.info("Upload a model to enable predictions.")

# ------------------------
# Tab 7: Model Performance
# ------------------------
with tabs[6]:
    st.subheader("📈 Model Performance: Before vs. After Feature Engineering")
    st.markdown(
        """
        Upload two small CSVs (or JSONs) containing metrics (accuracy, precision, recall) 
        and confusion‐matrix images for:
        1. **Before Feature Engineering** (original dataset).  
        2. **After Feature Engineering** (engineered, filtered dataset).
        """
    )

    col_before, col_after = st.columns(2)

    with col_before:
        st.markdown("#### Before Feature Engineering")
        bf_metrics_file = st.file_uploader("Metrics CSV/JSON (Before)", type=["csv", "json"], key="bf_metrics")
        bf_cm_file = st.file_uploader("Confusion Matrix Image (Before)", type=["png", "jpg", "jpeg"], key="bf_cm")

        if bf_metrics_file:
            # Read metrics
            if str(bf_metrics_file.name).endswith(".csv"):
                bf_metrics = pd.read_csv(bf_metrics_file)
            else:
                bf_metrics = pd.read_json(bf_metrics_file, orient="records")
            # Expect columns: metric, value
            if {"metric", "value"}.issubset(bf_metrics.columns):
                for _, row in bf_metrics.iterrows():
                    st.metric(label=row["metric"].capitalize(), value=f"{row['value']:.3f}")
            else:
                st.error("CSV/JSON must have columns: 'metric', 'value'.")
        if bf_cm_file:
            st.image(bf_cm_file, caption="Confusion Matrix (Before)", use_column_width=True)

    with col_after:
        st.markdown("#### After Feature Engineering")
        af_metrics_file = st.file_uploader("Metrics CSV/JSON (After)", type=["csv", "json"], key="af_metrics")
        af_cm_file = st.file_uploader("Confusion Matrix Image (After)", type=["png", "jpg", "jpeg"], key="af_cm")

        if af_metrics_file:
            if str(af_metrics_file.name).endswith(".csv"):
                af_metrics = pd.read_csv(af_metrics_file)
            else:
                af_metrics = pd.read_json(af_metrics_file, orient="records")
            if {"metric", "value"}.issubset(af_metrics.columns):
                for _, row in af_metrics.iterrows():
                    st.metric(label=row["metric"].capitalize(), value=f"{row['value']:.3f}")
            else:
                st.error("CSV/JSON must have columns: 'metric', 'value'.")
        if af_cm_file:
            st.image(af_cm_file, caption="Confusion Matrix (After)", use_column_width=True)

    st.markdown(
        """
        💡 **Note:**  
        - Your metrics CSV/JSON should look like:  
          ```
          metric,value
          accuracy,0.82
          precision,0.75
          recall,0.79
          ```
        - Confusion matrices can be simple PNG/JPG exports from Matplotlib or another library.
        """
    )

# ---------------------------------------------------------
# 4) Dataframe preview (collapsed)
# ---------------------------------------------------------
st.sidebar.markdown("---")
with st.sidebar.expander("📋 Show Sample of Filtered Data"):
    st.dataframe(filtered.head(200))
