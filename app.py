# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from trustpulse import load_and_preprocess, train_trust_model, summarize_results

st.set_page_config(page_title="FinSageAI – TrustPulse", layout="centered")

# -----------------------
st.title("💼 FinSageAI: TrustPulse Module")
st.markdown("""
Predict customer churn risk and visualize trust breakdowns using machine learning.
Upload a banking churn dataset to get started.
""")

# -----------------------
# Upload section
uploaded_file = st.file_uploader("📁 Upload Churn CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Show sample of data
        with st.expander("📄 Preview Raw Dataset"):
            st.write(df.head())

        # Preprocess
        X_train, X_test, y_train, y_test = load_and_preprocess(df)

        # Set threshold via slider
        threshold = st.slider("🎯 Set Churn Probability Threshold", 0.1, 0.9, 0.35, 0.05)

        # Train model
        model, results_df, report, accuracy, conf_matrix = train_trust_model(
            X_train, X_test, y_train, y_test, threshold
        )

        st.subheader("📊 Model Evaluation")
        st.write(f"**Accuracy:** `{round(accuracy * 100, 2)}%`")
        st.write("**Confusion Matrix:**")
        st.write(conf_matrix)
        st.write("**Classification Report:**")
        st.dataframe(pd.DataFrame(report).transpose())

        # Show trust score breakdown
        st.subheader("📈 Trust Risk Distribution")
        fig1, ax1 = plt.subplots()
        results_df["Trust_Score"].value_counts().plot(kind="bar", color="tomato", ax=ax1)
        plt.title("Trust Level Breakdown")
        plt.ylabel("Customers")
        st.pyplot(fig1)

        # Top risky customers
        st.subheader("🚨 Top 5 High-Risk Customers")
        top_risky = results_df.sort_values("Churn_Prob", ascending=False).head(5)
        st.dataframe(top_risky[["Churn_Prob", "Trust_Score"]])

        # Histogram of churn probabilities
        st.subheader("📉 Churn Probability Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(results_df["Churn_Prob"], bins=20, color="orange", edgecolor="black")
        ax2.axvline(results_df["Churn_Prob"].mean(), color="red", linestyle="--", label="Mean")
        ax2.set_title("Churn Probability Histogram")
        ax2.set_xlabel("Probability")
        ax2.set_ylabel("Count")
        ax2.legend()
        st.pyplot(fig2)

        # Feature importances
        st.subheader("🧠 Feature Importances")
        importances = model.feature_importances_
        features = model.feature_names_in_
        sorted_idx = importances.argsort()[::-1]

        fig3, ax3 = plt.subplots()
        sns.barplot(x=importances[sorted_idx], y=features[sorted_idx], ax=ax3, palette="viridis")
        ax3.set_title("Top Influential Features")
        st.pyplot(fig3)

        # Downloadable results (optional)
        st.download_button("📥 Download Full Results CSV", results_df.to_csv(index=False), "trustpulse_results.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
else:
    st.info("⬆️ Please upload a CSV file to begin.")

