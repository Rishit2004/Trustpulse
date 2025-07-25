🚀 FinSageAI – TrustPulse Module

🎯 Predict Customer Churn Risk with AI
TrustPulse is a part of the FinSageAI suite — a modular ML system for financial behavior analytics. This tool predicts customer churn risk using a machine learning pipeline and visualizes trust breakdowns in an interactive dashboard.

🌐 🔗 Live App
Upload your own dataset and get instant churn risk insights, trust scores, and feature impact visualizations — all in the browser.

📊 Features
📁 Upload real-world churn datasets (Churn_Modelling.csv format)
🤖 Train a Random Forest classifier to predict churn

🎯 Adjust prediction thresholds to balance precision/recall
📈 Visualize:
Churn probability distribution
Trust risk levels (High / Medium / Low)
Top 5 high-risk customers
Feature importance (what drives churn?)
📥 Download full results as CSV

🛠️ Tech Stack
Streamlit – frontend UI
scikit-learn – ML model
[Pandas / NumPy / Matplotlib / Seaborn] – data handling & plots

📦 Installation (Local)
git clone https://github.com/your-username/trustpulse.git
cd trustpulse
pip install -r requirements.txt
streamlit run app.py


📁 Folder Structure
trustpulse/
├── app.py                # Streamlit dashboard
├── trustpulse.py         # ML logic and utilities
├── requirements.txt      # Python dependencies
└── sample_data/
    └── Churn_Modelling.csv
🔄 Coming Soon
🔍 SHAP-based model explainability

🧭 Additional modules: FraudShift, PersonaMap, StressSim

🌐 Multi-page Streamlit interface

👤 Author
Rishit Sharma
GitHub | LinkedIn (Update these if needed)

📄 License
MIT License – feel free to use, modify, and share this project!