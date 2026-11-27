# RetainIQ — Customer Churn Prediction & Retention Intelligence

A beginner-friendly portfolio project that combines predictive analytics and business decision support. It predicts churn probability, compares ML models, segments customers by risk, explores churn drivers, estimates revenue at risk, and proposes example retention actions.


## Features
- Overview dashboard with churn KPIs, risk distribution, contract churn, model comparison, feature importance, and high-risk customer preview
- EDA: numeric distributions, category churn rates, and correlation heatmap
- Models: Logistic Regression, Random Forest, Gradient Boosting (stratified holdout metrics)
- Customer Risk Explorer with filters and downloadable risk list
- Rule-based retention action suggestions
- Campaign what-if simulator (illustrative assumptions, not an uplift model)
- SQL Insights page using an in-memory SQLite table
- Upload your own CSV to replace the demo dataset

## Quick start in VS Code
1. Install Python 3.10+.
2. Extract the project zip and open the `retainiq_project` folder in VS Code.
3. Open Terminal in VS Code and create a virtual environment:

   **Windows:**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt terminal:
   ```bat
   .venv\Scripts\activate.bat
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```
6. Streamlit opens in your browser, usually at `http://localhost:8501`.

## Demo dataset
`data/telco_customers_synthetic.csv` is a generated, fictional dataset of 3,000 telecom customers. The data is synthetic and is only suitable for demonstration and learning—not real business claims.

To regenerate it:
```bash
python generate_data.py
```

## CSV upload requirements
Your uploaded CSV should include a target column named `churn` (values `Yes`/`No`, `1`/`0`, or `True`/`False`). If it has no `customer_id`, the app generates IDs for display. The model treats all other columns (except ID and target) as input features. Use a clean, tabular dataset and avoid uploading sensitive personal data.


## Project structure
```
retainiq_project/
├── app.py
├── generate_data.py
├── requirements.txt
├── data/
│   └── telco_customers_synthetic.csv
└── src/
    └── model.py
```

## Notes about interpretation
- Holdout metrics are calculated on a stratified 25% split. The chosen model is refit on all labeled rows for risk scoring.
- Risk bands (Low < 35%, Medium 35–65%, High > 65%) are illustrative and can be adjusted for business capacity and validated costs.
- Feature importance indicates model reliance, not causation. To improve explainability, add SHAP values.
- Retention suggestions are simple rules. Validate offer costs, contact strategy, customer consent, and actual outcomes with controlled tests.
- Campaign simulator values are assumptions and not predictive estimates of incremental impact.
