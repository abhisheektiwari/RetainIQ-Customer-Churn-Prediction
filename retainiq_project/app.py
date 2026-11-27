import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from pathlib import Path
from generate_data import generate_customers
from src.model import prepare_target, train_models, get_feature_importance

st.set_page_config(page_title="RetainIQ | Churn Intelligence", page_icon="📈", layout="wide")

# Styling: dark analytics product look
st.markdown("""<style>
.stApp {background: linear-gradient(135deg,#090d1b 0%,#0c1225 65%,#11102a 100%); color:#edf1ff;}
[data-testid="stSidebar"] {background:#080c18; border-right:1px solid #232b44;}
[data-testid="stMetric"] {background:#141c32; border:1px solid #283454; padding:14px 16px; border-radius:14px;}
[data-testid="stMetricLabel"] {color:#9faed2;}
div[data-testid="stMetricValue"] {color:#eef3ff;}
section[data-testid="stMain"] div.block-container {padding-top:1.5rem;}
.card {background:#141c32; border:1px solid #283454; border-radius:15px; padding:18px;}
.small-note {color:#a6b2cc; font-size:0.92rem;}
</style>""", unsafe_allow_html=True)


def money(value):
    return f"${value:,.0f}"


def recommendations(row):
    acts=[]
    if row.get("contract") == "Month-to-month": acts.append("Offer long-term plan incentive")
    if row.get("monthly_charges", 0) > 90: acts.append("Review bill and offer a lower-cost plan")
    if row.get("support_tickets_3m", 0) >= 3: acts.append("Priority service recovery call")
    if row.get("tenure_months", 99) < 12: acts.append("New-customer onboarding check-in")
    if row.get("usage_drop_pct", 0) >= 25: acts.append("Usage re-engagement campaign")
    if not acts: acts.append("Loyalty benefit and satisfaction check-in")
    return " • ".join(acts[:2])


def load_data(upload):
    if upload is not None:
        try:
            df = pd.read_csv(upload)
            return prepare_target(df), "Uploaded CSV"
        except Exception as exc:
            st.sidebar.error(f"Couldn't read uploaded file: {exc}")
    return generate_customers(3000), "Demo synthetic data"



@st.cache_data(show_spinner=False)
def cached_train(df, model_choice, threshold):
    # Cache only when dataset contents and choices match.
    scored, metrics, model, features = train_models(df, model_choice, threshold)
    importance = get_feature_importance(model, features)
    return scored, metrics, model, features, importance



st.sidebar.markdown("## 🟣 RetainIQ")
st.sidebar.caption("Predict • Understand • Retain")
upload = st.sidebar.file_uploader("Upload your customer CSV", type=["csv"])
data, data_label = load_data(upload)

if "churn" not in data.columns:
    st.error("Dataset needs a churn column (Yes/No or 1/0). Please upload a compatible CSV.")
    st.stop()

if "customer_id" not in data.columns:
    data["customer_id"] = [f"CUSTOMER-{i+1:05d}" for i in range(len(data))]

model_choice = st.sidebar.selectbox("Scoring model", ["Random Forest", "Logistic Regression", "Gradient Boosting"], index=0)
threshold = st.sidebar.slider("Risk threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05, help="Used to classify customers as churn-risk for model evaluation; risk segments are separately shown by probability bands.")
page = st.sidebar.radio("Navigation", ["Overview", "EDA", "Model Insights", "Customer Risk", "Retention Actions", "SQL Insights", "About"])

with st.spinner("Training models and scoring customers..."):
    scored, model_metrics, trained_model, feature_cols, importance = cached_train(data, model_choice, threshold)

# Calculate risk bands for business display
scored["risk_level"] = pd.cut(scored["churn_probability"], bins=[-0.001, 0.35, 0.65, 1.001], labels=["Low", "Medium", "High"]).astype(str)
scored["retention_action"] = scored.apply(recommendations, axis=1)
actual_churn_rate = scored["churn"].mean()
customers = len(scored)
high_risk = scored[scored["risk_level"] == "High"]
revenue_risk = high_risk["monthly_charges"].sum() if "monthly_charges" in scored else 0

st.title("Customer Churn Prediction & Retention Intelligence")
st.markdown("*Turn data into action — predict churn, understand drivers, prioritize retention.*")
st.caption(f"Data source: {data_label}  •  {customers:,} customers  •  Risk segments are based on predicted probability bands.")

if page == "Overview":
    k1,k2,k3,k4=st.columns(4)
    k1.metric("Total Customers", f"{customers:,}")
    k2.metric("Actual Churn Rate", f"{actual_churn_rate:.1%}")
    k3.metric("High-risk Customers", f"{len(high_risk):,}", f"{len(high_risk)/max(customers,1):.1%} of customer base")
    k4.metric("Monthly Revenue at Risk", money(revenue_risk))
    left,right=st.columns([1.55,1])
    with left:
        st.subheader("Churn by Contract")
        if "contract" in scored:
            contract_stats=scored.groupby("contract",observed=False)["churn"].mean().reset_index().sort_values("churn",ascending=False)
            fig=px.bar(contract_stats,x="contract",y="churn",color="contract",text_auto=".1%",labels={"churn":"Churn rate","contract":"Contract type"})
            fig.update_layout(template="plotly_dark",showlegend=False,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
        st.subheader("Customer Risk Distribution")
        risk_counts=scored["risk_level"].value_counts().reindex(["Low","Medium","High"]).fillna(0).reset_index()
        risk_counts.columns=["Risk level","Customers"]
        fig=px.pie(risk_counts,names="Risk level",values="Customers",hole=.55,color="Risk level",color_discrete_map={"Low":"#4285f4","Medium":"#f59e0b","High":"#a855f7"})
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader("Top churn indicators")
        fig=px.bar(importance.head(8).sort_values("Importance"),x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale="Purples")
        fig.update_layout(template="plotly_dark",plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
        st.subheader("Model comparison")
        st.dataframe(model_metrics.style.format({c:"{:.3f}" for c in model_metrics.columns if c != "Model"}),use_container_width=True,hide_index=True)
    st.subheader("High-risk customer preview")
    show_cols=[c for c in ["customer_id","churn_probability","risk_level","contract","tenure_months","monthly_charges","support_tickets_3m","retention_action"] if c in scored.columns]
    st.dataframe(high_risk.sort_values("churn_probability",ascending=False)[show_cols].head(12).style.format({"churn_probability":"{:.1%}","monthly_charges":"${:,.2f}"}),use_container_width=True,hide_index=True)

elif page == "EDA":
    st.header("Exploratory Data Analysis")
    st.write("Explore customer behavior and churn patterns before modeling.")
    numeric=scored.select_dtypes(include="number").columns.tolist()
    categorical=[c for c in scored.columns if c not in numeric and c not in ["customer_id"]]
    a,b=st.columns(2)
    with a:
        x_col=st.selectbox("Numeric feature",[c for c in numeric if c not in ["churn","churn_probability"]],index=0)
        fig=px.hist(scored,x=x_col,color="churn",barmode="overlay",opacity=.75,labels={"churn":"Churn (0=No,1=Yes)"})
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    with b:
        cat_col=st.selectbox("Category feature",categorical,index=0 if categorical else None)
        if cat_col:
            grp=scored.groupby(cat_col,observed=False)["churn"].mean().reset_index().sort_values("churn",ascending=False)
            fig=px.bar(grp,x=cat_col,y="churn",color="churn",color_continuous_scale="Purples",labels={"churn":"Churn rate"},text_auto=".1%")
            fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
    st.subheader("Numeric correlations")
    corr=scored[numeric].corr(numeric_only=True)
    fig=px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",aspect="auto")
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    st.subheader("Dataset sample")
    st.dataframe(data.head(30),use_container_width=True)

elif page == "Model Insights":
    st.header("Model Insights & Evaluation")
    st.info("Evaluation is performed on a stratified 25% holdout set. For final scoring, the selected model is refit on all labeled rows. Compare recall and ROC-AUC—not accuracy alone—because churn is often imbalanced.")
    st.dataframe(model_metrics.style.format({c:"{:.3f}" for c in model_metrics.columns if c != "Model"}),use_container_width=True,hide_index=True)
    st.subheader(f"Feature importance — {model_choice}")
    fig=px.bar(importance.sort_values("Importance"),x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale="Viridis")
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("**Metric guide:** Precision = predicted-risk customers who actually churned; Recall = churners caught; F1 balances precision/recall; ROC-AUC measures ranking ability across thresholds.")

elif page == "Customer Risk":
    st.header("Customer Risk Explorer")
    min_prob=st.slider("Minimum churn probability",0.0,1.0,0.35,0.05)
    risk_filter=st.multiselect("Risk segment",["Low","Medium","High"],default=["Medium","High"])
    subset=scored[(scored["churn_probability"]>=min_prob)&(scored["risk_level"].isin(risk_filter))].copy()
    st.metric("Customers matching filters",f"{len(subset):,}")
    table_cols=[c for c in ["customer_id","churn_probability","risk_level","contract","tenure_months","monthly_charges","support_tickets_3m","usage_drop_pct","satisfaction_score","retention_action"] if c in subset.columns]
    st.dataframe(subset.sort_values("churn_probability",ascending=False)[table_cols].style.format({"churn_probability":"{:.1%}","monthly_charges":"${:,.2f}"}),use_container_width=True,hide_index=True)
    st.download_button("Download filtered risk customers CSV",subset.to_csv(index=False).encode("utf-8"),"customer_risk_list.csv","text/csv")

elif page == "Retention Actions":
    st.header("Retention Strategy Recommendations")
    st.write("Recommendations are rule-based examples. Validate them with experiments, campaign cost, customer consent, and real business outcomes before deploying.")
    high=high_risk.copy()
    if high.empty:
        st.success("No customers fall into the High risk band for the current model and data.")
    else:
        actions=high["retention_action"].value_counts().reset_index()
        actions.columns=["Suggested action","Customers"]
        fig=px.bar(actions,x="Customers",y="Suggested action",orientation="h",color="Customers",color_continuous_scale="Purples")
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
        st.subheader("Actionable customer list")
        view_cols=[c for c in ["customer_id","churn_probability","monthly_charges","contract","support_tickets_3m","retention_action"] if c in high.columns]
        st.dataframe(high.sort_values("churn_probability",ascending=False)[view_cols].style.format({"churn_probability":"{:.1%}","monthly_charges":"${:,.2f}"}),use_container_width=True,hide_index=True)
    st.subheader("Illustrative campaign budget simulator")
    discount_pct=st.slider("Discount as % of monthly bill",5,30,15,1)
    expected_save=st.slider("Assumed save rate for contacted high-risk customers",5,60,20,5)
    if "monthly_charges" in high.columns:
        contacted=len(high)
        campaign_cost=high["monthly_charges"].sum()*discount_pct/100
        expected_saved=contacted*expected_save/100
        retained_revenue=high["monthly_charges"].sum()*expected_save/100
        c1,c2,c3=st.columns(3)
        c1.metric("Campaign cost (illustrative)",money(campaign_cost))
        c2.metric("Expected customers saved",f"{expected_saved:,.0f}")
        c3.metric("Illustrative retained monthly revenue",money(retained_revenue))
        st.caption("This is a what-if calculation—not an uplift prediction. Validate save rate and offer costs using a controlled test.")

elif page == "SQL Insights":
    st.header("SQL-Powered Business Insights")
    st.write("A temporary SQLite database is built from the scored data. Try a query below, or use a preset.")
    conn=sqlite3.connect(":memory:")
    scored.to_sql("customers",conn,index=False,if_exists="replace")
    presets={
        "Churn rate by contract":"SELECT contract, COUNT(*) AS customers, ROUND(AVG(churn)*100,2) AS churn_pct FROM customers GROUP BY contract ORDER BY churn_pct DESC;",
        "Revenue by risk segment":"SELECT risk_level, COUNT(*) AS customers, ROUND(SUM(monthly_charges),2) AS monthly_revenue FROM customers GROUP BY risk_level ORDER BY monthly_revenue DESC;",
        "Top 10 highest risk customers":"SELECT customer_id, ROUND(churn_probability*100,1) AS risk_pct, contract, tenure_months, monthly_charges FROM customers ORDER BY churn_probability DESC LIMIT 10;",
        "Churn by payment method":"SELECT payment_method, COUNT(*) AS customers, ROUND(AVG(churn)*100,2) AS churn_pct FROM customers GROUP BY payment_method ORDER BY churn_pct DESC;",
    }
    selected=st.selectbox("Preset query",list(presets.keys()))
    sql=st.text_area("SQL query",value=presets[selected],height=110)
    if st.button("Run SQL",type="primary"):
        try:
            result=pd.read_sql_query(sql,conn)
            st.dataframe(result,use_container_width=True)
        except Exception as e:
            st.error(f"SQL error: {e}")
    st.caption("SQL access is limited to this in-memory dataset for the current session.")
    conn.close()

else:
    st.header("About RetainIQ")
    st.markdown("**RetainIQ** is a portfolio-grade learning demo combining customer churn prediction, exploratory analysis, customer risk segmentation, feature importance, SQL insights, and rule-based retention recommendations.")
    st.markdown("**Dataset:** synthetic telecom customer records generated locally. It is not real customer data and does not represent a company's actual churn outcomes.")
    st.markdown("**Limitations:** risk scores trained on synthetic patterns; recommendations are illustrative; no causal inference or actual campaign uplift is claimed.")
    st.markdown("**Suggested next improvements:** replace synthetic data with an approved real dataset; calibrate probabilities; add SHAP explanations; track retention experiments; deploy with authentication and privacy controls.")

st.sidebar.markdown("---")
st.sidebar.caption("RetainIQ • Portfolio Demo | Synthetic dataset")
