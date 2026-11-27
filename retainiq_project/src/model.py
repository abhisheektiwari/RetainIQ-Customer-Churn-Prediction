"""Reusable preprocessing, model training and scoring utilities for RetainIQ."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

TARGET = "churn"
ID_COLS = ["customer_id"]


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET not in df.columns:
        raise ValueError("Dataset must contain a 'churn' column (Yes/No or 1/0).")
    vals = df[TARGET].astype(str).str.strip().str.lower()
    df[TARGET] = vals.map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0, "1.0": 1, "0.0": 0})
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    return df



def build_pipeline(X: pd.DataFrame, model_name: str):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=220, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.06, max_depth=2, random_state=42),
    }
    return Pipeline([("preprocess", pre), ("model", models[model_name])])


def train_models(df: pd.DataFrame, model_choice="Random Forest", threshold=0.5):
    df = prepare_target(df)
    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    X, y = df[feature_cols], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    results = []
    fitted = {}
    for name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
        pipe = build_pipeline(X_train, name)
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= threshold).astype(int)
        results.append({"Model": name, "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, zero_division=0),
                       "Recall": recall_score(y_test, pred, zero_division=0), "F1": f1_score(y_test, pred, zero_division=0),
                       "ROC-AUC": roc_auc_score(y_test, prob)})
        fitted[name] = pipe
    chosen = fitted[model_choice]
    chosen.fit(X, y)  # refit on all labeled rows for deployment-style scoring
    predictions = chosen.predict_proba(X)[:, 1]
    scored = df.copy()
    scored["churn_probability"] = predictions
    scored["risk_level"] = pd.cut(predictions, bins=[-0.001, 0.35, 0.65, 1.001], labels=["Low", "Medium", "High"])
    scored["risk_level"] = scored["risk_level"].astype(str)
    return scored, pd.DataFrame(results).sort_values("ROC-AUC", ascending=False), chosen, feature_cols



def get_feature_importance(pipe, feature_cols):
    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    names = pre.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_[0])
    else:
        return pd.DataFrame({"Feature": feature_cols, "Importance": 0}).sort_values("Importance", ascending=False)
    out = pd.DataFrame({"Feature": names, "Importance": values})
    out["Feature"] = out["Feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    return out.sort_values("Importance", ascending=False).head(15)
