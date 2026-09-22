# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.inspection import permutation_importance

# Page config
st.set_page_config(
    page_title="Student Risk Classifier, CPUT Mech Eng",
    page_icon=":mortar_board:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header{font-size:2.2rem;font-weight:700;color:#1a3a5c;text-align:center;margin-bottom:.3rem}
.sub-header{font-size:1.1rem;color:#4a6a8c;text-align:center;margin-bottom:1.5rem}
.kpi-card{background:#f8f9fa;border-radius:10px;padding:1rem;box-shadow:0 2px 4px rgba(0,0,0,.08);text-align:center}
.kpi-val{font-size:2rem;font-weight:700;color:#1a3a5c}
.kpi-lbl{font-size:.85rem;color:#6c757d;margin-top:.2rem}
.at-risk-badge{background:#dc3545;color:white;padding:4px 12px;border-radius:20px;font-weight:600}
.safe-badge{background:#28a745;color:white;padding:4px 12px;border-radius:20px;font-weight:600}
</style>
""", unsafe_allow_html=True)

# Constants
MODULES = [
    "COMPUTER SKILLS AND APPLICATIONS",
    "ENGINEERING DRAWING 1",
    "ENGINEERING MATHEMATICS",
    "ENGINEERING MECHANICS",
    "MECHANICAL ENGINEERING MANUFACTURING",
    "PHYSICAL SCIENCE",
    "PROFESSIONAL COMMUNICATION",
]
SHORT = {
    "COMPUTER SKILLS AND APPLICATIONS":    "Comp Skills",
    "ENGINEERING DRAWING 1":               "Eng Drawing",
    "ENGINEERING MATHEMATICS":             "Eng Maths",
    "ENGINEERING MECHANICS":               "Eng Mechanics",
    "MECHANICAL ENGINEERING MANUFACTURING":"Mech Manuf",
    "PHYSICAL SCIENCE":                    "Phys Science",
    "PROFESSIONAL COMMUNICATION":          "Prof Comm",
}
PASS_MARK = 50

DROP_COLS = [
    "Birthdate","Unnamed: 0","Formal Time","Average Marks",
    "Faculty","Department Name","Qualification Code","Qualification Name",
    "Offering Type","Campus Name","Period Of Study","Qualification Type",
    "Post Graduate","RESIDENCE_NAME","Matric Type",
    "Secondary School Name","Marital Status",
]
ORDINAL_COLS = ["Gender","Res_status","Funded"]
NOMINAL_COLS = [
    "Language Name","Ethnic Group Name","Country Name",
    "Previous Activity Desc","Student Type Description","FUNDNAME",
]

# Data & model caching
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df = pd.read_excel("Enrol_model.xlsx")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    if "Funded" not in df.columns and "Bursary Type" in df.columns:
        df["Funded"] = df["Bursary Type"].fillna("Self-Funded").apply(
            lambda x: "No" if x == "Self-Funded" else "Yes"
        )
    for col, fill in [("FUNDNAME","Self-Funded"),("Res_status","No")]:
        if col in df.columns:
            df[col] = df[col].fillna(fill)

    # Keep only students registered for all seven modules
    existing_modules = [m for m in MODULES if m in df.columns]
    student_module_counts = df[existing_modules].notna().sum(axis=1)
    df = df[student_module_counts == len(existing_modules)].reset_index(drop=True)

    return df


@st.cache_resource(show_spinner="Training models, takes about 30 s on first load...")
def train_models():
    df = load_data().copy()
    df["At_Risk"] = df["At_Risk"].map({"Yes":1,"No":0})
    X = df.drop(columns=["At_Risk"])
    y = df["At_Risk"].dropna()
    X = X.loc[y.index]

    num_cols  = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    ord_cols  = [c for c in ORDINAL_COLS  if c in X.columns]
    nom_cols  = [c for c in NOMINAL_COLS  if c in X.columns]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    svm_pre = ColumnTransformer([
        ("num", Pipeline([("i",SimpleImputer(strategy="median")),
                          ("s",StandardScaler())]), num_cols),
        ("ord", Pipeline([("i",SimpleImputer(strategy="most_frequent")),
                          ("e",OrdinalEncoder(handle_unknown="use_encoded_value",
                                              unknown_value=-1))]), ord_cols),
        ("nom", Pipeline([("i",SimpleImputer(strategy="most_frequent")),
                          ("e",OneHotEncoder(handle_unknown="ignore",
                                             sparse_output=False))]), nom_cols),
    ])
    rf_pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("ord", Pipeline([("i",SimpleImputer(strategy="most_frequent")),
                          ("e",OrdinalEncoder(handle_unknown="use_encoded_value",
                                              unknown_value=-1))]), ord_cols),
        ("nom", Pipeline([("i",SimpleImputer(strategy="most_frequent")),
                          ("e",OneHotEncoder(handle_unknown="ignore",
                                             sparse_output=False))]), nom_cols),
    ])

    svm = Pipeline([("pre",svm_pre),
                    ("clf",SVC(kernel="rbf",probability=True,
                               class_weight="balanced",random_state=42,
                               C=1.0,gamma="scale"))])
    rf  = Pipeline([("pre",rf_pre),
                    ("clf",RandomForestClassifier(n_estimators=200,max_depth=10,
                        min_samples_split=10,min_samples_leaf=5,
                        class_weight="balanced",random_state=42,n_jobs=-1))])
    svm.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)

    y_pred_svm  = svm.predict(X_te)
    y_prob_svm  = svm.predict_proba(X_te)[:,1]
    y_pred_rf   = rf.predict(X_te)
    y_prob_rf   = rf.predict_proba(X_te)[:,1]

    # RF feature importance
    raw_names = rf.named_steps["pre"].get_feature_names_out()
    raw_imp   = rf.named_steps["clf"].feature_importances_
    agg = {}
    for fname, imp in zip(raw_names, raw_imp):
        if "__" in fname:
            prefix, rest = fname.split("__",1)
            orig = rest.rsplit("_",1)[0] if prefix == "nom" and "_" in rest else rest
        else:
            orig = fname
        agg[orig] = agg.get(orig, 0) + imp
    rf_imp_df = (pd.DataFrame({"Feature":list(agg),"Importance":list(agg.values())})
                 .sort_values("Importance",ascending=False).reset_index(drop=True))

    return dict(
        svm=svm, rf=rf,
        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
        y_pred_svm=y_pred_svm, y_prob_svm=y_prob_svm,
        y_pred_rf=y_pred_rf,  y_prob_rf=y_prob_rf,
        rf_imp_df=rf_imp_df,
        num_cols=num_cols, ord_cols=ord_cols, nom_cols=nom_cols,
        feature_cols=list(X.columns),
    )

# Sidebar
with st.sidebar:
    st.markdown("### CPUT: Mech Eng Risk")
    page = st.radio("Navigate to", [
        "Overview",
        "Exploratory Analysis",
        "Module Analysis",
        "Model Performance",
        "Feature Importance",
        "Predict a Student",
    ])
    st.markdown("---")
    st.caption("SVM (primary) + Random Forest\nDiploma Mech Eng, 2022 to 2025, CPUT")

# Load
try:
    df = load_data()
    data_ok = True
except Exception as e:
    st.error(f"Could not load Enrol_model.xlsx: {e}")
    data_ok = False
    st.stop()

# PAGE: OVERVIEW
if page == "Overview":
    st.markdown('<div class="main-header">Student Academic Risk Classifier</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Diploma in Mechanical Engineering, CPUT, 2022 to 2025</div>',
                unsafe_allow_html=True)

    n_total    = len(df)
    n_atrisk   = (df["At_Risk"] == "Yes").sum()
    n_safe     = (df["At_Risk"] == "No").sum()
    pct_atrisk = n_atrisk / n_total * 100

    c1,c2,c3,c4 = st.columns(4)
    for col, val, lbl, color in [
        (c1, n_total,           "Total students",       "#1a3a5c"),
        (c2, n_atrisk,          "At-risk students",     "#dc3545"),
        (c3, n_safe,            "Not at-risk students", "#28a745"),
        (c4, f"{pct_atrisk:.1f}%", "At-risk rate",     "#e67e22"),
    ]:
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-val" style="color:{color}">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        fig,ax = plt.subplots(figsize=(7,4))
        counts = df["At_Risk"].value_counts()
        ax.bar(counts.index, counts.values, color=["#2ecc71","#e74c3c"])
        ax.set_title("At-Risk Distribution", fontsize=13, fontweight="bold")
        ax.set_ylabel("Count")
        for i,v in enumerate(counts.values):
            ax.text(i, v+8, f"{v} ({v/n_total*100:.1f}%)", ha="center", fontweight="bold")
        st.pyplot(fig); plt.close()

    with col2:
        fig,ax = plt.subplots(figsize=(7,4))
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
               colors=["#2ecc71","#e74c3c"], explode=(0,0.07), startangle=90)
        ax.set_title("At-Risk Percentage", fontsize=13, fontweight="bold")
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("At-Risk Rate by Academic Year")
    yr = pd.crosstab(df["Academic Year"].astype(str), df["At_Risk"], normalize="index")*100
    fig,ax = plt.subplots(figsize=(10,5))
    yr.plot(kind="bar", stacked=False, color=["#2ecc71","#e74c3c"], ax=ax, rot=0)
    ax.set_xlabel("Academic Year"); ax.set_ylabel("Percentage (%)")
    ax.set_title("At-Risk Status by Academic Year", fontweight="bold")
    ax.legend(title="At-Risk", bbox_to_anchor=(1.01,1))
    for container, status in zip(ax.containers, yr.columns):
        for bar, pct in zip(container, yr[status]):
            if pct > 0:
                ax.annotate(f"{pct:.1f}%",
                            (bar.get_x() + bar.get_width()/2, bar.get_height()),
                            ha="center", va="bottom", fontsize=8,
                            xytext=(0,2), textcoords="offset points")
    ax.set_ylim(0, 110)
    for i, year in enumerate(yr.index):
        n = (df["Academic Year"].astype(str)==year).sum()
        ax.text(i, 105, f"n={n}", ha="center", fontsize=9)
    st.pyplot(fig); plt.close()

# PAGE: EXPLORATORY ANALYSIS
elif page == "Exploratory Analysis":
    st.markdown('<div class="main-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    st.subheader("Missing values, module marks only")
    miss = df[MODULES].isnull().sum().rename("Missing")
    miss_pct = (miss / len(df) * 100).rename("Missing (%)")
    miss_df = pd.concat([miss, miss_pct], axis=1).sort_values("Missing (%)", ascending=False)
    miss_df.index = [SHORT.get(m,m) for m in miss_df.index]
    fig,ax = plt.subplots(figsize=(10,4))
    ax.bar(miss_df.index, miss_df["Missing (%)"], color="#e67e22")
    ax.set_ylabel("Missing (%)")
    ax.set_title("Missing Mark Records by Module", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    for i,(idx,row) in enumerate(miss_df.iterrows()):
        ax.text(i, row["Missing (%)"]+0.05, f"{row['Missing (%)']:.1f}%",
                ha="center", fontsize=8)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Correlation heatmap, numeric features")
    num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    fig,ax = plt.subplots(figsize=(13,9))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                annot=True, fmt=".2f", square=True, linewidths=.4, ax=ax)
    ax.set_title("Correlation Heatmap, Numerical Variables", fontsize=14, fontweight="bold")
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Demographics by At-Risk status")
    demo_cols = [c for c in ["Gender","Ethnic Group Name","School Quintile",
                              "Funded","Res_status","Country Name"] if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    for i, col in enumerate(demo_cols[:6]):
        tab = pd.crosstab(df[col], df["At_Risk"], normalize="index")*100
        tab.plot(kind="bar", stacked=False, color=["#2ecc71","#e74c3c"],
                 ax=axes[i], rot=30)
        axes[i].set_title(f"At-Risk by {col}", fontsize=10, fontweight="bold")
        axes[i].set_ylabel("Percentage (%)"); axes[i].set_xlabel("")
        axes[i].legend(title="At-Risk", fontsize=8)
    plt.subplots_adjust(hspace=0.55, wspace=0.4)
    st.pyplot(fig); plt.close()

# PAGE: MODULE ANALYSIS
elif page == "Module Analysis":
    st.markdown('<div class="main-header">Module-Level Failure Analysis</div>', unsafe_allow_html=True)

    # Overall failure rates
    st.subheader("Overall failure rates (2022 to 2025)")
    rows = []
    for m in MODULES:
        valid  = df[m].dropna()
        n_fail = (valid < PASS_MARK).sum()
        rows.append({"Module": SHORT[m], "N": len(valid),
                     "Failed": n_fail,
                     "Failure rate (%)": round(n_fail/len(valid)*100,1) if len(valid) else 0})
    fr_df = pd.DataFrame(rows).sort_values("Failure rate (%)", ascending=False)

    fig, ax = plt.subplots(figsize=(10,5))
    colors = ["#dc3545" if r>=30 else "#3498db" for r in fr_df["Failure rate (%)"]]
    bars = ax.barh(fr_df["Module"], fr_df["Failure rate (%)"], color=colors)
    ax.axvline(30, color="black", linestyle="--", linewidth=1, label="30% threshold")
    ax.set_xlabel("Failure rate (%)")
    ax.set_title("Module Failure Rates, 2022 to 2025", fontweight="bold")
    ax.legend()
    for bar, val in zip(bars, fr_df["Failure rate (%)"]):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)
    st.pyplot(fig); plt.close()

    # Year-on-year heatmap
    st.markdown("---")
    st.subheader("Year-on-year failure rate trends")
    years = sorted(df["Academic Year"].unique())
    pivot_data = {}
    for yr in years:
        ydf = df[df["Academic Year"]==yr]
        pivot_data[yr] = {SHORT[m]: round((ydf[m].dropna()<PASS_MARK).mean()*100,1)
                          for m in MODULES if m in df.columns}
    pivot = pd.DataFrame(pivot_data).loc[[SHORT[m] for m in MODULES]]

    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", linewidths=.4,
                vmin=0, vmax=60, ax=ax, cbar_kws={"label":"Failure rate (%)"})
    ax.set_title("Module Failure Rates (%) by Academic Year", fontweight="bold")
    st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)
    with col1:
        fig,ax = plt.subplots(figsize=(8,5))
        for m in MODULES:
            vals = [pivot_data.get(yr,{}).get(SHORT[m],None) for yr in years]
            ax.plot([str(y) for y in years], vals, marker="o",
                    label=SHORT[m], linewidth=1.8)
        ax.set_xlabel("Academic Year"); ax.set_ylabel("Failure rate (%)")
        ax.set_title("Failure Rate Trends per Module", fontweight="bold")
        ax.legend(fontsize=7, bbox_to_anchor=(1.01,1))
        ax.axhline(30, color="black", linestyle="--", linewidth=.8, alpha=.6)
        st.pyplot(fig); plt.close()

    with col2:
        st.dataframe(pivot.style.background_gradient(cmap="Reds",vmin=0,vmax=60)
                     .format("{:.1f}"), use_container_width=True)

    # Mark distributions
    st.markdown("---")
    st.subheader("Mark distributions by at-risk status")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, m in enumerate(MODULES):
        ax = axes[i]
        data_no  = df[df["At_Risk"]=="No"][m].dropna()
        data_yes = df[df["At_Risk"]=="Yes"][m].dropna()
        bp = ax.boxplot([data_no, data_yes], positions=[0,1], widths=0.5,
                        patch_artist=True,
                        medianprops=dict(color="black",linewidth=2))
        bp["boxes"][0].set_facecolor("#2ca02c"); bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#d62728"); bp["boxes"][1].set_alpha(0.7)
        ax.set_xticks([0,1]); ax.set_xticklabels(["Not At-Risk","At-Risk"], fontsize=8)
        ax.set_title(SHORT[m], fontsize=9, fontweight="bold")
        ax.axhline(PASS_MARK, color="black", linestyle="--", linewidth=.8)
        ax.set_ylim(0,105)
    axes[-1].axis("off")
    from matplotlib.patches import Patch
    axes[-1].legend(handles=[Patch(facecolor="#2ca02c",alpha=.7,label="Not At-Risk"),
                              Patch(facecolor="#d62728",alpha=.7,label="At-Risk")],
                    loc="center", fontsize=11, title="At-Risk Status", frameon=True)
    plt.suptitle("Mark Distributions: At-Risk vs Not At-Risk", fontsize=13, fontweight="bold")
    plt.subplots_adjust(hspace=0.45, wspace=0.35, top=0.92)
    st.pyplot(fig); plt.close()

    # Co-failure heatmap
    st.markdown("---")
    st.subheader("Co-failure analysis, modules failed together")
    fail_mat = (df[MODULES] < PASS_MARK).fillna(0).astype(float)
    fail_mat.columns = [SHORT[m] for m in MODULES]
    cofail = pd.DataFrame(0.0, index=fail_mat.columns, columns=fail_mat.columns)
    for m1,m2 in combinations(fail_mat.columns, 2):
        both   = ((fail_mat[m1]==1)&(fail_mat[m2]==1)).sum()
        either = ((fail_mat[m1]==1)|(fail_mat[m2]==1)).sum()
        pct    = round(both/either*100,1) if either>0 else 0
        cofail.loc[m1,m2] = pct; cofail.loc[m2,m1] = pct
    arr = cofail.values.copy(); np.fill_diagonal(arr, np.nan)
    cofail_plot = pd.DataFrame(arr, index=cofail.index, columns=cofail.columns)

    fig,ax = plt.subplots(figsize=(9,7))
    sns.heatmap(cofail_plot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=.4, ax=ax, mask=np.eye(len(cofail_plot),dtype=bool),
                cbar_kws={"label":"% co-failing"})
    ax.set_title("Co-failure Matrix, Module Pairs Failed Together (%)", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig); plt.close()

    # Bottleneck
    st.markdown("---")
    st.subheader("Bottleneck identification")
    btn_rows = []
    for m in MODULES:
        failed = df[df[m].notna() & (df[m]<PASS_MARK)]
        n_f = len(failed)
        if n_f == 0: continue
        other_fails = (failed[MODULES].lt(PASS_MARK).sum(axis=1) - 1).clip(lower=0)
        sole_pct    = (other_fails==0).mean()*100
        avg_other   = other_fails.mean()
        btn_rows.append({"Module": SHORT[m], "N failed": n_f,
                         "Avg other modules failed": round(avg_other,2),
                         "Sole failure (%)": round(sole_pct,1),
                         "Bottleneck score": round(n_f * avg_other, 2)})
    btn_df = pd.DataFrame(btn_rows).sort_values("Bottleneck score",ascending=False)

    col1,col2 = st.columns(2)
    with col1:
        fig,ax = plt.subplots(figsize=(8,5))
        colors = ["#dc3545" if s>=70 else "#3498db" for s in btn_df["Bottleneck score"]]
        bars = ax.barh(btn_df["Module"][::-1], btn_df["Bottleneck score"][::-1], color=colors[::-1])
        ax.set_xlabel("Composite Bottleneck Score")
        ax.set_title("Module Bottleneck Ranking", fontweight="bold")
        for bar, val in zip(bars, btn_df["Bottleneck score"][::-1]):
            ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                    f"{val:.1f}", va="center", fontsize=9)
        st.pyplot(fig); plt.close()

    with col2:
        st.dataframe(btn_df[["Module","N failed","Avg other modules failed",
                              "Sole failure (%)","Bottleneck score"]],
                     use_container_width=True, hide_index=True)

# PAGE: MODEL PERFORMANCE
elif page == "Model Performance":
    st.markdown('<div class="main-header">Model Performance</div>', unsafe_allow_html=True)

    with st.spinner("Training models on your data..."):
        m = train_models()

    y_te      = m["y_te"]
    y_pred_sv = m["y_pred_svm"];  y_prob_sv = m["y_prob_svm"]
    y_pred_rf = m["y_pred_rf"];   y_prob_rf = m["y_prob_rf"]

    # Summary table
    results_tbl = pd.DataFrame({
        "Model":     ["SVM","Random Forest"],
        "Accuracy":  [accuracy_score(y_te,y_pred_sv),  accuracy_score(y_te,y_pred_rf)],
        "Precision": [precision_score(y_te,y_pred_sv), precision_score(y_te,y_pred_rf)],
        "Recall":    [recall_score(y_te,y_pred_sv),    recall_score(y_te,y_pred_rf)],
        "F1-score":  [f1_score(y_te,y_pred_sv),        f1_score(y_te,y_pred_rf)],
        "ROC-AUC":   [roc_auc_score(y_te,y_prob_sv),   roc_auc_score(y_te,y_prob_rf)],
    }).round(4)
    st.dataframe(results_tbl.set_index("Model"), use_container_width=True)

    # Confusion matrices
    st.markdown("---")
    st.subheader("Confusion Matrices")
    col1, col2 = st.columns(2)
    for col, y_pred, cmap, title in [
        (col1, y_pred_sv, "Blues",  "SVM"),
        (col2, y_pred_rf, "Greens", "Random Forest"),
    ]:
        cm = confusion_matrix(y_te, y_pred)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                    xticklabels=["Not At-Risk","At-Risk"],
                    yticklabels=["Not At-Risk","At-Risk"], ax=ax)
        ax.set_title(f"{title}: Confusion Matrix", fontweight="bold")
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        col.pyplot(fig); plt.close()

    # ROC curves
    st.markdown("---")
    st.subheader("ROC Curves")
    fig, ax = plt.subplots(figsize=(8,6))
    for y_prob, label, color in [
        (y_prob_sv, "SVM",           "#007bff"),
        (y_prob_rf, "Random Forest", "#28a745"),
    ]:
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        auc = roc_auc_score(y_te, y_prob)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", linewidth=2, color=color)
    ax.plot([0,1],[0,1],"k--", label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: SVM vs Random Forest", fontweight="bold")
    ax.legend(); ax.grid(alpha=.3)
    st.pyplot(fig); plt.close()

    # Classification reports
    st.markdown("---")
    st.subheader("Classification Reports")
    col1, col2 = st.columns(2)
    col1.text("SVM\n" + classification_report(y_te, y_pred_sv,
              target_names=["Not At-Risk","At-Risk"]))
    col2.text("Random Forest\n" + classification_report(y_te, y_pred_rf,
              target_names=["Not At-Risk","At-Risk"]))

# PAGE: FEATURE IMPORTANCE
elif page == "Feature Importance":
    st.markdown('<div class="main-header">Feature Importance</div>', unsafe_allow_html=True)

    with st.spinner("Training models..."):
        m = train_models()

    rf_imp = m["rf_imp_df"]

    # All features
    st.subheader("Random Forest: all features (Gini importance)")
    top20 = rf_imp.head(20).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(10,7))
    colors = ["#dc3545" if f in MODULES else "#3498db" for f in top20["Feature"]]
    ax.barh(top20["Feature"], top20["Importance"], color=colors)
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Features: Random Forest", fontweight="bold")
    red_p  = mpatches.Patch(color="#dc3545", label="Module mark")
    blue_p = mpatches.Patch(color="#3498db", label="Non-mark feature")
    ax.legend(handles=[red_p, blue_p])
    st.pyplot(fig); plt.close()

    # Module marks only
    st.markdown("---")
    st.subheader("Module marks: importance ranking")
    mod_imp = rf_imp[rf_imp["Feature"].isin(MODULES)].sort_values("Importance")
    mod_imp["Short"] = mod_imp["Feature"].map(SHORT)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.barh(mod_imp["Short"], mod_imp["Importance"], color="#dc3545", alpha=0.8)
    ax.set_xlabel("Importance")
    ax.set_title("Module Feature Importance (RF Gini)", fontweight="bold")
    for i, (_, row) in enumerate(mod_imp.iterrows()):
        ax.text(row["Importance"]+0.001, i, f"{row['Importance']:.4f}", va="center", fontsize=9)
    st.pyplot(fig); plt.close()

    # Non-mark features
    st.markdown("---")
    st.subheader("Non-mark features: what else predicts risk?")
    non_mod = rf_imp[~rf_imp["Feature"].isin(MODULES)].head(12).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9,5))
    ax.barh(non_mod["Feature"], non_mod["Importance"], color="#3498db", alpha=0.8)
    ax.set_xlabel("Importance")
    ax.set_title("Non-Mark Feature Importance (RF Gini)", fontweight="bold")
    st.pyplot(fig); plt.close()

    st.info("These features are observable at registration, before any marks exist. "
            "A preliminary risk flag at enrolment could be generated from them alone.")

    st.markdown("---")
    st.subheader("Full importance table")
    st.dataframe(rf_imp, use_container_width=True, hide_index=True)

# PAGE: PREDICT A STUDENT
elif page == "Predict a Student":
    st.markdown('<div class="main-header">Predict a Student\'s Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter the student\'s profile, the trained SVM and Random Forest make the prediction</div>',
                unsafe_allow_html=True)

    with st.spinner("Loading models..."):
        m = train_models()
    svm_pipe = m["svm"]
    rf_pipe  = m["rf"]
    feat_cols = m["feature_cols"]

    df_raw = load_data()

    with st.form("pred_form"):
        st.markdown("#### Module Marks")
        mark_cols = st.columns(4)
        marks = {}
        for i, mod in enumerate(MODULES):
            marks[mod] = mark_cols[i%4].number_input(
                SHORT[mod], min_value=0, max_value=100, value=60, step=1
            )

        st.markdown("#### Student Background")
        c1,c2,c3 = st.columns(3)

        acad_year = c1.selectbox("Academic Year",
            sorted(df_raw["Academic Year"].unique()))
        gender = c1.selectbox("Gender",
            sorted(df_raw["Gender"].dropna().unique()) if "Gender" in df_raw.columns else ["M","F"])
        age = c2.number_input("Age at Registration", min_value=17, max_value=40, value=19)
        school_q = c2.selectbox("School Quintile", [0,1,2,3,4,5])
        prev_act = c3.selectbox("Previous Activity",
            sorted(df_raw["Previous Activity Desc"].dropna().unique())
            if "Previous Activity Desc" in df_raw.columns
            else ["GRADE12 LEARNER","UNEMPLOYED","UNIVERSITY STUDENT"])
        language = c3.selectbox("Language",
            sorted(df_raw["Language Name"].dropna().unique())
            if "Language Name" in df_raw.columns else ["ENGLISH","XHOSA","AFRIKAANS"])
        ethnicity = c1.selectbox("Ethnic Group",
            sorted(df_raw["Ethnic Group Name"].dropna().unique())
            if "Ethnic Group Name" in df_raw.columns else ["AFRICAN","COLOURED","INDIAN","WHITE"])
        country = c2.selectbox("Country",
            sorted(df_raw["Country Name"].dropna().unique())
            if "Country Name" in df_raw.columns else ["R.S.A."])
        funded = c3.selectbox("Funded", ["Yes","No"])
        res_status = c1.selectbox("Residence Status", ["Yes","No"])
        fundname = c2.selectbox("Fund Name",
            sorted(df_raw["FUNDNAME"].dropna().unique())
            if "FUNDNAME" in df_raw.columns else ["NSFAS","Self-Funded"])
        stu_type = c3.selectbox("Student Type",
            sorted(df_raw["Student Type Description"].dropna().unique())
            if "Student Type Description" in df_raw.columns else ["New","Returning"])

        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    if submitted:
        # Build a one-row dataframe matching the training feature set
        row = {
            "Academic Year":           acad_year,
            "Gender":                  gender,
            "Language Name":           language,
            "Ethnic Group Name":       ethnicity,
            "Country Name":            country,
            "Ften Status This Year":   "First Time Entering",
            "Previous Activity Desc":  prev_act,
            "School Quintile":         school_q,
            "Age_at_Registration":     age,
            "Funded":                  funded,
            "Res_status":              res_status,
            "FUNDNAME":                fundname,
            "Student Type Description":stu_type,
            **marks,
        }
        # Add any remaining columns the model expects, filled with the training median/mode
        input_df = pd.DataFrame([row])
        for col in feat_cols:
            if col not in input_df.columns:
                if col in df_raw.select_dtypes(include=["int64","float64"]).columns:
                    input_df[col] = df_raw[col].median()
                else:
                    input_df[col] = df_raw[col].mode()[0] if col in df_raw.columns else "Unknown"
        input_df = input_df[feat_cols]

        svm_prob = svm_pipe.predict_proba(input_df)[0,1]
        rf_prob  = rf_pipe.predict_proba(input_df)[0,1]
        svm_pred = svm_pipe.predict(input_df)[0]
        rf_pred  = rf_pipe.predict(input_df)[0]
        ensemble = (svm_prob + rf_prob) / 2

        st.markdown("---")
        st.subheader("Prediction Results")
        c1,c2,c3 = st.columns(3)

        def badge(pred):
            if pred == 1:
                return '<span class="at-risk-badge">AT-RISK</span>'
            return '<span class="safe-badge">NOT AT-RISK</span>'

        c1.markdown(f"**SVM**<br>{badge(svm_pred)}<br>Probability: {svm_prob*100:.1f}%",
                    unsafe_allow_html=True)
        c2.markdown(f"**Random Forest**<br>{badge(rf_pred)}<br>Probability: {rf_prob*100:.1f}%",
                    unsafe_allow_html=True)
        c3.markdown(f"**Ensemble average**<br>{badge(int(ensemble>=0.5))}<br>Probability: {ensemble*100:.1f}%",
                    unsafe_allow_html=True)

        # Risk gauge
        st.markdown("---")
        st.subheader("Risk probability gauge")
        fig, ax = plt.subplots(figsize=(10,2))
        ax.barh(["SVM","Random Forest","Ensemble"],
                [svm_prob, rf_prob, ensemble],
                color=["#dc3545" if p>=0.5 else "#28a745"
                       for p in [svm_prob, rf_prob, ensemble]],
                height=0.5)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Decision threshold (0.5)")
        ax.set_xlim(0,1); ax.set_xlabel("At-risk probability")
        ax.set_title("Risk Probability by Model", fontweight="bold")
        ax.legend(loc="lower right")
        for i, p in enumerate([svm_prob, rf_prob, ensemble]):
            ax.text(p+0.01, i, f"{p*100:.1f}%", va="center", fontsize=10, fontweight="bold")
        st.pyplot(fig); plt.close()

        # Module mark breakdown
        st.markdown("---")
        st.subheader("Module mark profile")
        mark_df = pd.DataFrame({
            "Module": [SHORT[m] for m in MODULES],
            "Mark": [marks[m] for m in MODULES],
            "Status": ["Fail" if marks[m]<PASS_MARK else "Pass" for m in MODULES],
        })
        fig, ax = plt.subplots(figsize=(10,4))
        colors_m = ["#dc3545" if s=="Fail" else "#28a745" for s in mark_df["Status"]]
        ax.bar(mark_df["Module"], mark_df["Mark"], color=colors_m, alpha=0.85)
        ax.axhline(PASS_MARK, color="black", linestyle="--", linewidth=1.2, label="Pass mark (50)")
        ax.set_ylabel("Mark"); ax.set_ylim(0,110)
        ax.set_title("Student Module Marks", fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        ax.legend()
        for i, (_, row_m) in enumerate(mark_df.iterrows()):
            ax.text(i, row_m["Mark"]+1.5, str(int(row_m["Mark"])),
                    ha="center", fontsize=9, fontweight="bold")
        st.pyplot(fig); plt.close()

        st.warning("This tool is a screening aid only. "
                   "Model outputs should be used alongside lecturer observation, "
                   "attendance monitoring, and student self-referral.")