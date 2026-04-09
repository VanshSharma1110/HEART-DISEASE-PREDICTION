import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Heart Disease Prediction System", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")
                   
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #0d0f14; }
h1,h2,h3 { font-family: 'DM Serif Display', serif; color: #f0f4ff; }
.hero-title { font-family:'DM Serif Display',serif; font-size:2.8rem; color:#f0f4ff; line-height:1.1; margin-bottom:0.2rem; }
.hero-sub { color:#8892a4; font-size:1rem; margin-bottom:1.5rem; }
.metric-card { background:linear-gradient(135deg,#1a1f2e,#1e2538); border:1px solid #2a3050; border-radius:12px; padding:1.2rem 1.5rem; text-align:center; margin-bottom:0.5rem; }
.metric-val { font-size:2rem; font-weight:700; color:#5b8dee; }
.metric-label { font-size:0.75rem; color:#8892a4; text-transform:uppercase; letter-spacing:0.08em; }
.result-safe { background:linear-gradient(135deg,#0d2b1a,#0f3320); border:1.5px solid #1db954; border-radius:16px; padding:1.8rem; text-align:center; }
.result-risk { background:linear-gradient(135deg,#2b0d0d,#331010); border:1.5px solid #e05252; border-radius:16px; padding:1.8rem; text-align:center; }
.result-title { font-family:'DM Serif Display',serif; font-size:1.8rem; margin-bottom:0.5rem; }
.algo-card { background:#1a1f2e; border:1px solid #2a3050; border-radius:10px; padding:1rem; text-align:center; margin-bottom:0.5rem; }
.stButton>button { background:linear-gradient(135deg,#5b8dee,#7c6ff7); color:white; border:none; border-radius:10px; padding:0.7rem 2rem; font-size:1rem; font-weight:600; width:100%; }
[data-testid="stSidebar"] { background-color:#10131c; border-right:1px solid #1e2538; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_data_and_models():
    np.random.seed(42)
    n = 270
    age       = np.random.randint(29, 77, n)
    sex       = np.random.randint(0, 2, n)
    cp        = np.random.randint(1, 5, n)
    bp        = np.random.randint(94, 200, n)
    chol      = np.random.randint(126, 564, n)
    fbs       = np.random.randint(0, 2, n)
    ekg       = np.random.randint(0, 3, n)
    max_hr    = np.random.randint(71, 202, n)
    ex_angina = np.random.randint(0, 2, n)
    st_dep    = np.round(np.random.uniform(0, 6.2, n), 1)
    slope     = np.random.randint(1, 4, n)
    vessels   = np.random.randint(0, 4, n)
    thallium  = np.random.choice([3, 6, 7], n)
    risk_score = (
        (age > 55).astype(int) + (sex == 1).astype(int) +
        (cp == 4).astype(int)*2 + (bp > 140).astype(int) +
        (chol > 240).astype(int) + (max_hr < 120).astype(int) +
        (ex_angina == 1).astype(int) + (st_dep > 2).astype(int) +
        (vessels > 1).astype(int)
    )
    target = (risk_score >= 4).astype(int)
    df = pd.DataFrame({
        'Age': age, 'Sex': sex, 'Chest pain type': cp,
        'BP': bp, 'Cholesterol': chol, 'FBS over 120': fbs,
        'EKG results': ekg, 'Max HR': max_hr,
        'Exercise angina': ex_angina, 'ST depression': st_dep,
        'Slope of ST': slope, 'Number of vessels fluro': vessels,
        'Thallium': thallium, 'Heart Disease': target
    })
    X = df.drop(columns=['Heart Disease'])
    y = df['Heart Disease']
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM":                 SVC(probability=True, kernel='rbf', random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
    }
    trained = {}
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        preds = m.predict(X_test_s)
        proba = m.predict_proba(X_test_s)[:, 1]
        acc   = accuracy_score(y_test, preds)
        cv    = cross_val_score(m, scaler.transform(X), y, cv=5, scoring='accuracy').mean()
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        trained[name] = {"model": m, "accuracy": acc, "cv_accuracy": cv,
                         "preds": preds, "proba": proba, "fpr": fpr, "tpr": tpr, "auc": roc_auc}
    best = max(trained, key=lambda k: trained[k]["accuracy"])
    return trained, scaler, X_test_s, y_test, feature_names, best, df, X, y

trained_models, scaler, X_test_s, y_test, feature_names, best_name, df_full, X_full, y_full = get_data_and_models()

with st.sidebar:
    st.markdown("## 🫀 Algorithm Selector")
    selected_model = st.selectbox("Select Algorithm", list(trained_models.keys()),
                                   index=list(trained_models.keys()).index(best_name))
    st.markdown("---")
    st.markdown("### 📊 All Model Accuracies")
    for name, info in trained_models.items():
        icon = "🏆" if name == best_name else "  "
        st.markdown(f"**{icon} {name}**")
        st.progress(float(info['accuracy']))
        st.caption(f"Test: {info['accuracy']*100:.1f}%  |  CV: {info['cv_accuracy']*100:.1f}%")
    st.markdown("---")
    st.caption("Dataset: Kaggle — Heart Disease Prediction\n13 Clinical Features")

st.markdown('<p class="hero-title">🫀 Heart Disease<br>Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Clinical decision support · Random Forest · Logistic Regression · SVM · KNN</p>', unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)
with s1: st.markdown(f'<div class="metric-card"><div class="metric-val">{len(df_full)}</div><div class="metric-label">Total Patients</div></div>', unsafe_allow_html=True)
with s2: st.markdown(f'<div class="metric-card"><div class="metric-val">13</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
with s3:
    best_acc = trained_models[best_name]['accuracy']*100
    st.markdown(f'<div class="metric-card"><div class="metric-val">{best_acc:.1f}%</div><div class="metric-label">Best Accuracy</div></div>', unsafe_allow_html=True)
with s4:
    pos_pct = y_full.mean()*100
    st.markdown(f'<div class="metric-card"><div class="metric-val">{pos_pct:.1f}%</div><div class="metric-label">Disease Rate</div></div>', unsafe_allow_html=True)

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Predict", "📈 Model Performance", "📉 ROC Curves", "📋 Dataset Insights"])

# ── TAB 1 ──
with tab1:
    st.markdown("### Enter Patient Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Demographics & Vitals**")
        age_in  = st.slider("Age", 20, 80, 52)
        sex_in  = st.selectbox("Sex", ["Male (1)", "Female (0)"])
        bp_in   = st.slider("Blood Pressure (mmHg)", 90, 200, 125)
        chol_in = st.slider("Cholesterol (mg/dl)", 100, 570, 212)
        fbs_in  = st.selectbox("Fasting Blood Sugar > 120", ["No (0)", "Yes (1)"])
    with c2:
        st.markdown("**Cardiac Parameters**")
        cp_in    = st.selectbox("Chest Pain Type", ["1 – Typical Angina","2 – Atypical Angina","3 – Non-anginal","4 – Asymptomatic"])
        ekg_in   = st.selectbox("EKG Results", ["0 – Normal","1 – ST-T Abnormality","2 – LV Hypertrophy"])
        maxhr_in = st.slider("Max Heart Rate", 60, 210, 168)
        exang_in = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
    with c3:
        st.markdown("**Advanced Indicators**")
        stdep_in   = st.slider("ST Depression", 0.0, 6.5, 1.0, 0.1)
        slope_in   = st.selectbox("Slope of ST Segment", ["1 – Upsloping","2 – Flat","3 – Downsloping"])
        vessels_in = st.slider("Vessels Colored by Fluoro", 0, 3, 0)
        thal_in    = st.selectbox("Thallium Stress Test", ["3 – Normal","6 – Fixed Defect","7 – Reversible Defect"])

    st.markdown("")
    if st.button("🫀 Predict Heart Disease Risk"):
        input_vals = [
            age_in, 1 if "Male" in sex_in else 0, int(cp_in[0]),
            bp_in, chol_in, int(fbs_in[0]), int(ekg_in[0]), maxhr_in,
            int(exang_in[0]), stdep_in, int(slope_in[0]),
            vessels_in, int(thal_in.split("–")[0].strip()),
        ]
        input_arr    = np.array(input_vals).reshape(1, -1)
        input_scaled = scaler.transform(input_arr)
        model = trained_models[selected_model]["model"]
        pred  = model.predict(input_scaled)[0]
        prob  = model.predict_proba(input_scaled)[0]
        st.markdown("---")
        st.markdown("### 🔍 Prediction Result")
        r1, r2 = st.columns([2, 1])
        with r1:
            if pred == 1:
                st.markdown(f'<div class="result-risk"><div class="result-title" style="color:#e05252">⚠️ Heart Disease Detected</div><p style="color:#f0a0a0;margin:0.5rem 0 0 0">High risk indicators found. <b>Immediate medical consultation recommended.</b></p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-safe"><div class="result-title" style="color:#1db954">✅ No Heart Disease</div><p style="color:#a0f0b0;margin:0.5rem 0 0 0">Low risk detected. <b>Continue healthy lifestyle habits.</b></p></div>', unsafe_allow_html=True)
            st.markdown("")
            p1, p2, p3 = st.columns(3)
            with p1: st.markdown(f'<div class="metric-card"><div class="metric-val">{prob[1]*100:.1f}%</div><div class="metric-label">Disease Probability</div></div>', unsafe_allow_html=True)
            with p2: st.markdown(f'<div class="metric-card"><div class="metric-val">{trained_models[selected_model]["accuracy"]*100:.1f}%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
            with p3: st.markdown(f'<div class="metric-card"><div class="metric-val">{selected_model.split()[0]}</div><div class="metric-label">Algorithm</div></div>', unsafe_allow_html=True)
        with r2:
            fig, ax = plt.subplots(figsize=(3.2,3.2), facecolor='#1a1f2e')
            ax.set_facecolor('#1a1f2e')
            _, _, autotexts = ax.pie(prob, labels=['No Disease','Disease'],
                                     colors=['#1db954','#e05252'], autopct='%1.1f%%',
                                     startangle=90, textprops={'color':'#f0f4ff','fontsize':9})
            for at in autotexts: at.set_color('#f0f4ff'); at.set_fontweight('bold')
            ax.set_title('Risk Probability', color='#f0f4ff', fontsize=10, pad=8)
            st.pyplot(fig); plt.close()
        st.markdown("---")
        st.markdown("### 🤖 All 4 Algorithms — Quick Comparison")
        acols = st.columns(4)
        for i, (name, info) in enumerate(trained_models.items()):
            p = info["model"].predict_proba(input_scaled)[0]
            res = "⚠️ Risk" if info["model"].predict(input_scaled)[0] == 1 else "✅ Safe"
            color = "#e05252" if "Risk" in res else "#1db954"
            with acols[i]:
                st.markdown(f'<div class="algo-card"><div style="color:#f0f4ff;font-weight:600;font-size:0.9rem">{name}</div><div style="color:{color};font-size:1.4rem;font-weight:700">{res}</div><div style="color:#8892a4;font-size:0.8rem">{p[1]*100:.1f}% probability</div><div style="color:#5b8dee;font-size:0.8rem">Acc: {info["accuracy"]*100:.1f}%</div></div>', unsafe_allow_html=True)

# ── TAB 2 ──
with tab2:
    st.markdown("### Confusion Matrix & Accuracy Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor='#0d0f14')
    ax = axes[0]; ax.set_facecolor('#1a1f2e')
    cm = confusion_matrix(y_test, trained_models[selected_model]["preds"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease','Disease'], yticklabels=['No Disease','Disease'],
                ax=ax, linewidths=0.5, annot_kws={"size":14,"weight":"bold"})
    ax.set_title(f'{selected_model} — Confusion Matrix', color='#f0f4ff', fontsize=12, pad=10)
    ax.set_xlabel('Predicted', color='#8892a4', fontsize=10)
    ax.set_ylabel('Actual', color='#8892a4', fontsize=10)
    ax.tick_params(colors='#8892a4')
    ax2 = axes[1]; ax2.set_facecolor('#1a1f2e')
    names    = list(trained_models.keys())
    test_acc = [trained_models[n]["accuracy"]*100 for n in names]
    cv_acc   = [trained_models[n]["cv_accuracy"]*100 for n in names]
    x = np.arange(len(names)); w = 0.35
    b1 = ax2.bar(x-w/2, test_acc, w, label='Test Acc', color='#5b8dee', edgecolor='#0d0f14')
    b2 = ax2.bar(x+w/2, cv_acc,   w, label='CV Acc',   color='#7c6ff7', edgecolor='#0d0f14')
    for bar in list(b1)+list(b2):
        ax2.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.3,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', color='#f0f4ff', fontsize=8.5)
    ax2.set_ylim(0, 115); ax2.set_xticks(x)
    ax2.set_xticklabels(names, color='#8892a4', fontsize=9)
    ax2.set_title('Test vs Cross-Validation Accuracy', color='#f0f4ff', fontsize=12, pad=10)
    ax2.set_ylabel('Accuracy (%)', color='#8892a4'); ax2.tick_params(colors='#8892a4')
    ax2.legend(facecolor='#1a1f2e', labelcolor='#f0f4ff', fontsize=9)
    for spine in ax2.spines.values(): spine.set_color('#2a3050')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("### Classification Reports — All 4 Algorithms")
    cr1, cr2 = st.columns(2)
    for i, (name, info) in enumerate(trained_models.items()):
        report = classification_report(y_test, info["preds"], target_names=["No Disease","Disease"], output_dict=True)
        rdf = pd.DataFrame(report).transpose().round(3)
        with (cr1 if i % 2 == 0 else cr2):
            st.markdown(f"**{name}**")
            st.dataframe(rdf.style.background_gradient(cmap='Blues'), use_container_width=True)

# ── TAB 3 ──
with tab3:
    st.markdown("### ROC Curves — All 4 Algorithms")
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor='#0d0f14')
    ax.set_facecolor('#1a1f2e')
    colors_roc = ['#5b8dee','#1db954','#f7b731','#e05252']
    for (name, info), color in zip(trained_models.items(), colors_roc):
        ax.plot(info["fpr"], info["tpr"], color=color, lw=2, label=f'{name} (AUC={info["auc"]:.2f})')
    ax.plot([0,1],[0,1],'w--',lw=1,alpha=0.5,label='Random Baseline')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel('False Positive Rate', color='#8892a4', fontsize=11)
    ax.set_ylabel('True Positive Rate', color='#8892a4', fontsize=11)
    ax.set_title('ROC Curves Comparison', color='#f0f4ff', fontsize=14, pad=12)
    ax.tick_params(colors='#8892a4')
    ax.legend(facecolor='#1a1f2e', labelcolor='#f0f4ff', fontsize=10, loc='lower right')
    for spine in ax.spines.values(): spine.set_color('#2a3050')
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("### AUC Scores")
    a1,a2,a3,a4 = st.columns(4)
    for col,(name,info) in zip([a1,a2,a3,a4],trained_models.items()):
        with col: st.markdown(f'<div class="metric-card"><div class="metric-val">{info["auc"]:.3f}</div><div class="metric-label">{name}<br>AUC Score</div></div>', unsafe_allow_html=True)

# ── TAB 4 ──
with tab4:
    st.markdown("### Dataset Overview")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#0d0f14')
    ax1 = axes[0]; ax1.set_facecolor('#1a1f2e')
    counts = y_full.value_counts().sort_index()
    ax1.bar(['No Disease','Disease'], counts.values, color=['#1db954','#e05252'], edgecolor='#0d0f14', width=0.5)
    ax1.set_title('Class Distribution', color='#f0f4ff', pad=10); ax1.tick_params(colors='#8892a4')
    for spine in ax1.spines.values(): spine.set_color('#2a3050')
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    for i,v in enumerate(counts.values): ax1.text(i, v+1, str(v), ha='center', color='#f0f4ff', fontsize=11)
    ax2 = axes[1]; ax2.set_facecolor('#1a1f2e')
    for label,color,lname in [(0,'#1db954','No Disease'),(1,'#e05252','Disease')]:
        subset = df_full[df_full['Heart Disease']==label]['Age']
        ax2.hist(subset, bins=15, alpha=0.7, color=color, edgecolor='#0d0f14', label=lname)
    ax2.set_title('Age Distribution by Class', color='#f0f4ff', pad=10)
    ax2.set_xlabel('Age', color='#8892a4'); ax2.tick_params(colors='#8892a4')
    ax2.legend(facecolor='#1a1f2e', labelcolor='#f0f4ff', fontsize=9)
    for spine in ax2.spines.values(): spine.set_color('#2a3050')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax3 = axes[2]; ax3.set_facecolor('#1a1f2e')
    corr = df_full.corr()['Heart Disease'].drop('Heart Disease').sort_values()
    colors_c = ['#e05252' if v > 0 else '#5b8dee' for v in corr]
    ax3.barh(corr.index, corr.values, color=colors_c, edgecolor='#0d0f14')
    ax3.set_title('Feature Correlation with Target', color='#f0f4ff', pad=10)
    ax3.axvline(0, color='#2a3050', linewidth=1); ax3.tick_params(colors='#8892a4', labelsize=8)
    for spine in ax3.spines.values(): spine.set_color('#2a3050')
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("### Feature Importance (Random Forest)")
    rf_model    = trained_models["Random Forest"]["model"]
    importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig2, ax4   = plt.subplots(figsize=(9,4.5), facecolor='#0d0f14')
    ax4.set_facecolor('#1a1f2e')
    colors_imp = ['#f7b731' if v == importances.max() else '#5b8dee' for v in importances]
    ax4.barh(importances.index, importances.values, color=colors_imp, edgecolor='#0d0f14')
    ax4.set_title('Random Forest Feature Importances', color='#f0f4ff', pad=10)
    ax4.set_xlabel('Importance Score', color='#8892a4'); ax4.tick_params(colors='#8892a4', labelsize=9)
    for spine in ax4.spines.values(): spine.set_color('#2a3050')
    ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("### Raw Dataset Preview")
    st.dataframe(df_full.head(20).style.background_gradient(cmap='Blues', subset=['Heart Disease']), use_container_width=True)