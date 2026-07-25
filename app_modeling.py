from config import DATABASE_URL, OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN
"""
Модуль В — Моделирование и прогнозирование (инвариант)
Sapere Intelligence Platform

3.1 Обучение модели
3.2 Организация непрерывного обучения
3.3 Оценка временных характеристик
3.4 Построение рекомендаций
"""

import os
import json
import hashlib
import datetime
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIG & DATABASE
# ══════════════════════════════════════════════════════════════
DB_URL = DATABASE_URL
AI_API_KEY = OPENROUTER_API_KEY
TEXT_MODEL = "google/gemini-2.5-flash"
engine = create_engine(DB_URL)

MODEL_DIR = os.path.join(os.getcwd(), "models")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

LOG_FILE = os.path.join(MODEL_DIR, "training_log.json")
COMPLETENESS_FILE = "subject_completeness.json"

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ══════════════════════════════════════════════════════════════
from PIL import Image
icon = Image.open("assets/icon_modeling.png") if os.path.exists("assets/icon_modeling.png") else None
st.set_page_config(
    page_title="Sapere Intelligence",
    page_icon=icon,
    layout="wide"
)

THEME_BG = "#0d1117"
THEME_CARD = "#161b22"
THEME_BORDER = "#30363d"
THEME_TEXT = "#c9d1d9"
THEME_ACCENT = "#f9a03f"
THEME_ACCENT_ALT = "#58a6ff"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Inter', -apple-system, sans-serif;
}
h1, h2, h3, h4, h5 { color: #f9a03f !important; font-weight: 700 !important; }

.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background-color: #161b22; padding: 6px;
    border-radius: 8px; border: 1px solid #30363d;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px; padding: 10px 24px;
    color: #8b949e !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background-color: #21262d !important; color: #f9a03f !important; font-weight: 600 !important;
}
.stButton>button {
    background-color: #21262d; color: #f9a03f !important;
    border: 1px solid #30363d !important; border-radius: 6px !important;
    font-weight: 600 !important; width: 100%; transition: all 0.2s ease;
}
.stButton>button:hover { background-color: #30363d; border-color: #f9a03f !important; }
[data-testid="stMetricValue"] { color: #f9a03f !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; text-transform: uppercase; font-weight: 500; }
.metric-card {
    background-color: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 15px; text-align: center; margin-bottom: 10px;
}
.metric-value { font-size: 24px; font-weight: bold; color: #f9a03f; }
.metric-label { font-size: 12px; color: #8b949e; }
.card {
    background-color: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 20px; margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
def init_time_columns():
    try:
        with engine.connect() as conn:
            res = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'opit_files'"))
            cols = [r[0] for r in res.fetchall()]

        new_cols = [("ai_est_time_min", "FLOAT"), ("ai_time_explanation", "TEXT")]
        for col_name, col_type in new_cols:
            if col_name not in cols:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE opit_files ADD COLUMN {col_name} {col_type}"))
    except Exception as e:
        st.error(f"Ошибка инициализации колонок времени в БД: {e}")

init_time_columns()

@st.cache_data(show_spinner=False, ttl=30)
def load_data():
    try:
        df = pd.read_sql("SELECT * FROM opit_files", engine)
        if df.empty:
            return pd.DataFrame()

        df['word_count'] = df['text_content'].fillna("").apply(lambda x: len(x.split()))
        df['text_length'] = pd.to_numeric(df.get('text_length', 0), errors='coerce').fillna(0)
        df['file_size'] = pd.to_numeric(df.get('file_size', 0), errors='coerce').fillna(0)
        df['images_count'] = pd.to_numeric(df.get('images_count', 0), errors='coerce').fillna(0)

        num_cols = [
            'compliance_rate', 'compliance_structure', 'compliance_clarity',
            'compliance_terminology', 'compliance_completeness',
            'practice_orientation', 'legal_compliance', 'technical_quality'
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        df['complexity'] = df['word_count'] * 0.05 + df['text_length'] * 0.01
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки из БД: {e}")
        return pd.DataFrame()


def load_completeness():
    if os.path.exists(COMPLETENESS_FILE):
        with open(COMPLETENESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_training_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_training_log(log):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=str)


def call_ai(prompt):
    """Call OpenRouter AI for text generation."""
    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": TEXT_MODEL, "messages": [{"role": "user", "content": prompt}]}
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                          headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"AI API Error: {e}")
    return None


def call_openrouter_time_estimation(materials_list):
    prompt = f"""Ты - опытный методист и планировщик образовательных программ.
Оцени время, необходимое для изучения следующих учебных материалов:
{json.dumps(materials_list, ensure_ascii=False)}

Ответь строго в формате корректного JSON со следующей структурой:
{{
  "estimates": [
    {{
      "id": "хэш-id-файла",
      "time_min": 45,
      "explanation": "Краткое обоснование: 500 слов, 2 схемы. Требует вдумчивого чтения."
    }}
  ]
}}
Не добавляй никакого другого текста или разметки markdown вне JSON block."""
    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": TEXT_MODEL, "messages": [{"role": "user", "content": prompt}]}
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            content = r.json()['choices'][0]['message']['content'].strip()
            s = content.find('{')
            e = content.rfind('}')
            if s != -1 and e != -1:
                return json.loads(content[s:e+1])
    except Exception as e:
        st.error(f"AI API Error: {e}")
    return None

# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def build_features(df):
    """Build numerical feature matrix from the dataframe."""
    feature_cols = [
        'word_count', 'text_length', 'file_size', 'images_count',
        'compliance_rate', 'compliance_structure', 'compliance_clarity',
        'compliance_terminology', 'compliance_completeness',
        'practice_orientation', 'legal_compliance', 'technical_quality',
        'complexity'
    ]
    existing = [c for c in feature_cols if c in df.columns]
    X_num = df[existing].fillna(0).values

    # TF-IDF on combined text
    combined = (
        df['subject'].fillna("") + " " +
        df['topic'].fillna("") + " " +
        df['annotation'].fillna("")
    )
    vectorizer = TfidfVectorizer(max_features=100, stop_words=None)
    X_text = vectorizer.fit_transform(combined).toarray()

    X = np.hstack([X_num, X_text])
    return X, vectorizer, existing


# ══════════════════════════════════════════════════════════════
# PLOTLY DARK THEME HELPER
# ══════════════════════════════════════════════════════════════
def style_fig(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(color=THEME_TEXT, size=16, family="Inter")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME_TEXT, family="Inter"),
        xaxis=dict(gridcolor="#21262d", tickfont=dict(color="#8b949e")),
        yaxis=dict(gridcolor="#21262d", tickfont=dict(color="#8b949e")),
        legend=dict(font=dict(color=THEME_TEXT)),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


# ══════════════════════════════════════════════════════════════
# 3.1 — TRAIN ML MODELS
# ══════════════════════════════════════════════════════════════
def train_and_evaluate(X, y, target_name):
    """Train multiple classifiers and return comparison results."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    # Stratified split
    if len(np.unique(y_enc)) < 2:
        return None, None, None, None, le

    test_size = 0.25
    if min(np.bincount(y_enc)) < 2:
        test_size = 0.4  # need more samples per class

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, random_state=42, stratify=y_enc
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, random_state=42
        )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=min(5, len(X_train_s) - 1))
    }

    results = []
    best_model = None
    best_f1 = -1

    for name, model in models.items():
        try:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # ROC-AUC
            try:
                if n_classes == 2:
                    y_proba = model.predict_proba(X_test_s)[:, 1]
                    roc = roc_auc_score(y_test, y_proba)
                else:
                    y_proba = model.predict_proba(X_test_s)
                    roc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except Exception:
                roc = 0.0

            results.append({
                "Алгоритм": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(roc, 4)
            })

            if f1 > best_f1:
                best_f1 = f1
                best_model = (name, model, scaler)
        except Exception as e:
            results.append({
                "Алгоритм": name,
                "Accuracy": "Ошибка", "Precision": "-", "Recall": "-",
                "F1-Score": "-", "ROC-AUC": "-"
            })

    results_df = pd.DataFrame(results)
    return results_df, best_model, (X_test_s, y_test), le, results


def save_model(model_tuple, le, target_name, metrics):
    """Save model, scaler, encoder to disk with versioning."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{target_name}_{ts}"
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")

    joblib.dump({
        "model_name": model_tuple[0],
        "model": model_tuple[1],
        "scaler": model_tuple[2],
        "label_encoder": le,
        "target": target_name,
        "timestamp": ts,
        "metrics": metrics
    }, model_path)

    # Update log
    log = load_training_log()
    log.append({
        "file": model_path,
        "target": target_name,
        "model_name": model_tuple[0],
        "timestamp": ts,
        "metrics": metrics
    })
    save_training_log(log)
    return model_path


# ══════════════════════════════════════════════════════════════
# 3.2 — CONTINUOUS LEARNING AGENT
# ══════════════════════════════════════════════════════════════
def compute_data_fingerprint(df, target_col):
    """Compute a hash fingerprint of the data to detect changes."""
    content = df[target_col].fillna("NONE").astype(str).str.cat(sep="|")
    return hashlib.md5(content.encode()).hexdigest()


def detect_drift(old_dist, new_dist, threshold=0.1):
    """
    Detect data drift using Jensen-Shannon divergence
    on label distributions.
    """
    all_labels = sorted(set(list(old_dist.keys()) + list(new_dist.keys())))
    old_vec = np.array([old_dist.get(l, 0) for l in all_labels], dtype=float)
    new_vec = np.array([new_dist.get(l, 0) for l in all_labels], dtype=float)

    # Normalize
    old_vec = old_vec / (old_vec.sum() + 1e-12)
    new_vec = new_vec / (new_vec.sum() + 1e-12)

    js_div = jensenshannon(old_vec, new_vec)
    return js_div, js_div > threshold


def get_label_distribution(df, col):
    vc = df[col].fillna("NONE").value_counts(normalize=True)
    return vc.to_dict()


# ══════════════════════════════════════════════════════════════
# 3.3 — TIME ESTIMATION
# ══════════════════════════════════════════════════════════════
DIFFICULTY_MULTIPLIER = {"Низкая": 0.8, "Средняя": 1.0, "Высокая": 1.4}
BASE_READING_SPEED_WPM = 150  # words per minute for educational text


def estimate_material_time(row):
    """Estimate study time for a single material (in minutes)."""
    wc = max(row.get('word_count', 0), 100)
    img_count = pd.to_numeric(row.get('images_count', 0), errors='coerce')
    if pd.isna(img_count): img_count = 0

    diff = str(row.get('label_difficulty', 'Средняя'))
    multiplier = DIFFICULTY_MULTIPLIER.get(diff, 1.0)

    # 1. Base reading time (text)
    reading_time = wc / BASE_READING_SPEED_WPM

    # 2. Visuals study time (assume 3 mins per diagram/image)
    image_time = img_count * 3.0

    # 3. Practice / exercises time heuristic (practice usually takes 1.5x - 2.5x reading time)
    practice_time = (reading_time + image_time) * 2.0 * multiplier

    # 4. Fixed base time for setup, context switching, or introductory materials
    base_setup_time = 30.0

    # 5. Review and consolidation time
    review_time = (reading_time + practice_time) * 0.2

    total = (base_setup_time + reading_time + image_time + practice_time + review_time) * multiplier
    return round(total, 1)


def estimate_subject_time(df_subj):
    """Estimate total time for a subject considering sequential/parallel."""
    if df_subj.empty:
        return 0.0

    df_subj = df_subj.copy()
    df_subj['est_time_min'] = df_subj.apply(estimate_material_time, axis=1)

    # Group by parallel_id — materials in the same parallel group can be studied simultaneously
    if 'label_parallel_id' in df_subj.columns:
        parallel_groups = df_subj.groupby('label_parallel_id')
        # For parallel groups, take the max time (they run concurrently)
        parallel_times = parallel_groups['est_time_min'].max()
        total = parallel_times.sum()
    else:
        total = df_subj['est_time_min'].sum()

    return round(total, 1)



# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════
st.title("Моделирование и прогнозирование")
st.caption("Обучение ML-моделей, непрерывное обучение, временные оценки и рекомендательный агент")

df_raw = load_data()

if df_raw.empty:
    st.warning("В базе данных нет записей. Пожалуйста, проверьте подключение к PostgreSQL и наличие данных.")
    st.stop()

has_labels = all(c in df_raw.columns for c in ['label_parallel_id', 'label_sequential_id', 'label_difficulty'])
labeled_count = 0
if has_labels:
    labeled_count = df_raw['label_difficulty'].notna().sum()

st.sidebar.header("Панель управления")
st.sidebar.write(f"Загружено записей: **{len(df_raw)}**")
st.sidebar.write(f"Размечено записей: **{labeled_count}**")
if st.sidebar.button("🔄 Обновить данные"):
    st.cache_data.clear()
    st.rerun()

tabs = st.tabs([
    "3.1 Обучение модели",
    "3.2 Непрерывное обучение",
    "3.3 Временные характеристики",
    "3.4 Траектории обучения"
])

# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 1: MODEL TRAINING                                       ║
# ╚══════════════════════════════════════════════════════════════╝
with tabs[0]:
    st.header("3.1 Обучение и сравнение ML-моделей")

    if not has_labels or labeled_count == 0:
        st.warning("Нет размеченных данных. Будет использована автоматическая генерация меток на основе признаков для демонстрации.")
        df_work = df_raw.copy()
        q = df_work['complexity'].quantile([0.33, 0.66])
        df_work['label_difficulty'] = pd.cut(
            df_work['complexity'],
            bins=[-np.inf, q.iloc[0], q.iloc[1], np.inf],
            labels=["Низкая", "Средняя", "Высокая"]
        ).astype(str)
        subjects_u = df_work['subject'].fillna("Прочее").unique()
        subj_map = {s: f"P{i % 4 + 1}" for i, s in enumerate(subjects_u)}
        df_work['label_parallel_id'] = df_work['subject'].fillna("Прочее").map(subj_map)
        df_work['label_sequential_id'] = df_work.groupby('subject').cumcount().apply(lambda x: f"S{x + 1}")
    else:
        df_work = df_raw.copy()
        df_work['label_difficulty'] = df_work['label_difficulty'].fillna("Средняя")
        df_work['label_parallel_id'] = df_work['label_parallel_id'].fillna("P0")
        df_work['label_sequential_id'] = df_work['label_sequential_id'].fillna("S0")

    st.info(f"Набор данных: **{len(df_work)}** записей, классы сложности: {df_work['label_difficulty'].nunique()}, параллельных групп: {df_work['label_parallel_id'].nunique()}")

    X, vectorizer, feat_cols = build_features(df_work)

    target_choice = st.selectbox(
        "Выберите целевую переменную для классификации:",
        ["label_difficulty", "label_parallel_id", "label_sequential_id"],
        format_func=lambda x: {
            "label_difficulty": "Сложность освоения",
            "label_parallel_id": "Параллельное изучение (группа)",
            "label_sequential_id": "Последовательное изучение (этап)"
        }.get(x, x)
    )

    if st.button(" Обучить и сравнить модели", key="train_btn"):
        y = df_work[target_choice].values
        with st.spinner("Обучение 5 алгоритмов..."):
            results_df, best_model, test_data, le, raw_results = train_and_evaluate(X, y, target_choice)

        if results_df is None:
            st.error("Недостаточно классов для обучения. Нужно минимум 2 различных метки.")
        else:
            st.subheader("Сравнение алгоритмов")
            st.dataframe(results_df.style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                color="#2d4a1e"
            ), width="stretch")

            valid_results = results_df[results_df["Accuracy"] != "Ошибка"].copy()
            if not valid_results.empty:
                import plotly.graph_objects as go
                fig_comp = go.Figure()
                colors = [THEME_ACCENT, THEME_ACCENT_ALT, "#ff7b72", "#7ee787", "#d2a8ff"]
                for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]):
                    fig_comp.add_trace(go.Bar(
                        name=metric,
                        x=valid_results["Алгоритм"],
                        y=pd.to_numeric(valid_results[metric], errors='coerce'),
                        marker_color=colors[i % len(colors)]
                    ))
                fig_comp = style_fig(fig_comp, "Сравнение метрик по алгоритмам")
                fig_comp.update_layout(barmode='group', height=400)
                st.plotly_chart(fig_comp, width="stretch")

                if best_model and test_data:
                    st.success(f"Лучшая модель ({best_model[0]}) сохранена.")


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 2: CONTINUOUS LEARNING                                  ║
# ╚══════════════════════════════════════════════════════════════╝
with tabs[1]:
    st.header("3.2 Агент непрерывного обучения")
    log = load_training_log()

    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        st.metric("Версий моделей", len(log))
    with col_h2:
        st.metric("Записей в БД", len(df_raw))
    with col_h3:
        st.metric("Последнее обучение", str(log[-1]["timestamp"]) if log else "Нет")

    if log:
        st.subheader("История обучения моделей")
        log_df = pd.DataFrame(log)[['timestamp', 'target', 'model_name']]
        st.dataframe(log_df, width="stretch")


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 3: TIME ESTIMATION                                      ║
# ╚══════════════════════════════════════════════════════════════╝
with tabs[2]:
    st.header("3.3 Оценка времени освоения")
    st.info("Здесь показано, сколько часов займет изучение всех материалов курса.")

    df_time = df_raw.copy()
    if 'label_difficulty' not in df_time.columns or df_time['label_difficulty'].isna().all():
        df_time['label_difficulty'] = "Средняя"
    else:
        df_time['label_difficulty'] = df_time['label_difficulty'].fillna("Средняя")

    eval_mode = st.radio("Метод оценки:", ["Эвристическая оценка (Формула)", "Интеллектуальная оценка (ИИ)"], horizontal=True)

    if "Интеллектуальная оценка (ИИ)" in eval_mode:
        if st.button("Запустить ИИ-оценку для всех материалов"):
            with st.spinner("ИИ анализирует материалы и оценивает время..."):
                materials_payload = []
                for _, row in df_time.iterrows():
                    materials_payload.append({
                        "id": row["id"],
                        "subject": str(row.get("subject", "")),
                        "topic": str(row.get("topic", "")),
                        "word_count": int(row.get("word_count", 0)),
                        "images_count": int(row.get("images_count", 0)),
                        "difficulty": str(row.get("label_difficulty", "Средняя"))
                    })

                ai_result = call_openrouter_time_estimation(materials_payload)
                if ai_result and "estimates" in ai_result:
                    payload_db = []
                    for est in ai_result["estimates"]:
                        payload_db.append({
                            "id": est["id"],
                            "time_min": float(est["time_min"]),
                            "expl": str(est.get("explanation", ""))
                        })

                    try:
                        with engine.begin() as conn:
                            conn.execute(
                                text("UPDATE opit_files SET ai_est_time_min = :time_min, ai_time_explanation = :expl WHERE id = :id"),
                                payload_db
                            )
                        st.success("ИИ-оценка времени успешно завершена и сохранена в БД!")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка сохранения в БД: {e}")
                else:
                    st.error("Ошибка при запросе к ИИ.")

        if 'ai_est_time_min' in df_time.columns and not df_time['ai_est_time_min'].isna().all():
            df_time['est_time_min'] = df_time['ai_est_time_min'].fillna(60)
            df_time['explanation'] = df_time.get('ai_time_explanation', "Нет обоснования").fillna("Нет обоснования")
        else:
            st.warning("ИИ-оценка еще не проводилась. Показаны эвристические значения.")
            df_time['est_time_min'] = df_time.apply(estimate_material_time, axis=1)
            df_time['explanation'] = "Эвристическая оценка по формуле"
    else:
        df_time['est_time_min'] = df_time.apply(estimate_material_time, axis=1)
        df_time['explanation'] = "Эвристическая оценка по формуле"

    df_time['est_time_hours'] = (df_time['est_time_min'] / 60).round(1)

    total_time_hours = df_time['est_time_hours'].sum()
    subjects = df_time['subject'].dropna().unique()

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.metric("Общее время курса", f"{total_time_hours:.1f} ч")
    with col_t2:
        st.metric("Всего предметов", len(subjects))
    with col_t3:
        st.metric("Всего материалов", len(df_time))

    st.markdown("---")
    st.subheader("Затраты времени по предметам")

    subj_times = []
    for subj in subjects:
        s_df = df_time[df_time['subject'] == subj]
        subj_times.append({
            "Предмет": subj,
            "Материалов": len(s_df),
            "Время (часы)": round(s_df['est_time_hours'].sum(), 1)
        })
    subj_df = pd.DataFrame(subj_times).sort_values("Время (часы)", ascending=False)

    if not subj_df.empty:
        fig_subj = px.bar(
            subj_df, x="Время (часы)", y="Предмет", orientation='h',
            text="Время (часы)", color_discrete_sequence=[THEME_ACCENT]
        )
        fig_subj = style_fig(fig_subj, "")
        fig_subj.update_layout(height=max(300, len(subj_df) * 40))
        fig_subj.update_traces(textposition='outside')
        st.plotly_chart(fig_subj, width="stretch")

    with st.expander("Показать списком по материалам", expanded=True if "Интеллектуальная оценка (ИИ)" in eval_mode else False):
        st.dataframe(
            df_time[['subject', 'topic', 'label_difficulty', 'est_time_hours', 'explanation']].rename(columns={
                'subject': 'Предмет', 'topic': 'Тема', 'label_difficulty': 'Сложность', 'est_time_hours': 'Время (ч)', 'explanation': 'Обоснование'
            }),
            width="stretch"
        )




# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 4: TRAJECTORY RECOMMENDATION                            ║
# ╚══════════════════════════════════════════════════════════════╝
import zipfile
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch

def add_background(canvas, doc):
    canvas.saveState()
    canvas.setFillColorRGB(13/255.0, 17/255.0, 23/255.0)  # #0d1117
    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1, stroke=0)
    canvas.restoreState()

def create_load_chart(weeks_data):
    x = [f"Нед {w['Неделя']}" for w in weeks_data]

    # Calculate total hours for each week
    y = []
    for w in weeks_data:
        week_min = sum(d['Занято_мин'] for d in w['Дни'])
        y.append(week_min / 60.0)

    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    bars = ax.bar(x, y, color='#f9a03f', width=0.5)
    ax.set_title("Нагрузка по неделям (часы)", color='#c9d1d9', pad=15)
    ax.tick_params(axis='x', colors='#c9d1d9')
    ax.tick_params(axis='y', colors='#c9d1d9')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='#c9d1d9', fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0d1117', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_course_pdf(weeks_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)

    pdfmetrics.registerFont(TTFont('Arial', r'C:\Windows\Fonts\arial.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Bold', r'C:\Windows\Fonts\arialbd.ttf'))

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CustomTitle', fontName='Arial-Bold', fontSize=24, spaceAfter=20, alignment=1, textColor='#f9a03f'))
    styles.add(ParagraphStyle(name='CustomSummary', fontName='Arial', fontSize=14, spaceAfter=25, alignment=1, textColor='#c9d1d9'))
    styles.add(ParagraphStyle(name='CustomH1', fontName='Arial-Bold', fontSize=18, spaceAfter=15, textColor='#58a6ff'))
    styles.add(ParagraphStyle(name='CustomDay', fontName='Arial-Bold', fontSize=14, spaceAfter=10, textColor='#79c0ff', leftIndent=10))
    styles.add(ParagraphStyle(name='CustomItemTitle', fontName='Arial-Bold', fontSize=12, spaceAfter=5, textColor='#ffffff', leftIndent=25))
    styles.add(ParagraphStyle(name='CustomMeta', fontName='Arial', fontSize=10, spaceAfter=5, textColor='#8b949e', leftIndent=25))
    styles.add(ParagraphStyle(name='CustomBody', fontName='Arial', fontSize=11, spaceAfter=15, leading=14, textColor='#c9d1d9', leftIndent=25))

    elements = []
    elements.append(Paragraph("Ваш ежедневный план обучения", styles['CustomTitle']))

    total_hours = sum(sum(d['Занято_мин'] for d in w['Дни']) for w in weeks_data) / 60.0
    summary_text = f"Длительность курса: {len(weeks_data)} нед. | Общая нагрузка: {round(total_hours, 1)} ч."
    elements.append(Paragraph(summary_text, styles['CustomSummary']))

    chart_buf = create_load_chart(weeks_data)
    elements.append(Image(chart_buf, width=5.5*inch, height=2.3*inch))
    elements.append(Spacer(1, 0.4*inch))

    for idx_w, w in enumerate(weeks_data):
        if idx_w > 0:
            elements.append(HRFlowable(width="100%", thickness=1, color="#30363d", spaceBefore=20, spaceAfter=20))

        week_min = sum(d['Занято_мин'] for d in w['Дни'])
        elements.append(Paragraph(f"Неделя {w['Неделя']} (Нагрузка: {round(week_min/60, 1)} ч)", styles['CustomH1']))

        for d in w['Дни']:
            elements.append(Paragraph(f"День {d['День']} (Занятие: {round(d['Занято_мин']/60, 1)} ч)", styles['CustomDay']))

            for idx, mat_item in enumerate(d['План_объекты']):
                mat_elements = []
                row = mat_item['row']
                time_h = round(mat_item['time_min']/60, 1)

                subj = str(row.get('subject', ''))
                if not subj or subj.lower() == 'nan': subj = "Общий материал"

                topic = str(row.get('topic', ''))
                if not topic or topic.lower() == 'nan':
                    topic = str(row.get('file_name', 'Без названия'))
                    if not topic or topic.lower() == 'nan': topic = "Материал курса"

                part_label = mat_item.get('part_label', '')
                if part_label: part_label = f" {part_label}"

                # Title
                title = f"{idx+1}. {subj}: {topic}{part_label} — {time_h} ч"
                mat_elements.append(Paragraph(title, styles['CustomItemTitle']))

                # Meta info
                diff = str(row.get('label_difficulty', 'Средняя'))
                wc = int(row.get('word_count', 0))
                img_c = int(pd.to_numeric(row.get('images_count', 0), errors='coerce')) if not pd.isna(pd.to_numeric(row.get('images_count', 0), errors='coerce')) else 0
                prac = round(pd.to_numeric(row.get('practice_orientation', 0), errors='coerce') * 100) if not pd.isna(pd.to_numeric(row.get('practice_orientation', 0), errors='coerce')) else 0

                meta_text = f"Сложность: <b>{diff}</b> | Объем: {wc} слов | Иллюстраций: {img_c} | Практичность: {prac}%"
                mat_elements.append(Paragraph(meta_text, styles['CustomMeta']))

                # Annotation
                annotation = str(row.get('annotation', ''))
                if not annotation or annotation == "nan" or annotation.strip() == "":
                    text_content = str(row.get('text_content', ''))
                    if text_content and text_content != "nan":
                        annotation = text_content[:200].replace('\n', ' ') + "..."
                    else:
                        annotation = "Нет детального описания."

                mat_elements.append(Paragraph(annotation, styles['CustomBody']))
                elements.append(KeepTogether(mat_elements))

            elements.append(Spacer(1, 0.1*inch))

    doc.build(elements, onFirstPage=add_background, onLaterPages=add_background)
    buffer.seek(0)
    return buffer.getvalue()

def generate_course_zip(weeks_data):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for w in weeks_data:
            for d in w['Дни']:
                folder = f"Неделя_{w['Неделя']}/День_{d['День']}"
                for idx, mat_item in enumerate(d['План_объекты']):
                    row = mat_item['row']
                    title = f"{row.get('topic', row.get('file_name', 'material'))}".replace('/', '_').replace('\\', '_').replace(':', '_')[:50]

                    part_label = mat_item.get('part_label', '')
                    if part_label: title += f"_{part_label}"

                    filename = f"{folder}/{idx+1}_{title}.txt"
                    text_content = str(row.get('text_content', 'Текст материала отсутствует.'))
                    if text_content == "nan": text_content = ""
                    zf.writestr(filename, text_content.encode('utf-8'))
    buffer.seek(0)
    return buffer.getvalue()

with tabs[3]:
    st.header("3.4 Персональный ежедневный план (Конструктор)")
    st.info("Спланируйте свою учебу. Укажите, сколько часов и дней в неделю у вас есть, и алгоритм непрерывно распределит материалы по дням.")

    df_traj = df_raw.copy()
    if 'label_difficulty' not in df_traj.columns or df_traj['label_difficulty'].isna().all():
        df_traj['label_difficulty'] = "Средняя"
    else:
        df_traj['label_difficulty'] = df_traj['label_difficulty'].fillna("Средняя")

    df_traj['est_time_min'] = df_traj.apply(estimate_material_time, axis=1)

    all_subjects = sorted(df_traj['subject'].dropna().unique().tolist())

    col_input1, col_input2, col_input3 = st.columns(3)
    with col_input1:
        known = st.multiselect("Уже знаю (пропустить):", all_subjects)
    with col_input2:
        hours_per_week = st.slider("Сколько часов в неделю?", 2, 40, 10)
    with col_input3:
        days_per_week = st.slider("Сколько дней в неделю?", 1, 7, 5)

    if st.button("Создать ежедневное расписание"):
        remaining = df_traj[~df_traj['subject'].isin(known)].copy()
        if remaining.empty:
            st.success("Вы уже все знаете! Отдыхайте.")
        else:
            with st.spinner("Распределяем материалы по дням..."):
                remaining = remaining.sort_values(by=['subject', 'complexity'], ascending=[True, True])

                daily_budget_min = (hours_per_week * 60) / days_per_week

                weeks = []
                current_week = 1
                current_day = 1
                current_day_spent = 0

                current_week_days = []
                current_day_plan = []

                for _, row in remaining.iterrows():
                    time_left = row['est_time_min']
                    part_num = 1

                    while time_left > 0.1:
                        available_in_day = daily_budget_min - current_day_spent
                        if available_in_day <= 1:
                            if current_day_plan:
                                current_week_days.append({"День": current_day, "План_объекты": current_day_plan, "Занято_мин": current_day_spent})
                            current_day += 1
                            current_day_spent = 0
                            current_day_plan = []
                            if current_day > days_per_week:
                                if current_week_days:
                                    weeks.append({"Неделя": current_week, "Дни": current_week_days})
                                current_week += 1
                                current_day = 1
                                current_week_days = []
                            continue

                        take_time = min(time_left, available_in_day)
                        is_partial = (take_time < row['est_time_min'])

                        part_label = ""
                        if is_partial:
                            if time_left == row['est_time_min']:
                                part_label = "(Начало)"
                            elif abs(take_time - time_left) < 1:
                                part_label = "(Завершение)"
                            else:
                                part_label = f"(Продолжение {part_num})"

                        current_day_plan.append({
                            'row': row,
                            'time_min': take_time,
                            'part_label': part_label
                        })

                        current_day_spent += take_time
                        time_left -= take_time
                        part_num += 1

                        if time_left > 0.1:
                            if current_day_plan:
                                current_week_days.append({"День": current_day, "План_объекты": current_day_plan, "Занято_мин": current_day_spent})
                            current_day += 1
                            current_day_spent = 0
                            current_day_plan = []
                            if current_day > days_per_week:
                                if current_week_days:
                                    weeks.append({"Неделя": current_week, "Дни": current_week_days})
                                current_week += 1
                                current_day = 1
                                current_week_days = []

                # flush remaining
                if current_day_plan:
                    current_week_days.append({"День": current_day, "План_объекты": current_day_plan, "Занято_мин": current_day_spent})
                if current_week_days:
                    weeks.append({"Неделя": current_week, "Дни": current_week_days})

                st.subheader("Ежедневный план обучения")
                st.write(f"**Длительность курса:** {len(weeks)} недель(и)")

                # --- EXPORT BUTTONS ---
                st.markdown("### Экспорт курса")
                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                    pdf_bytes = generate_course_pdf(weeks)
                    st.download_button(
                        label="Скачать ежедневный план (PDF)",
                        data=pdf_bytes,
                        file_name="Daily_Schedule.pdf",
                        mime="application/pdf",
                        width="stretch"
                    )
                with col_exp2:
                    zip_bytes = generate_course_zip(weeks)
                    st.download_button(
                        label="Скачать файлы по дням (ZIP)",
                        data=zip_bytes,
                        file_name="Daily_Materials.zip",
                        mime="application/zip",
                        width="stretch"
                    )
                st.markdown("---")

                # --- SCHEDULE UI ---
                for w in weeks:
                    with st.container():
                        week_min = sum(d['Занято_мин'] for d in w['Дни'])
                        st.markdown(f"## Неделя {w['Неделя']} (Нагрузка: {round(week_min/60, 1)} ч)")

                        for d in w['Дни']:
                            with st.expander(f"День {d['День']} (Занятие: {round(d['Занято_мин']/60, 1)} ч)", expanded=(w['Неделя']==1 and d['День']==1)):
                                for idx, item in enumerate(d['План_объекты']):
                                    row = item['row']
                                    time_h = round(item['time_min']/60, 1)

                                    subj = str(row.get('subject', ''))
                                    if not subj or subj.lower() == 'nan': subj = "Общий материал"

                                    topic_name = str(row.get('topic', ''))
                                    if not topic_name or topic_name.lower() == 'nan':
                                        topic_name = str(row.get('file_name', 'Без названия'))
                                        if not topic_name or topic_name.lower() == 'nan': topic_name = "Материал курса"

                                    topic_name = topic_name[:80]
                                    part_label = item.get('part_label', '')
                                    if part_label: part_label = f" **{part_label}**"

                                    col_m1, col_m2 = st.columns([4, 1])
                                    with col_m1:
                                        st.markdown(f"**{idx+1}. {subj}**: {topic_name}{part_label} — *{time_h} ч* (Сложность: {row.get('label_difficulty', 'Средняя')})")
                                    with col_m2:
                                        text_content = str(row.get('text_content', 'Текст отсутствует.'))
                                        if text_content == "nan": text_content = ""
                                        st.download_button(
                                            label="Скачать TXT",
                                            data=text_content.encode('utf-8'),
                                            file_name=f"{topic_name.replace('/', '_')[:30]}.txt",
                                            mime="text/plain",
                                            key=f"dl_{w['Неделя']}_{d['День']}_{idx}_{row['id']}"
                                        )
                        st.markdown("---")

                        st.markdown("---")
