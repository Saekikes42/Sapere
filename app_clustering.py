from config import DATABASE_URL, OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN
import os
import json
import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- CONFIG & DATABASE ---
DB_URL = DATABASE_URL
AI_API_KEY = OPENROUTER_API_KEY
engine = create_engine(DB_URL)

# --- INIT DATABASE COLUMNS FOR LABELS ---
def init_label_columns():
    try:
        with engine.connect() as conn:
            res = conn.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'opit_files'"
            ))
            cols = [r[0] for r in res.fetchall()]

        new_cols = [
            ("label_parallel_id", "TEXT"),
            ("label_sequential_id", "TEXT"),
            ("label_difficulty", "TEXT")
        ]
        for col_name, col_type in new_cols:
            if col_name not in cols:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE opit_files ADD COLUMN {col_name} {col_type}"))
    except Exception as e:
        st.error(f"Ошибка инициализации колонок разметки в БД: {e}")

init_label_columns()

# --- LOAD DATA ---
@st.cache_data(show_spinner=False)
def load_data_from_db():
    try:
        df = pd.read_sql("SELECT * FROM opit_files", engine)
        if df.empty:
            return pd.DataFrame()

        # Programmatically calculate word count and complexity
        df['word_count'] = df['text_content'].fillna("").apply(lambda x: len(x.split()))
        df['text_length'] = pd.to_numeric(df['text_length'], errors='coerce').fillna(0.0)
        df['compliance_rate'] = pd.to_numeric(df['compliance_rate'], errors='coerce').fillna(0.0)
        df['complexity'] = df['word_count'] * 0.05 + df['text_length'] * 0.01

        # Categorize generated vs original
        df['is_generated'] = df['file_name'].str.startswith('AI_') | df['annotation'].str.contains('Сгенерировано', na=False)

        # Ensure numerical values are parsed
        num_cols = [
            'compliance_rate', 'compliance_structure', 'compliance_clarity',
            'compliance_terminology', 'compliance_completeness',
            'practice_orientation', 'legal_compliance', 'technical_quality'
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных из БД: {e}")
        return pd.DataFrame()

# --- HELPERS FOR METRICS & PROJECTIONS ---
def calculate_metrics(X, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2 or n_clusters >= len(labels):
        return {
            "Silhouette": "Н/Д",
            "Calinski": "Н/Д",
            "Davies-Bouldin": "Н/Д",
            "n_clusters": n_clusters
        }

    try:
        sil = round(silhouette_score(X, labels), 3)
        cal = round(calinski_harabasz_score(X, labels), 1)
        db = round(davies_bouldin_score(X, labels), 3)
        return {
            "Silhouette": sil,
            "Calinski": cal,
            "Davies-Bouldin": db,
            "n_clusters": n_clusters
        }
    except Exception:
        return {
            "Silhouette": "Ошибка",
            "Calinski": "Ошибка",
            "Davies-Bouldin": "Ошибка",
            "n_clusters": n_clusters
        }

# --- AI AGENT: OPENROUTER CLUSTERING ---
def call_openrouter_labeling(materials_list):
    prompt = f"""Ты - эксперт по интеллектуальному анализу данных и методист. Перед тобой список учебных материалов:
{json.dumps(materials_list, ensure_ascii=False)}

Тебе необходимо выполнить полуавтоматическую разметку этих материалов для будущих рекомендательных систем по трем критериям:
1. Группы для совместного (параллельного) изучения. Совместное изучение возможно только если темы материалов находятся в независимых последовательностях учебных предметов (дисциплин). Например, изучение темы по Истории и темы по Физике. Присвой каждой группе уникальный parallel_id (например, P1, P2, P3...).
2. Группы для последовательного изучения. Материалы могут быть как из одного, так и из нескольких предметов, если они являются логическим продолжением (например, базовые сети -> маршрутизация -> сетевая безопасность). Присвой каждой последовательности уникальный sequential_id (например, S1, S2, S3...).
3. Сложность освоения. Распредели материалы по сложности (Низкая / Средняя / Высокая) на основе их тем, аннотаций и объема.

Ответь строго в формате корректного JSON со следующей структурой:
{{
  "labels": [
    {{
      "id": "хэш-id-файла",
      "parallel_id": "P1",
      "sequential_id": "S1",
      "difficulty": "Средняя"
    }}
  ],
  "explanation": "Подробное методическое обоснование распределения по группам параллельного и последовательного изучения, а также критериев сложности."
}}

Не добавляй никакого другого текста, разметки markdown или описаний вне JSON block. Отвечай на русском языке."""

    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            content = r.json()['choices'][0]['message']['content'].strip()
            s = content.find('{')
            e = content.rfind('}')
            if s != -1 and e != -1:
                return json.loads(content[s:e+1])
        else:
            st.error(f"OpenRouter API Error: {r.status_code} {r.text}")
    except Exception as e:
        st.error(f"Исключение при вызове OpenRouter API: {e}")
    return None

# --- STREAMLIT PAGE CONFIG ---
from PIL import Image
icon = Image.open("assets/icon_clustering.png") if os.path.exists("assets/icon_clustering.png") else None
st.set_page_config(
    page_title="Sapere Intelligence - Разметка набора данных",
    page_icon=icon,
    layout="wide"
)

# Premium Matte Dark CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    h1, h2, h3, h4, h5 {
        color: #f9a03f !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #161b22;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 10px 24px;
        color: #8b949e !important;
        transition: all 0.2s ease;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #f9a03f !important;
        font-weight: 600 !important;
    }

    .stButton>button {
        background-color: #21262d;
        color: #f9a03f !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #30363d;
        border-color: #f9a03f !important;
    }

    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #f9a03f;
    }
    .metric-label {
        font-size: 12px;
        color: #8b949e;
    }

    .card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Разметка и анализ набора данных")
st.caption("Автоматическая разметка учебных материалов по сложности, последовательному и параллельному изучению")

df_raw = load_data_from_db()

if df_raw.empty:
    st.warning("В базе данных нет записей. Пожалуйста, проверьте подключение к PostgreSQL.")
else:
    # Sidebar control panel
    st.sidebar.header("Панель управления")
    st.sidebar.write(f"Загружено записей из БД: **{len(df_raw)}**")

    if st.sidebar.button("Обновить данные из БД"):
        st.cache_data.clear()
        st.rerun()

    # Pre-calculate features and dimensionality reduction globally
    max_feats = st.sidebar.slider("Макс. признаков TF-IDF", 50, 500, 150, step=10)

    # Global projection type for classic visuals (PCA vs t-SNE)
    proj_type = st.sidebar.radio("Проекция для 9 визуализаций:", ["PCA", "t-SNE"])

    df_raw['combined_text'] = (
        df_raw['subject'].fillna("") + " " +
        df_raw['topic'].fillna("") + " " +
        df_raw['annotation'].fillna("") + " " +
        df_raw['text_content'].fillna("")
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_feats, stop_words=None)
    X_text = vectorizer.fit_transform(df_raw['combined_text']).toarray()

    # PCA & t-SNE for text
    pca_t = PCA(n_components=2, random_state=42)
    X_pca_t = pca_t.fit_transform(X_text)
    df_raw['PCA_text_1'] = X_pca_t[:, 0]
    df_raw['PCA_text_2'] = X_pca_t[:, 1]

    tsne_perp = min(5, max(2, len(df_raw) // 5))
    tsne_t = TSNE(n_components=2, perplexity=tsne_perp, random_state=42, init='random')
    X_tsne_t = tsne_t.fit_transform(X_text)
    df_raw['tSNE_text_1'] = X_tsne_t[:, 0]
    df_raw['tSNE_text_2'] = X_tsne_t[:, 1]

    # Difficulty Space
    scaler = StandardScaler()
    X_diff = scaler.fit_transform(df_raw[['complexity', 'word_count', 'compliance_rate']].fillna(0.0))

    # PCA & t-SNE for difficulty
    pca_d = PCA(n_components=2, random_state=42)
    X_pca_d = pca_d.fit_transform(X_diff)
    df_raw['PCA_diff_1'] = X_pca_d[:, 0]
    df_raw['PCA_diff_2'] = X_pca_d[:, 1]

    tsne_d = TSNE(n_components=2, perplexity=tsne_perp, random_state=42, init='random')
    X_tsne_d = tsne_d.fit_transform(X_diff)
    df_raw['tSNE_diff_1'] = X_tsne_d[:, 0]
    df_raw['tSNE_diff_2'] = X_tsne_d[:, 1]

    # Select Mode
    mode = st.radio(
        "Выберите вариант разметки:",
        [
            "Вариант 1: Интеллектуальная разметка ИИ (OpenRouter)",
            "Вариант 2: Классическое машинное обучение (TF-IDF / Числовые метрики)"
        ],
        horizontal=True
    )

    st.markdown("---")

    # --- VARIANT 1: AI LABELS ---
    if "Вариант 1" in mode:
        st.subheader("Вариант 1: Семантическая разметка через ИИ")
        st.info("ИИ-агент проанализирует семантический контекст учебных материалов и разобьет их на модули для параллельного и последовательного изучения, а также присвоит уровень сложности.")

        col_btn1, col_btn2 = st.columns([1, 2])
        with col_btn1:
            run_ai = st.button("Запустить ИИ-разметку")

        if run_ai:
            with st.spinner("ИИ выполняет семантическую разметку через OpenRouter..."):
                materials_payload = []
                for _, row in df_raw.iterrows():
                    subj = str(row["subject"]) if pd.notna(row["subject"]) else "Не указан"
                    top = str(row["topic"]) if pd.notna(row["topic"]) else "Не указана"
                    annot = str(row["annotation"])[:250] if pd.notna(row["annotation"]) else ""
                    materials_payload.append({
                        "id": row["id"],
                        "file_name": row["file_name"],
                        "subject": subj,
                        "topic": top,
                        "annotation": annot
                    })

                ai_result = call_openrouter_labeling(materials_payload)
                if ai_result and "labels" in ai_result:
                    st.session_state["ai_labels_data"] = ai_result["labels"]
                    st.session_state["ai_explanation"] = ai_result.get("explanation", "Обоснование отсутствует.")
                    st.success("ИИ-разметка успешно завершена!")
                else:
                    st.error("Не удалось выполнить ИИ-разметку. Проверьте логи.")

        if "ai_labels_data" in st.session_state:
            ai_df = pd.DataFrame(st.session_state["ai_labels_data"])
            df_ai = df_raw.merge(ai_df, on="id", how="left")

            df_ai['parallel_id'] = df_ai['parallel_id'].fillna("N/A")
            df_ai['sequential_id'] = df_ai['sequential_id'].fillna("N/A")
            df_ai['difficulty'] = df_ai['difficulty'].fillna("Средняя")

            st.markdown(f"**Методическое обоснование ИИ:**")
            st.markdown(f"> *{st.session_state['ai_explanation']}*")

            ai_tabs = st.tabs([
                "Совместное (параллельное) изучение",
                "Последовательное изучение",
                "Сложность освоения"
            ])

            # AI Tab 1: Parallel Study
            with ai_tabs[0]:
                st.write("#### Метрики и 2D-визуализация: Параллельное изучение (ИИ)")

                p_metrics = calculate_metrics(X_text, df_ai['parallel_id'])
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{p_metrics['Silhouette']}</div><div class='metric-label'>Silhouette Score (Силуэт)</div></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{p_metrics['Calinski']}</div><div class='metric-label'>Calinski-Harabasz Index</div></div>", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{p_metrics['Davies-Bouldin']}</div><div class='metric-label'>Davies-Bouldin Index</div></div>", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{p_metrics['n_clusters']}</div><div class='metric-label'>Число выделенных ИИ-групп</div></div>", unsafe_allow_html=True)

                cp1, cp2 = st.columns(2)
                with cp1:
                    fig_pca = px.scatter(
                        df_ai, x='PCA_text_1', y='PCA_text_2', color='parallel_id',
                        hover_name='file_name', hover_data=['subject', 'topic'],
                        title='Проекция PCA (TF-IDF пространство)',
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    fig_pca.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                    st.plotly_chart(fig_pca, width="stretch")
                with cp2:
                    fig_tsne = px.scatter(
                        df_ai, x='tSNE_text_1', y='tSNE_text_2', color='parallel_id',
                        hover_name='file_name', hover_data=['subject', 'topic'],
                        title='Проекция t-SNE (TF-IDF пространство)',
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    fig_tsne.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                    st.plotly_chart(fig_tsne, width="stretch")

                st.write("**Данные разметки:**")
                st.dataframe(df_ai[['file_name', 'subject', 'topic', 'parallel_id']])

            # AI Tab 2: Sequential Study
            with ai_tabs[1]:
                st.write("#### Метрики и 2D-визуализация: Последовательное изучение (ИИ)")

                s_metrics = calculate_metrics(X_text, df_ai['sequential_id'])
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{s_metrics['Silhouette']}</div><div class='metric-label'>Silhouette Score (Силуэт)</div></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{s_metrics['Calinski']}</div><div class='metric-label'>Calinski-Harabasz Index</div></div>", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{s_metrics['Davies-Bouldin']}</div><div class='metric-label'>Davies-Bouldin Index</div></div>", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{s_metrics['n_clusters']}</div><div class='metric-label'>Число выделенных ИИ-цепочек</div></div>", unsafe_allow_html=True)

                cp1, cp2 = st.columns(2)
                with cp1:
                    fig_pca = px.scatter(
                        df_ai, x='PCA_text_1', y='PCA_text_2', color='sequential_id',
                        hover_name='file_name', hover_data=['subject', 'topic'],
                        title='Проекция PCA (TF-IDF пространство)',
                        color_discrete_sequence=px.colors.qualitative.Prism
                    )
                    fig_pca.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                    st.plotly_chart(fig_pca, width="stretch")
                with cp2:
                    fig_tsne = px.scatter(
                        df_ai, x='tSNE_text_1', y='tSNE_text_2', color='sequential_id',
                        hover_name='file_name', hover_data=['subject', 'topic'],
                        title='Проекция t-SNE (TF-IDF пространство)',
                        color_discrete_sequence=px.colors.qualitative.Prism
                    )
                    fig_tsne.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                    st.plotly_chart(fig_tsne, width="stretch")

                st.write("**Данные разметки:**")
                st.dataframe(df_ai[['file_name', 'subject', 'topic', 'sequential_id']])

            # AI Tab 3: Difficulty Levels
            with ai_tabs[2]:
                st.write("#### Метрики и 2D-визуализация: Сложность освоения (ИИ)")

                d_metrics = calculate_metrics(X_diff, df_ai['difficulty'])
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{d_metrics['Silhouette']}</div><div class='metric-label'>Silhouette Score (Силуэт)</div></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{d_metrics['Calinski']}</div><div class='metric-label'>Calinski-Harabasz Index</div></div>", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{d_metrics['Davies-Bouldin']}</div><div class='metric-label'>Davies-Bouldin Index</div></div>", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"<div class='metric-card'><div class='metric-value'>{d_metrics['n_clusters']}</div><div class='metric-label'>Классы сложности</div></div>", unsafe_allow_html=True)

                cp1, cp2 = st.columns(2)
                with cp1:
                    fig_pca = px.scatter(
                        df_ai, x='PCA_diff_1', y='PCA_diff_2', color='difficulty',
                        hover_name='file_name', hover_data=['subject', 'topic', 'complexity'],
                        title='Проекция PCA (Метрики сложности)',
                        color_discrete_map={"Низкая": "#58a6ff", "Средняя": "#f9a03f", "Высокая": "#ff7b72"}
                    )
                    fig_pca.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                    st.plotly_chart(fig_pca, width="stretch")
                with cp2:
                    fig_tsne = px.scatter(
                        df_ai, x='tSNE_diff_1', y='tSNE_diff_2', color='difficulty',
                        hover_name='file_name', hover_data=['subject', 'topic', 'complexity'],
                        title='Проекция t-SNE (Метрики сложности)',
                        color_discrete_map={"Низкая": "#58a6ff", "Средняя": "#f9a03f", "Высокая": "#ff7b72"}
                    )
                    fig_tsne.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                    st.plotly_chart(fig_tsne, width="stretch")

                st.write("**Данные разметки:**")
                st.dataframe(df_ai[['file_name', 'subject', 'topic', 'word_count', 'complexity', 'difficulty']])

            st.session_state["final_labels_df"] = df_ai[['id', 'parallel_id', 'sequential_id', 'difficulty']].rename(
                columns={'parallel_id': 'label_parallel_id', 'sequential_id': 'label_sequential_id', 'difficulty': 'label_difficulty'}
            )
            st.session_state["active_metrics"] = {
                "parallel": p_metrics,
                "sequential": s_metrics,
                "difficulty": d_metrics
            }

    # --- VARIANT 2: CLASSIC ML ---
    else:
        st.subheader("Вариант 2: Классическая кластеризация (3 алгоритма)")
        st.info("Разметка производится с помощью векторов TF-IDF и численных признаков с использованием K-Means, Иерархической кластеризации (Agglomerative) и DBSCAN.")

        classic_tabs = st.tabs([
            "Совместное (параллельное) изучение",
            "Последовательное изучение",
            "Сложность освоения"
        ])

        if "labels_parallel" not in st.session_state:
            st.session_state["labels_parallel"] = None
        if "labels_sequential" not in st.session_state:
            st.session_state["labels_sequential"] = None
        if "labels_difficulty" not in st.session_state:
            st.session_state["labels_difficulty"] = None

        # Determine projection columns based on sidebar input
        x_text_col = 'PCA_text_1' if proj_type == 'PCA' else 'tSNE_text_1'
        y_text_col = 'PCA_text_2' if proj_type == 'PCA' else 'tSNE_text_2'

        x_diff_col = 'PCA_diff_1' if proj_type == 'PCA' else 'tSNE_diff_1'
        y_diff_col = 'PCA_diff_2' if proj_type == 'PCA' else 'tSNE_diff_2'

        proj_label = 'PCA' if proj_type == 'PCA' else 't-SNE'

        # TAB 1: Parallel Study
        with classic_tabs[0]:
            st.write("#### Классическая кластеризация для выявления параллельных курсов")

            # Sliders
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.markdown("**K-Means**")
                k_p_kmeans = st.slider("Число кластеров (K-Means) ", 2, 10, 4, key="k_p_km")
            with col_p2:
                st.markdown("**Agglomerative**")
                k_p_agg = st.slider("Число кластеров (Hierarchical) ", 2, 10, 4, key="k_p_ag")
            with col_p3:
                st.markdown("**DBSCAN**")
                eps_p_db = st.slider("Радиус eps ", 0.2, 2.0, 1.1, step=0.1, key="eps_p")
                min_p_db = st.slider("Мин. точек в ядре ", 1, 5, 2, key="min_p")

            # Run Algorithms
            lbl_km = KMeans(n_clusters=k_p_kmeans, random_state=42, n_init='auto').fit_predict(X_text)
            lbl_agg = AgglomerativeClustering(n_clusters=k_p_agg).fit_predict(X_text)
            lbl_db = DBSCAN(eps=eps_p_db, min_samples=min_p_db).fit_predict(X_text)

            # Calculate metrics
            m_km = calculate_metrics(X_text, lbl_km)
            m_agg = calculate_metrics(X_text, lbl_agg)
            m_db = calculate_metrics(X_text, lbl_db)

            # Comparison table
            df_m = pd.DataFrame({
                "Алгоритм": ["K-Means", "Agglomerative", "DBSCAN"],
                "Silhouette Score ↑": [m_km["Silhouette"], m_agg["Silhouette"], m_db["Silhouette"]],
                "Calinski-Harabasz Index ↑": [m_km["Calinski"], m_agg["Calinski"], m_db["Calinski"]],
                "Davies-Bouldin Index ↓": [m_km["Davies-Bouldin"], m_agg["Davies-Bouldin"], m_db["Davies-Bouldin"]],
                "Число кластеров": [m_km["n_clusters"], m_agg["n_clusters"], m_db["n_clusters"]]
            })
            st.table(df_m)

            st.write("##### Визуальный анализ моделей (3 метода):")
            # 3 visual columns side-by-side
            vcol1, vcol2, vcol3 = st.columns(3)

            with vcol1:
                df_km = df_raw.copy()
                df_km['cluster'] = lbl_km
                fig_km = px.scatter(
                    df_km, x=x_text_col, y=y_text_col, color=df_km['cluster'].astype(str),
                    hover_name='file_name', hover_data=['subject', 'topic'],
                    title=f"K-Means ({proj_label})",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_km.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                st.plotly_chart(fig_km, width="stretch")

            with vcol2:
                df_agg = df_raw.copy()
                df_agg['cluster'] = lbl_agg
                fig_agg = px.scatter(
                    df_agg, x=x_text_col, y=y_text_col, color=df_agg['cluster'].astype(str),
                    hover_name='file_name', hover_data=['subject', 'topic'],
                    title=f"Agglomerative ({proj_label})",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_agg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                st.plotly_chart(fig_agg, width="stretch")

            with vcol3:
                df_db = df_raw.copy()
                df_db['cluster'] = lbl_db
                fig_db = px.scatter(
                    df_db, x=x_text_col, y=y_text_col, color=df_db['cluster'].astype(str),
                    hover_name='file_name', hover_data=['subject', 'topic'],
                    title=f"DBSCAN ({proj_label})",
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_db.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                st.plotly_chart(fig_db, width="stretch")

            # Selection for database saving
            sel_algo_p = st.radio(
                "Выберите алгоритм для сохранения разметки параллельного изучения в БД:",
                ["K-Means", "Agglomerative", "DBSCAN"],
                key="sel_algo_p"
            )

            lbls_selected_p = lbl_km if sel_algo_p == "K-Means" else (lbl_agg if sel_algo_p == "Agglomerative" else lbl_db)
            df_p = df_raw.copy()
            df_p['cluster'] = lbls_selected_p

            parallel_mapping = {}
            for cl in np.unique(lbls_selected_p):
                cl_rows = df_p[df_p['cluster'] == cl]
                unique_subjs = cl_rows['subject'].dropna().unique()
                if len(unique_subjs) > 1:
                    parallel_mapping[cl] = f"P_{cl} ({' + '.join(list(unique_subjs)[:2])})"
                else:
                    parallel_mapping[cl] = f"P_{cl} (Монопредметный)"

            df_p['label_parallel_id'] = df_p['cluster'].map(parallel_mapping)
            st.session_state["labels_parallel"] = df_p[['id', 'label_parallel_id']]

            st.write("Результат полуавтоматической разметки параллельного изучения:")
            st.dataframe(df_p[['file_name', 'subject', 'topic', 'cluster', 'label_parallel_id']])

        # TAB 2: Sequential Study
        with classic_tabs[1]:
            st.write("#### Классическая кластеризация для выявления последовательных траекторий")

            # Sliders
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.markdown("**K-Means**")
                k_s_kmeans = st.slider("Число кластеров (K-Means)  ", 2, 10, 5, key="k_s_km")
            with col_s2:
                st.markdown("**Agglomerative**")
                k_s_agg = st.slider("Число кластеров (Hierarchical)  ", 2, 10, 5, key="k_s_ag")
            with col_s3:
                st.markdown("**DBSCAN**")
                eps_s_db = st.slider("Радиус eps  ", 0.2, 2.0, 1.0, step=0.1, key="eps_s")
                min_s_db = st.slider("Мин. точек в ядре  ", 1, 5, 2, key="min_s")

            # Run Algorithms
            lbl_km_s = KMeans(n_clusters=k_s_kmeans, random_state=42, n_init='auto').fit_predict(X_text)
            lbl_agg_s = AgglomerativeClustering(n_clusters=k_s_agg).fit_predict(X_text)
            lbl_db_s = DBSCAN(eps=eps_s_db, min_samples=min_s_db).fit_predict(X_text)

            # Calculate metrics
            m_km_s = calculate_metrics(X_text, lbl_km_s)
            m_agg_s = calculate_metrics(X_text, lbl_agg_s)
            m_db_s = calculate_metrics(X_text, lbl_db_s)

            # Comparison table
            df_m_s = pd.DataFrame({
                "Алгоритм": ["K-Means", "Agglomerative", "DBSCAN"],
                "Silhouette Score ↑": [m_km_s["Silhouette"], m_agg_s["Silhouette"], m_db_s["Silhouette"]],
                "Calinski-Harabasz Index ↑": [m_km_s["Calinski"], m_agg_s["Calinski"], m_db_s["Calinski"]],
                "Davies-Bouldin Index ↓": [m_km_s["Davies-Bouldin"], m_agg_s["Davies-Bouldin"], m_db_s["Davies-Bouldin"]],
                "Число кластеров": [m_km_s["n_clusters"], m_agg_s["n_clusters"], m_db_s["n_clusters"]]
            })
            st.table(df_m_s)

            st.write("##### Визуальный анализ моделей (3 метода):")
            # 3 visual columns side-by-side
            vcol1_s, vcol2_s, vcol3_s = st.columns(3)

            with vcol1_s:
                df_km_s = df_raw.copy()
                df_km_s['cluster'] = lbl_km_s
                fig_km_s = px.scatter(
                    df_km_s, x=x_text_col, y=y_text_col, color=df_km_s['cluster'].astype(str),
                    hover_name='file_name', hover_data=['subject', 'topic'],
                    title=f"K-Means ({proj_label})",
                    color_discrete_sequence=px.colors.qualitative.D3
                )
                fig_km_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                st.plotly_chart(fig_km_s, width="stretch")

            with vcol2_s:
                df_agg_s = df_raw.copy()
                df_agg_s['cluster'] = lbl_agg_s
                fig_agg_s = px.scatter(
                    df_agg_s, x=x_text_col, y=y_text_col, color=df_agg_s['cluster'].astype(str),
                    hover_name='file_name', hover_data=['subject', 'topic'],
                    title=f"Agglomerative ({proj_label})",
                    color_discrete_sequence=px.colors.qualitative.D3
                )
                fig_agg_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                st.plotly_chart(fig_agg_s, width="stretch")

            with vcol3_s:
                df_db_s = df_raw.copy()
                df_db_s['cluster'] = lbl_db_s
                fig_db_s = px.scatter(
                    df_db_s, x=x_text_col, y=y_text_col, color=df_db_s['cluster'].astype(str),
                    hover_name='file_name', hover_data=['subject', 'topic'],
                    title=f"DBSCAN ({proj_label})",
                    color_discrete_sequence=px.colors.qualitative.D3
                )
                fig_db_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                st.plotly_chart(fig_db_s, width="stretch")

            # Selection for database saving
            sel_algo_s = st.radio(
                "Выберите алгоритм для сохранения разметки последовательного изучения в БД:",
                ["K-Means", "Agglomerative", "DBSCAN"],
                key="sel_algo_s"
            )

            lbls_selected_s = lbl_km_s if sel_algo_s == "K-Means" else (lbl_agg_s if sel_algo_s == "Agglomerative" else lbl_db_s)
            df_s = df_raw.copy()
            df_s['cluster'] = lbls_selected_s

            seq_mapping = {cl: f"S_{cl}" for cl in np.unique(lbls_selected_s)}
            df_s['label_sequential_id'] = df_s['cluster'].map(seq_mapping)
            st.session_state["labels_sequential"] = df_s[['id', 'label_sequential_id']]

            st.write("Результат полуавтоматической разметки последовательного изучения:")
            st.dataframe(df_s[['file_name', 'subject', 'topic', 'cluster', 'label_sequential_id']])

        # TAB 3: Difficulty Levels
        with classic_tabs[2]:
            st.write("#### Классическая кластеризация на основе численных метрик сложности")
            st.write("Используются признаки: объём текста (word_count), сложность материала (complexity), уровень соответствия требованиям (compliance_rate).")

            # Sliders
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.markdown("**K-Means**")
                k_d_kmeans = st.slider("Число кластеров (K-Means)   ", 2, 6, 3, key="k_d_km")
            with col_d2:
                st.markdown("**Agglomerative**")
                k_d_agg = st.slider("Число кластеров (Hierarchical)   ", 2, 6, 3, key="k_d_ag")
            with col_d3:
                st.markdown("**DBSCAN**")
                eps_d_db = st.slider("Радиус eps   ", 0.1, 3.0, 0.8, step=0.1, key="eps_d")
                min_d_db = st.slider("Мин. точек в ядре   ", 1, 5, 2, key="min_d")

            # Run Algorithms
            lbl_km_d = KMeans(n_clusters=k_d_kmeans, random_state=42, n_init='auto').fit_predict(X_diff)
            lbl_agg_d = AgglomerativeClustering(n_clusters=k_d_agg).fit_predict(X_diff)
            lbl_db_d = DBSCAN(eps=eps_d_db, min_samples=min_d_db).fit_predict(X_diff)

            # Calculate metrics
            m_km_d = calculate_metrics(X_diff, lbl_km_d)
            m_agg_d = calculate_metrics(X_diff, lbl_agg_d)
            m_db_d = calculate_metrics(X_diff, lbl_db_d)

            # Comparison table
            df_m_d = pd.DataFrame({
                "Алгоритм": ["K-Means", "Agglomerative", "DBSCAN"],
                "Silhouette Score ↑": [m_km_d["Silhouette"], m_agg_d["Silhouette"], m_db_d["Silhouette"]],
                "Calinski-Harabasz Index ↑": [m_km_d["Calinski"], m_agg_d["Calinski"], m_db_d["Calinski"]],
                "Davies-Bouldin Index ↓": [m_km_d["Davies-Bouldin"], m_agg_d["Davies-Bouldin"], m_db_d["Davies-Bouldin"]],
                "Число кластеров": [m_km_d["n_clusters"], m_agg_d["n_clusters"], m_db_d["n_clusters"]]
            })
            st.table(df_m_d)

            st.write("##### Визуальный анализ моделей (3 метода):")
            # 3 visual columns side-by-side
            vcol1_d, vcol2_d, vcol3_d = st.columns(3)

            with vcol1_d:
                df_km_d = df_raw.copy()
                df_km_d['cluster'] = lbl_km_d
                fig_km_d = px.scatter(
                    df_km_d, x=x_diff_col, y=y_diff_col, color=df_km_d['cluster'].astype(str),
                    hover_name='file_name', hover_data=['subject', 'topic', 'complexity'],
                    title=f"K-Means ({proj_label})",
                    color_discrete_sequence=px.colors.qualitative.Alphabet
                )
                fig_km_d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                st.plotly_chart(fig_km_d, width="stretch")

            with vcol2_d:
                df_agg_d = df_raw.copy()
                df_agg_d['cluster'] = lbl_agg_d
                fig_agg_d = px.scatter(
                    df_agg_d, x=x_diff_col, y=y_diff_col, color=df_agg_d['cluster'].astype(str),
                    hover_name='file_name', hover_data=['subject', 'topic', 'complexity'],
                    title=f"Agglomerative ({proj_label})",
                    color_discrete_sequence=px.colors.qualitative.Alphabet
                )
                fig_agg_d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                st.plotly_chart(fig_agg_d, width="stretch")

            with vcol3_d:
                df_db_d = df_raw.copy()
                df_db_d['cluster'] = lbl_db_d
                fig_db_d = px.scatter(
                    df_db_d, x=x_diff_col, y=y_diff_col, color=df_db_d['cluster'].astype(str),
                    hover_name='file_name', hover_data=['subject', 'topic', 'complexity'],
                    title=f"DBSCAN ({proj_label})",
                    color_discrete_sequence=px.colors.qualitative.Alphabet
                )
                fig_db_d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark')
                st.plotly_chart(fig_db_d, width="stretch")

            # Selection for database saving
            sel_algo_d = st.radio(
                "Выберите алгоритм для сохранения разметки сложности в БД:",
                ["K-Means", "Agglomerative", "DBSCAN"],
                key="sel_algo_d"
            )

            lbls_selected_d = lbl_km_d if sel_algo_d == "K-Means" else (lbl_agg_d if sel_algo_d == "Agglomerative" else lbl_db_d)
            df_d = df_raw.copy()
            df_d['cluster'] = lbls_selected_d

            mean_complexities = df_d.groupby('cluster')['complexity'].mean().sort_values()
            diff_map = {}
            for i, cluster_idx in enumerate(mean_complexities.index):
                if i == 0:
                    diff_map[cluster_idx] = "Низкая"
                elif i == len(mean_complexities) - 1:
                    diff_map[cluster_idx] = "Высокая"
                else:
                    diff_map[cluster_idx] = "Средняя"

            df_d['label_difficulty'] = df_d['cluster'].map(diff_map)
            st.session_state["labels_difficulty"] = df_d[['id', 'label_difficulty']]

            st.write("Результат полуавтоматической разметки сложности:")
            st.dataframe(df_d[['file_name', 'subject', 'topic', 'cluster', 'complexity', 'label_difficulty']])

        # Assemble Final DataFrame from choices
        if (st.session_state["labels_parallel"] is not None and
            st.session_state["labels_sequential"] is not None and
            st.session_state["labels_difficulty"] is not None):

            final_df = df_raw[['id']].merge(st.session_state["labels_parallel"], on="id")
            final_df = final_df.merge(st.session_state["labels_sequential"], on="id")
            final_df = final_df.merge(st.session_state["labels_difficulty"], on="id")
            st.session_state["final_labels_df"] = final_df
            st.session_state["active_metrics"] = {
                "parallel": m_km if sel_algo_p == "K-Means" else (m_agg if sel_algo_p == "Agglomerative" else m_db),
                "sequential": m_km_s if sel_algo_s == "K-Means" else (m_agg_s if sel_algo_s == "Agglomerative" else m_db_s),
                "difficulty": m_km_d if sel_algo_d == "K-Means" else (m_agg_d if sel_algo_d == "Agglomerative" else m_db_d)
            }

    # --- SAVE TO DATABASE SECTION ---
    st.markdown("---")
    st.subheader("Сохранение результатов разметки в базу данных")

    if "final_labels_df" not in st.session_state:
        st.warning("Пожалуйста, сначала выполните разметку (для классического варианта убедитесь, что вы прошлись по всем 3 вкладкам).")
    else:
        final_df = st.session_state["final_labels_df"]
        st.write("Данные, готовые к отправке в PostgreSQL:")
        df_preview = df_raw[['id', 'file_name', 'subject']].merge(final_df, on="id")
        st.dataframe(df_preview.head(10))
        st.write(f"Всего строк для записи: **{len(df_preview)}**")

        if st.button("Записать разметку в Postgres"):
            try:
                payload = []
                for _, row in final_df.iterrows():
                    payload.append({
                        "id": row["id"],
                        "p_id": row["label_parallel_id"],
                        "s_id": row["label_sequential_id"],
                        "diff": row["label_difficulty"]
                    })

                with engine.begin() as conn:
                    conn.execute(
                        text("""
                            UPDATE opit_files
                            SET label_parallel_id = :p_id,
                                label_sequential_id = :s_id,
                                label_difficulty = :diff
                            WHERE id = :id
                        """),
                        payload
                    )
                st.success("Разметка успешно обновлена в базе данных Postgres!")
            except Exception as e:
                st.error(f"Не удалось обновить базу данных: {e}")

    # --- TAB 3: QUALITY ANALYSIS & RECOMMENDATION ---
    st.markdown("---")
    st.subheader("2.4 Анализ качества разметки и обоснование выбора подхода")

    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        st.write("Метрики стабильности и качества разметки (Выбранный вариант):")
        if "active_metrics" in st.session_state:
            active_m = st.session_state["active_metrics"]
            metrics_comparison = pd.DataFrame({
                "Критерий разметки": ["Параллельное изучение", "Последовательное изучение", "Сложность освоения"],
                "Silhouette Score (Силуэт) ↑": [active_m["parallel"]["Silhouette"], active_m["sequential"]["Silhouette"], active_m["difficulty"]["Silhouette"]],
                "Calinski-Harabasz Index ↑": [active_m["parallel"]["Calinski"], active_m["sequential"]["Calinski"], active_m["difficulty"]["Calinski"]],
                "Davies-Bouldin Index ↓": [active_m["parallel"]["Davies-Bouldin"], active_m["sequential"]["Davies-Bouldin"], active_m["difficulty"]["Davies-Bouldin"]],
                "Число кластеров": [active_m["parallel"]["n_clusters"], active_m["sequential"]["n_clusters"], active_m["difficulty"]["n_clusters"]]
            })
            st.table(metrics_comparison)
        else:
            st.info("Выполните разметку, чтобы увидеть результирующие метрики.")
