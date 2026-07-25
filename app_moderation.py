from config import DATABASE_URL, OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN
import os
import hashlib
import requests
import json
import re
import base64
import pandas as pd
import numpy as np
import streamlit as st
import io
import PyPDF2
from PIL import Image
from docx import Document as DocxDocument
from pptx import Presentation
from sqlalchemy import create_engine, text, types
from datetime import datetime
from bs4 import BeautifulSoup
import altair as alt
from urllib.parse import urljoin, unquote
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib
import mammoth
from streamlit_adjustable_columns import adjustable_columns
from streamlit_theta import theta_spreadsheet_editor, theta_image_editor
from streamlit_quill import st_quill
from html2docx import html2docx
matplotlib.use('Agg') # Ensure non-interactive backend for Streamlit

# --- PATCH FOR OLD STREAMLIT VERSIONS ---
# Fixes: TypeError: columns() got an unexpected keyword argument 'vertical_alignment'/'border'/'gap'
# Fixes: AttributeError: module 'streamlit' has no attribute 'rerun'
# This happens because streamlit-adjustable-columns assumes a newer Streamlit API
if not hasattr(st, 'rerun'):
    st.rerun = st.experimental_rerun

_original_st_columns = st.columns
def _patched_st_columns(*args, **kwargs):
    # Strip all arguments introduced in newer Streamlit versions
    kwargs.pop('vertical_alignment', None)
    kwargs.pop('gap', None)
    kwargs.pop('border', None)
    return _original_st_columns(*args, **kwargs)
st.columns = _patched_st_columns

# --- CONFIG ---
icon = Image.open("assets/icon_moderation.png") if os.path.exists("assets/icon_moderation.png") else None
st.set_page_config(
    page_title="Sapere Intelligence",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Enterprise CSS (Matte Dark)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* Modern Headers */
    h1 {
        color: #f9a03f !important;
        font-size: 52px !important;
        font-weight: 700 !important;
        margin-bottom: 30px !important;
        letter-spacing: -0.02em;
    }

    h2 {
        color: #f9a03f !important;
        font-size: 28px !important;
        font-weight: 600 !important;
        margin-top: 30px !important;
        margin-bottom: 20px !important;
    }

    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #161b22;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 20px;
        color: #8b949e !important;
        transition: all 0.2s ease;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #f9a03f !important;
        font-weight: 600 !important;
    }

    /* Solid Enterprise Buttons */
    .stButton>button {
        background-color: #21262d;
        color: #f9a03f !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 18px !important;
        width: 100%;
        height: 3.2em !important;
        transition: all 0.1s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #30363d;
        border-color: #f9a03f !important;
    }

    .stButton>button:active {
        background-color: #f9a03f;
        color: #0d1117 !important;
    }

    /* Professional Metrics */
    [data-testid="stMetricValue"] {
        color: #f9a03f !important;
        font-size: 38px !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 15px !important;
        text-transform: uppercase;
        font-weight: 500;
    }

    /* Card Styling */
    .stAlert {
        border-radius: 8px;
        border: 1px solid #30363d;
        background-color: #161b22;
        color: #c9d1d9;
    }

    .stDataFrame {
        border: 1px solid #30363d;
        border-radius: 8px;
    }

    /* Expander Layout */
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #f9a03f !important;
        font-size: 18px !important;
    }

    .block-container {
        padding-top: 3rem !important;
        max-width: 1200px !important;
    }

    /* Input Fields */
    .stTextInput input {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        color: #c9d1d9 !important;
        font-size: 17px !important;
    }

    /* Sandbox & Editor Styles */
    .doc-container {
        background-color: white;
        color: black;
        padding: 40px;
        border-radius: 5px;
        overflow-y: auto;
        max-height: 800px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    .ql-container {
        background-color: white !important;
        color: black !important;
        font-size: 16px !important;
        height: 600px !important;
    }
    .ql-toolbar { background-color: #f0f0f0 !important; }

    .react-resizable-handle {
        background-color: #30363d !important;
        width: 6px !important;
        height: 100vh !important;
        top: 0 !important;
        right: -3px !important;
        cursor: col-resize !important;
        opacity: 0.5;
    }
    .react-resizable-handle:hover { background-color: #f9a03f !important; opacity: 1; }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIG & DB (Hardcoded) ---
DB_URL = DATABASE_URL
AI_API_KEY = OPENROUTER_API_KEY
engine = create_engine(DB_URL)

DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloaded_docs")
if not os.path.exists(DOWNLOAD_DIR): os.makedirs(DOWNLOAD_DIR)

TARGET_URL = "https://cchgeu.ru/education/programms/planned/rupo-9kl-4/?docs2026"
TEXT_MODEL = "google/gemini-2.5-flash"

# --- HELPERS FOR GENERATION TAB ---
def display_pdf(file_bytes):
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_docx(file_bytes):
    result = mammoth.convert_to_html(io.BytesIO(file_bytes))
    st.markdown(f'<div class="doc-container">{result.value}</div>', unsafe_allow_html=True)

# --- PROMPTS ---
METHODOLOGY_PROMPT = """Оцени ЭОМ (электронный образовательный материал) по критериям:
АВТОМАТИЧЕСКИЕ: описание заполнено; указаны темы/дидактические единицы; >=6 фрагментов; экраны заполнено; шрифт>=22пт; заимствование<=20%; текст не выходит за слайд; названия и время фрагментов указаны.
НОРМАТИВНО-ПРАВОВЫЕ: соответствие ФГОС; защита детей (436-ФЗ); нет противоречий Конституции РФ; нет экстремизма; актуальные НПА; корректные карты/границы РФ; соответствие ГК РФ ст.1274,1259; гражданская идентичность.
НОРМАТИВНО-ТЕХНИЧЕСКИЕ: соответствие ТУ/регламентам; актуальные стандарты и СанПиН.
МЕТОДОЛОГИЧЕСКИЕ: последовательность и завершённость; логичность; практико-ориентированность; критерии оценивания; практические задания; современные тенденции; потребности рынка труда; новые технологии; междисциплинарность; экология; критическое мышление; активное обучение; деятельностный подход.
МЕТОДИЧЕСКИЕ: соответствие темам каркаса; корректная терминология; научная достоверность; однозначность формулировок; указание источников; корректные спецсимволы; нормы речи; соответствие названия содержанию; возрастное соответствие; доп.источники; полнота по ФРП; познавательная ценность; уникальность фрагментов.
ТЕХНИЧЕСКИЕ: нет водяных знаков/рекламы; чёткое аудио/видео; единообразие названий; единый стиль; пастельный фон; не перегружены; нет сторонних ссылок; контент по зонам (доска/планшет учителя/планшет ученика).
Верни JSON: {"compliance_rate":0-1,"structure":0-5,"clarity":0-5,"terminology":0-5,"completeness":0-5,"practice_orientation":0-5,"legal_compliance":0-5,"technical_quality":0-5,"subject":"","topic":"","annotation":"","material_type":"","issues":[],"recommendations":[],"usage_allowed": "yes/no"}"""

# --- CORE LOGIC ---
def calculate_thematic_links(df):
    if df.empty or len(df) < 2: return df
    try:
        # Prep data for AI with null-safety
        topics_list = []
        for _, r in df.iterrows():
            t = str(r['topic']) if r['topic'] else "Без темы"
            a = str(r['annotation'])[:200] if r['annotation'] else "Без описания"
            topics_list.append({"id": r['id'], "topic": t, "annotation": a})

        prompt = f"""Ты - эксперт-методист. Выстрой логический порядок изучения следующих тем учебного курса от простого к сложному.
        Верни ТОЛЬКО JSON-список ID в правильном порядке, например: ["id1", "id2", "id3"].
        Вот темы: {json.dumps(topics_list, ensure_ascii=False)}"""

        ordered_ids = call_ai(prompt, is_json=True)

        if isinstance(ordered_ids, list):
            # Create a mapping from ID to its order in the list
            order_map = {str(id_): i for i, id_ in enumerate(ordered_ids)}
            df['embedding_order'] = df['id'].map(order_map).fillna(999)
            df['has_previous'] = df['embedding_order'] > 0
            df['has_next'] = df['embedding_order'] < (len(ordered_ids) - 1)
    except Exception as e:
        st.error(f"Ошибка AI-сортировки: {e}")
    return df

def get_document_links(url):
    try:
        r = requests.get(url, timeout=30)
        soup = BeautifulSoup(r.text, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if any(ext in href.lower() for ext in ['.pdf', '.docx', '.pptx', '.doc', '.png', '.jpg', '.csv', '.xlsx']):
                if not href.lower().endswith('.sig'):
                    links.append({'Название': unquote(os.path.basename(href)), 'url': urljoin(url, href)})
        return links
    except: return []

def download_file(url, dest):
    fname = unquote(os.path.basename(url))
    fpath = os.path.join(dest, fname)
    if os.path.exists(fpath): return fpath, False
    try:
        r = requests.get(url, stream=True, timeout=60)
        with open(fpath, 'wb') as f:
            for c in r.iter_content(8192): f.write(c)
        return fpath, True
    except: return None, False

def extract_content(fpath):
    ext = os.path.splitext(fpath)[1].lower()
    text, imgs = "", 0
    try:
        if ext == '.docx':
            d = DocxDocument(fpath); text = "\n".join([p.text for p in d.paragraphs]); imgs = len(d.inline_shapes)
        elif ext == '.pdf':
            with open(fpath, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                for p in pdf.pages: text += (p.extract_text() or "") + "\n"
        elif ext == '.pptx':
            p = Presentation(fpath)
            for s in p.slides:
                for sh in s.shapes:
                    if hasattr(sh, "text"): text += sh.text + "\n"
                    if sh.shape_type == 13: imgs += 1
        elif ext in ['.csv', '.xlsx']:
            df = pd.read_csv(fpath) if ext == '.csv' else pd.read_excel(fpath); text = df.to_string()
        elif ext in ['.mp3', '.wav', '.mp4', '.m4a']:
            text = process_media_content(fpath)
    except: pass
    return text, imgs

def process_media_content(fpath):
    # Specialized logic for transcribing audio/video via AI
    # from moviepy import VideoFileClip, AudioFileClip
    import tempfile

    ext = os.path.splitext(fpath)[1].lower()
    temp_audio = os.path.join(tempfile.gettempdir(), f"temp_audio_{hashlib.md5(fpath.encode()).hexdigest()}.mp3")

    try:
        if ext == '.mp4':
            video = VideoFileClip(fpath)
            if video.audio:
                video.audio.write_audiofile(temp_audio, codec='libmp3lame', verbose=False, logger=None)
            video.close()
        else:
            audio = AudioFileClip(fpath)
            audio.write_audiofile(temp_audio, codec='libmp3lame', verbose=False, logger=None)
            audio.close()

        if os.path.exists(temp_audio):
            with open(temp_audio, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')

            # Using Gemini to transcribe. Note: Large files might hit API limits.
            # We send a multimodal prompt if the model supports it.
            # For simplicity in this prototype, we'll ask Gemini to act as a transcriber.
            prompt = "Ты - эксперт по транскрибации. Переложи содержание этого аудиофайла в подробный текстовый конспект лекции. Сохрани все ключевые термины и структуру."

            h = {"Authorization": f"Bearer {AI_API_KEY}"}
            payload = {
                "model": TEXT_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url", # OpenRouter often uses image_url for any data block or specific multimodal formats
                                "image_url": {"url": f"data:audio/mp3;base64,{audio_b64}"}
                            }
                        ]
                    }
                ]
            }
            # Note: Many OpenRouter models support this via standard data URLs
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=h, json=payload, timeout=120).json()
            return r['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Ошибка обработки медиа: {e}"
    finally:
        if os.path.exists(temp_audio): os.remove(temp_audio)
    return ""

def call_ai(prompt, is_json=False):
    h = {"Authorization": f"Bearer {AI_API_KEY}"}
    d = {"model": TEXT_MODEL, "messages": [{"role": "user", "content": prompt}]}
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=h, json=d, timeout=60).json()
        c = r['choices'][0]['message']['content'].strip()
        if is_json:
            m = re.search(r'\{.*\}', c, re.DOTALL)
            return json.loads(m.group()) if m else None
        return c
    except: return None

# --- DB INIT ---
def init_db():
    try:
        # 1. Create table if not exists (separate transaction)
        with engine.begin() as conn:
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS opit_files (
                    id TEXT PRIMARY KEY,
                    file_name TEXT,
                    processed_at TIMESTAMP
                )
            '''))

        # 2. Get existing columns
        with engine.connect() as conn:
            existing_cols = conn.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'opit_files'"
            )).fetchall()
            existing_cols = [c[0] for c in existing_cols]

        # 3. Add missing columns (one by one)
        columns = [
            ("file_path", "TEXT"), ("extension", "TEXT"), ("file_size", "BIGINT"),
            ("text_content", "TEXT"), ("text_length", "INTEGER"), ("images_count", "INTEGER"),
            ("image_analysis", "TEXT"), ("subject", "TEXT"), ("topic", "TEXT"),
            ("annotation", "TEXT"), ("compliance_rate", "REAL"), ("compliance_structure", "REAL"),
            ("compliance_clarity", "REAL"), ("compliance_terminology", "REAL"),
            ("compliance_completeness", "REAL"), ("practice_orientation", "REAL"),
            ("legal_compliance", "REAL"), ("technical_quality", "REAL"),
            ("usage_allowed", "TEXT"), ("issues", "TEXT"), ("recommendations", "TEXT"),
            ("embedding_order", "INTEGER"), ("has_previous", "BOOLEAN"), ("has_next", "BOOLEAN")
        ]

        for col_name, col_type in columns:
            if col_name not in existing_cols:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE opit_files ADD COLUMN {col_name} {col_type}"))
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
init_db()

# --- REPOSITORY CONFIG ---
if 'repo_path' not in st.session_state:
    st.session_state['repo_path'] = os.path.join(os.getcwd(), "downloaded_docs")

# --- UI ---
st.title("Sapere Intelligence")
tabs = st.tabs(["Загрузка материалов", "Аналитический цикл", "Репозиторий", "Набор данных", "Генерация и Редактор"])

with tabs[0]:
# (rest of code stays same until the end)

    st.header("Управление хранилищем")

    # Custom Path Selection
    custom_path = st.text_input("Локальный путь к репозиторию", value=st.session_state['repo_path'])
    if custom_path != st.session_state['repo_path']:
        st.session_state['repo_path'] = custom_path
        if not os.path.exists(custom_path):
            try:
                os.makedirs(custom_path)
                st.success(f"Создана новая директория: {custom_path}")
            except Exception as e:
                st.error(f"Ошибка доступа к пути: {e}")

    repo_dir = st.session_state['repo_path']

    # --- FOLDER STATISTICS ---
    if os.path.exists(repo_dir):
        files_data = []
        for f in os.listdir(repo_dir):
            fp = os.path.join(repo_dir, f)
            if os.path.isfile(fp):
                stat = os.stat(fp)
                ext = os.path.splitext(f)[1].lower() or "no_ext"
                files_data.append({
                    "Имя": f,
                    "Расширение": ext,
                    "Размер (МБ)": round(stat.st_size / (1024 * 1024), 3),
                    "Изменен": datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                })

        if files_data:
            df_files = pd.DataFrame(files_data)
            st.subheader("Технический аудит репозитория")

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Всего файлов", len(df_files))
            m2.metric("Общий вес", f"{df_files['Размер (МБ)'].sum():.2f} МБ")
            m3.metric("Типов файлов", df_files['Расширение'].nunique())

            # Extension Breakdown
            ext_stats = df_files.groupby("Расширение").agg(
                Кол_во=("Имя", "count"),
                Общий_размер_МБ=("Размер (МБ)", "sum")
            ).reset_index()

            col_l, col_r = st.columns([1, 1])
            with col_l:
                st.write("**Детализация по типам**")
                st.dataframe(ext_stats, height=300)
            with col_r:
                # Spacer to push chart below the header level of col_l
                st.write("")
                st.write("")
                st.write("")
                # High-End Dashboard Style Chart
                base = alt.Chart(ext_stats).encode(
                    y=alt.Y('Расширение:N', sort='-x', title=None, axis=alt.Axis(labelFontSize=14, labelColor='#8b949e')),
                    x=alt.X('Кол_во:Q', title=None, axis=None)
                )

                bars = base.mark_bar(
                    color='#ff8c00',
                    cornerRadiusEnd=6,
                    height=25
                ).encode(
                    tooltip=['Расширение', 'Кол_во']
                )

                labels = base.mark_text(
                    align='left',
                    baseline='middle',
                    dx=10,
                    color='#f1f5f9',
                    fontSize=15,
                    fontWeight=600
                ).encode(
                    text='Кол_во:Q'
                )

                chart = (bars + labels).properties(
                    height=300,
                    width='container'
                ).configure_view(
                    strokeWidth=0
                ).configure_axis(
                    grid=False
                )

                st.altair_chart(chart, use_container_width=True)

            with st.expander("Посмотреть полный список файлов в папке"):
                st.table(df_files)
        else:
            st.info("Папка пуста")

    # Manual Upload Section
    with st.expander("Загрузить файлы вручную с компьютера"):
        uploaded_files = st.file_uploader("Выберите файлы для добавления в репозиторий", accept_multiple_files=True)
        if uploaded_files:
            if st.button("Сохранить загруженные файлы"):
                saved_count = 0
                for f in uploaded_files:
                    fpath = os.path.join(repo_dir, f.name)
                    with open(fpath, "wb") as buffer:
                        buffer.write(f.getbuffer())
                    saved_count += 1
                st.success(f"Сохранено в репозиторий: {saved_count} файл(ов)")

    st.markdown("---")
    st.header("Сбор учебного контента с сайта")
    target_url = st.text_input("Введите URL для сканирования", value="https://cchgeu.ru/education/programms/planned/rupo-9kl-4/?docs2026")

    if st.button("Начать сканирование сайта"):
        if target_url:
            st.session_state['web_links'] = get_document_links(target_url)
        else:
            st.error("Пожалуйста, введите корректный URL")

    if 'web_links' in st.session_state:
        links = st.session_state['web_links']
        st.dataframe(pd.DataFrame(links), height=300)
        if st.button(f"Скачать в репозиторий ({len(links)} ед.)"):
            prog = st.progress(0); status = st.empty(); count = 0
            for i, l in enumerate(links):
                status.text(f"Процесс: {l['Название']}")
                _, new = download_file(l['url'], repo_dir)
                if new: count += 1
                prog.progress((i+1)/len(links))
            st.success(f"Загрузка завершена. Новых материалов: {count}")

with tabs[1]:
    st.header("Аналитический цикл")
    repo_dir = st.session_state['repo_path']
    all_files = [f for f in os.listdir(repo_dir)] if os.path.exists(repo_dir) else []

    if not all_files:
        st.warning("Хранилище пусто. Загрузите файлы на первой вкладке.")
    else:
        # Section 1: AI Moderation with Interactivity
        st.subheader("1. Интеллектуальная модерация (OpenRouter)")
        st.info("Анализ контента, извлечение тем и формирование справок.")

        # Identify available extensions
        available_exts = sorted(list(set([os.path.splitext(f)[1].lower() for f in all_files if os.path.splitext(f)[1]])))

        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            selected_exts = st.multiselect("Выберите типы файлов для анализа", available_exts, default=available_exts)
        with col_f2:
            limit_files = st.number_input("Лимит файлов (0 - без лимита)", min_value=0, value=0)

        # Filter the file list
        files_to_process = [f for f in all_files if os.path.splitext(f)[1].lower() in selected_exts]
        if limit_files > 0:
            files_to_process = files_to_process[:limit_files]

        st.write(f"Объектов к обработке: **{len(files_to_process)}** (отфильтровано из {len(all_files)})")

        if st.button("Запустить AI-модерацию выбранных файлов"):
            prog = st.progress(0); status = st.empty()
            for i, fname in enumerate(files_to_process):
                fpath = os.path.join(repo_dir, fname); status.text(f"Анализ ({i+1}/{len(files_to_process)}): {fname}")
                txt, imgs = extract_content(fpath)

                # Full AI Analysis with detailed metrics
                prompt = f"{METHODOLOGY_PROMPT}\n\nТЕКСТ ДЛЯ ОЦЕНКИ:\n{txt[:3000]}"
                res = call_ai(prompt, is_json=True)

                row = {
                    "id": hashlib.md5(fname.encode()).hexdigest(),
                    "file_name": fname,
                    "extension": os.path.splitext(fname)[1],
                    "text_content": txt,
                    "file_path": fpath,
                    "processed_at": datetime.now()
                }

                if res:
                    row.update({
                        "subject": res.get('subject', ''),
                        "topic": res.get('topic', ''),
                        "annotation": res.get('annotation', ''),
                        "compliance_rate": float(res.get('compliance_rate', 0)),
                        "compliance_structure": float(res.get('structure', 0)),
                        "compliance_clarity": float(res.get('clarity', 0)),
                        "compliance_terminology": float(res.get('terminology', 0)),
                        "compliance_completeness": float(res.get('completeness', 0)),
                        "practice_orientation": float(res.get('practice_orientation', 0)),
                        "legal_compliance": float(res.get('legal_compliance', 0)),
                        "technical_quality": float(res.get('technical_quality', 0)),
                        "usage_allowed": res.get('usage_allowed', 'no'),
                        "issues": json.dumps(res.get('issues', []), ensure_ascii=False),
                        "recommendations": json.dumps(res.get('recommendations', []), ensure_ascii=False)
                    })

                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM opit_files WHERE id = :id"), {"id": row['id']})
                    pd.DataFrame([row]).to_sql("opit_files", conn, if_exists="append", index=False)
                prog.progress((i+1)/len(files_to_process))
            st.success("Модерация выбранных файлов завершена.")

        st.markdown("---")

        # Section 2: Semantic Linkage
        st.subheader("2. Семантическое связывание (Эмбеддинги)")
        st.info("Расчет тематической близости и выстраивание логической последовательности.")

        if st.button("Обновить логические связи в базе"):
            try:
                df_final = pd.read_sql("SELECT * FROM opit_files", engine)
                if df_final.empty:
                    st.error("База данных пуста. Сначала проведите модерацию.")
                else:
                    with st.spinner("Рассчитываем эмбеддинги..."):
                        df_linked = calculate_thematic_links(df_final)
                        for _, r in df_linked.iterrows():
                            with engine.begin() as conn:
                                conn.execute(text("UPDATE opit_files SET embedding_order=:o, has_previous=:p, has_next=:n WHERE id=:id"),
                                             {"o":int(r['embedding_order']), "p":bool(r['has_previous']), "n":bool(r['has_next']), "id":r['id']})
                        st.success("Семантическая структура репозитория обновлена.")
            except Exception as e:
                st.error(f"Ошибка при связывании: {e}")

with tabs[2]:
    st.header("Репозиторий объектов")
    search = st.text_input("Поиск по базе знаний (названия, контент, аннотации)")
    try:
        df_repo = pd.read_sql("SELECT * FROM opit_files ORDER BY embedding_order", engine)
        if not df_repo.empty:
            if search:
                mask = (df_repo['text_content'].str.contains(search, case=False, na=False) |
                        df_repo['annotation'].str.contains(search, case=False, na=False) |
                        df_repo['file_name'].str.contains(search, case=False, na=False))
                df_repo = df_repo[mask]

            for _, r in df_repo.iterrows():
                with st.expander(f"{r['file_name']} | Предмет: {r['subject'] or 'Не определен'} | Рейтинг: {int(r['compliance_rate']*100)}%"):
                    st.markdown(f"### Тема: {r['topic']}")
                    st.write(f"**Аннотация:** {r['annotation']}")

                    c1, c2 = st.columns(2)
                    with c1:
                        if not r['has_previous'] and r['has_next']:
                            st.warning("Статус: Вводный материал (Начало курса)")
                        elif r['has_previous'] and r['has_next']:
                            st.info("Статус: Промежуточный материал")
                        elif r['has_previous'] and not r['has_next']:
                            st.success("Статус: Завершающий материал (Конец курса)")
                        else:
                            st.write("Логические связи не определены")

                    with c2:
                        st.write(f"**Допуск к использованию:** {'ДА' if r['usage_allowed']=='yes' else 'НЕТ'}")

                    # Formatted Lists
                    col_iss, col_rec = st.columns(2)
                    with col_iss:
                        st.markdown("**Критические замечания:**")
                        try:
                            iss = json.loads(r['issues'])
                            if isinstance(iss, list) and iss:
                                for item in iss: st.write(f"• {item}")
                            else: st.write("Замечаний нет")
                        except: st.write(r['issues'])

                    with col_rec:
                        st.markdown("**Методические рекомендации:**")
                        try:
                            recs = json.loads(r['recommendations'])
                            if isinstance(recs, list) and recs:
                                for item in recs: st.write(f"• {item}")
                            else: st.write("Рекомендаций нет")
                        except: st.write(r['recommendations'])

                    if st.checkbox("Показать извлеченный текст документа", key=f"txt_{r['id']}"):
                        st.text_area("Raw Text", value=r['text_content'][:5000], height=300)
        else:
            st.info("База знаний пуста. Проведите анализ на предыдущей вкладке.")
    except Exception as e:
        st.error(f"Ошибка загрузки репозитория: {e}")

@st.cache_data(show_spinner=False)
def prepare_analysis_df(df):
    if df.empty: return df

    # Feature Engineering
    def get_stats(txt):
        if not txt or not isinstance(txt, str): return 0, 0, 0, 0, 0, 0
        words = re.findall(r'\w+', txt.lower())
        sentences = re.split(r'[.!?]+', txt)
        w_count = len(words)
        s_count = len(sentences) if len(sentences) > 0 else 1
        avg_w_len = np.mean([len(w) for w in words]) if w_count > 0 else 0
        avg_s_len = w_count / s_count if s_count > 0 else 0

        # Simple complexity metrics
        unique_words = len(set(words))
        term_density = unique_words / w_count if w_count > 0 else 0
        complexity = (avg_w_len * 0.5) + (avg_s_len * 0.5)

        return w_count, s_count, avg_w_len, avg_s_len, term_density, complexity

    stats = df['text_content'].apply(get_stats)
    df[['word_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length', 'term_density', 'complexity']] = pd.DataFrame(stats.tolist(), index=df.index)

    # Fill numeric NaNs
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df

@st.cache_data(show_spinner=False)
def train_and_get_shap(df):
    features = ['text_length', 'word_count', 'avg_word_length', 'sentence_count', 'avg_sentence_length', 'term_density', 'complexity']
    X = df[features]
    y = df['compliance_rate']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Importance df
    imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance')

    return model, shap_values, imp_df, X

with tabs[3]:
    st.header("Глубинная аналитика датасета")

    try:
        raw_df = pd.read_sql("SELECT * FROM opit_files", engine)
        if raw_df.empty:
            st.info("Нет данных для анализа. Проведите модерацию во вкладке 'Аналитический цикл'.")
        else:
            df = prepare_analysis_df(raw_df)

            # --- TOP METRICS ---
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Объектов", len(df))
            with m2:
                st.metric("Средний Score", f"{df['compliance_rate'].mean():.2%}")
            with m3:
                st.metric("Допущено", len(df[df['usage_allowed'] == 'yes']))
            with m4:
                st.metric("Средняя сложность", f"{df['complexity'].mean():.1f}")

            st.markdown("---")

            # --- 1. DETAILED DATA TABLE ---
            st.subheader("1. Детализированный реестр базы данных")

            # Dynamic Column Selector
            all_cols = df.columns.tolist()
            default_cols = ['file_name', 'subject', 'compliance_rate', 'word_count', 'complexity', 'usage_allowed']

            with st.expander("Настройка отображения таблицы"):
                selected_display_cols = st.multiselect(
                    "Выберите поля для вывода:",
                    options=all_cols,
                    default=[c for c in default_cols if c in all_cols]
                )

            if selected_display_cols:
                st.dataframe(
                    df[selected_display_cols].sort_values('compliance_rate', ascending=False) if 'compliance_rate' in selected_display_cols else df[selected_display_cols]
                )
            else:
                st.warning("Выберите хотя бы одну колонку для отображения.")

            st.markdown("---")

            # --- 2. STATISTICAL ANALYSIS (2 Columns) ---
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("2. Матрица корреляции")
                num_df = df.select_dtypes(include=[np.number]).drop(columns=['embedding_order'], errors='ignore')

                # High-End Heatmap
                plt.figure(figsize=(10, 8), facecolor='#0d1117')
                sns.set(rc={'axes.facecolor':'#0d1117', 'figure.facecolor':'#0d1117', 'text.color':'#c9d1d9'})
                ax = sns.heatmap(num_df.corr(), annot=True, cmap='YlOrBr', fmt=".2f",
                                 cbar=False, annot_kws={"size": 10, "weight": "bold"})
                plt.xticks(color='#8b949e', rotation=45, ha='right')
                plt.yticks(color='#8b949e')
                st.pyplot(plt.gcf())
                plt.close()

            with col_right:
                st.subheader("3. Дисперсия признаков")
                variances = num_df.var().sort_values(ascending=False).head(10)
                fig_var = px.bar(
                    x=variances.index,
                    y=variances.values,
                    labels={'x': 'Признак', 'y': 'Дисперсия'},
                    template='plotly_dark',
                    color_discrete_sequence=['#f9a03f']
                )
                fig_var.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_var, width="stretch")

            st.markdown("---")

            # --- 3. MACHINE LEARNING (Importance & SHAP) ---
            st.subheader("4. Факторный анализ (Random Forest)")

            if len(df) > 1:
                # Optimized call with caching
                model, shap_values, imp_df, X_features = train_and_get_shap(df)

                col_ml1, col_ml2 = st.columns(2)

                with col_ml1:
                    fig_imp = px.bar(
                        imp_df, x='Importance', y='Feature', orientation='h',
                        title="Вес признаков в оценке качества",
                        template='plotly_dark',
                        color_discrete_sequence=['#ff8c00']
                    )
                    fig_imp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_imp, width="stretch")

                with col_ml2:
                    st.write("**Влияние признаков (SHAP Analysis)**")
                    try:
                        plt.figure(figsize=(10, 6), facecolor='#0d1117')
                        shap.summary_plot(shap_values, X_features, show=False, plot_type="bar", color='#f9a03f')
                        plt.gcf().set_facecolor('#0d1117')
                        ax = plt.gca()
                        ax.set_facecolor('#0d1117')
                        ax.xaxis.label.set_color('#c9d1d9')
                        ax.yaxis.label.set_color('#c9d1d9')
                        ax.tick_params(colors='#8b949e')
                        st.pyplot(plt.gcf())
                        plt.close()
                    except Exception as e:
                        st.error(f"Ошибка SHAP: {e}")
            else:
                st.warning("Недостаточно данных для обучения модели RF (нужно хотя бы 2 объекта).")

            # --- 4. CONCLUSIONS & EXPORT ---
            st.markdown("---")
            st.subheader("5. Итоговые выводы и экспорт")

            if len(df) > 1:
                top_3 = imp_df.iloc[-3:][::-1]
                cols = st.columns(3)
                for i, (_, row) in enumerate(top_3.iterrows()):
                    cols[i].info(f"**{row['Feature']}**\n\nВлияние на рейтинг: {row['Importance']:.1%}")

            c1, c2 = st.columns([1, 4])
            with c1:
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("Скачать Full Dataset (CSV)", csv, "sapere_dataset.csv", "text/csv")
            with c2:
                st.write("Датасет включает все извлеченные лингвистические признаки и оценки AI.")

    except Exception as e:
        st.error(f"Критическая ошибка аналитики: {e}")
        st.exception(e)

with tabs[4]:
    st.header("Интеллектуальная генерация и редактор")

    # --- UNIFIED AI WORKSTATION ---
    gen_cols = adjustable_columns(2, labels=["Панель управления ИИ", "Интеллектуальный редактор"])

    with gen_cols[0]:
        st.subheader("Инструментарий")

        # 1. Context Source (File Uploader)
        uploaded_file = st.file_uploader(
            "Загрузить образец / контекст",
            type=["pdf", "docx", "csv", "xlsx", "png", "jpg", "jpeg"],
            key="gen_doc_uploader"
        )
        if uploaded_file is not None:
            file_id = f"gen_{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("gen_current_file_id") != file_id:
                st.session_state["gen_file_bytes"] = uploaded_file.getvalue()
                st.session_state["gen_file_name"] = uploaded_file.name
                st.session_state["gen_current_file_id"] = file_id
                # Reset editor if it was for a different file, but NOT if it was generated
                if "gen_is_ai_generated" not in st.session_state:
                    for key in ["gen_docx_html", "gen_df_data"]:
                        if key in st.session_state: del st.session_state[key]

        st.markdown("---")

        # 2. AI Expansion Agent (Gap Analysis)
        st.write("**Анализ и дополнение курса**")
        if st.button("Найти методические пробелы"):
            try:
                df_gap = pd.read_sql("SELECT * FROM opit_files ORDER BY embedding_order", engine)
                if df_gap.empty:
                    st.warning("База пуста. Проведите модерацию в Аналитическом цикле.")
                else:
                    with st.spinner("ИИ анализирует структуру..."):
                        topics_info = []
                        for _, r in df_gap.iterrows():
                            t = str(r['topic']) if r['topic'] else "Без темы"
                            a = str(r['annotation'])[:100] if r['annotation'] else "Без описания"
                            topics_info.append({"topic": t, "annotation": a})

                        prompt_gaps = f"""Ты - эксперт-методист. Найди 3 логических разрыва в списке тем курса: {json.dumps(topics_info, ensure_ascii=False)}
                        Верни JSON: {{"gaps": [[{{"after": "A", "before": "B", "subject": "S", "new_topic": "T"}}]]}}"""
                        gap_res = call_ai(prompt_gaps, is_json=True)
                        if gap_res and 'gaps' in gap_res:
                            st.session_state['ai_gaps'] = gap_res['gaps']
            except Exception as e: st.error(f"Ошибка аудита: {e}")

        if 'ai_gaps' in st.session_state:
            selected_gap = st.selectbox("Выберите разрыв:", options=st.session_state['ai_gaps'],
                                        format_func=lambda x: f"{x['new_topic']} (между {x['after']} и {x['before']})")
            if st.button("Заполнить разрыв"):
                with st.spinner("Генерация..."):
                    gen_prompt = f"Сгенерируй подробный HTML-материал (h2, p, ul) по теме '{selected_gap['new_topic']}' для предмета '{selected_gap['subject']}'."
                    html = call_ai(gen_prompt)
                    if html:
                        st.session_state["gen_docx_html"] = html
                        st.session_state["gen_file_name"] = f"AI_{selected_gap['new_topic'].replace(' ', '_')}.docx"
                        st.session_state["gen_is_ai_generated"] = True
                        st.rerun()

        st.markdown("---")

        # 3. Free-form Generation
        st.write("**Свободная генерация**")
        custom_prompt = st.text_area("Ваш запрос к ИИ", placeholder="Например: Напиши план лекции по квантовой физике...")
        if st.button("Генерировать в редактор"):
            if custom_prompt:
                with st.spinner("ИИ пишет..."):
                    gen_html = call_ai(f"Сгенерируй учебный материал в формате HTML (h2, p, ul, table) по запросу: {custom_prompt}")
                    if gen_html:
                        st.session_state["gen_docx_html"] = gen_html
                        st.session_state["gen_file_name"] = "AI_Custom_Material.docx"
                        st.session_state["gen_is_ai_generated"] = True
                        st.rerun()

        st.markdown("---")

        # 4. Save to Database
        if "gen_docx_html" in st.session_state:
            st.write("**Сохранение**")
            if st.button("Модерировать и сохранить в БД"):
                with st.spinner("Анализ и запись..."):
                    clean_text = re.sub('<[^<]+?>', '', st.session_state["gen_docx_html"])
                    mod_res = call_ai(f"{METHODOLOGY_PROMPT}\n\nТЕКСТ:\n{clean_text[:4000]}", is_json=True)
                    if mod_res:
                        row = {
                            "id": hashlib.md5(f"ai_edit_{datetime.now().timestamp()}".encode()).hexdigest(),
                            "file_name": st.session_state.get("gen_file_name", "generated.docx"),
                            "extension": ".docx", "text_content": clean_text, "processed_at": datetime.now(),
                            "subject": mod_res.get('subject', 'Educational'), "topic": mod_res.get('topic', 'AI Generated'),
                            "annotation": mod_res.get('annotation', 'Сгенерировано и отредактировано в системе'),
                            "compliance_rate": float(mod_res.get('compliance_rate', 0)),
                            "usage_allowed": mod_res.get('usage_allowed', 'no'),
                            "issues": json.dumps(mod_res.get('issues', []), ensure_ascii=False),
                            "recommendations": json.dumps(mod_res.get('recommendations', []), ensure_ascii=False)
                        }
                        with engine.begin() as conn:
                            pd.DataFrame([row]).to_sql("opit_files", conn, if_exists="append", index=False)
                        st.success("Объект добавлен в репозиторий!")
                        prepare_analysis_df.clear()
                        train_and_get_shap.clear()

    with gen_cols[1]:
        if "gen_docx_html" in st.session_state:
            st.subheader("Редактор (Quill)")
            content = st_quill(value=st.session_state["gen_docx_html"], html=True, key="gen_quill_editor")
            if content: st.session_state["gen_docx_html"] = content

            st.markdown(" ")
            if st.button("Экспорт в DOCX"):
                buf = html2docx(st.session_state["gen_docx_html"], title="Edited Document")
                st.download_button("Скачать файл", buf.getvalue(), f"edited_{st.session_state.get('gen_file_name', 'document.docx')}",
                                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        elif "gen_file_bytes" in st.session_state and st.session_state["gen_file_bytes"]:
            f_bytes = st.session_state["gen_file_bytes"]
            f_ext = st.session_state["gen_file_name"].split(".")[-1].lower()
            if f_ext == "pdf": display_pdf(f_bytes)
            elif f_ext == "docx":
                if "gen_docx_html" not in st.session_state:
                    st.session_state["gen_docx_html"] = mammoth.convert_to_html(io.BytesIO(f_bytes)).value
                st.rerun()
            elif f_ext in ["csv", "xlsx"]:
                if "gen_df_data" not in st.session_state:
                    st.session_state["gen_df_data"] = pd.read_csv(io.BytesIO(f_bytes)) if f_ext == "csv" else pd.read_excel(io.BytesIO(f_bytes))
                df = st.session_state["gen_df_data"]
                theta_spreadsheet_editor(data=[df.columns.tolist()] + df.values.tolist(), key="gen_theta_sheet")
            elif f_ext in ["png", "jpg", "jpeg"]:
                theta_image_editor(image_file=base64.b64encode(f_bytes).decode(), key="gen_theta_img")
        else:
            st.info("Используйте панель слева для генерации или загрузки документа.")

