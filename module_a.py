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

# --- CONFIG ---
st.set_page_config(page_title="ОПИТ Аналитика", layout="wide")

OPIT_ROOT = r"C:\Users\Nurlan\Desktop\ОПИТ"
# DB_URL = "postgresql+pg8000://bd9:fdgdfgtyuer3@192.168.103.73:5432/bd9"
DB_URL = "postgresql+pg8000://postgres:1234@localhost:5432/postgres"


engine = create_engine(DB_URL)

DEFAULT_KEY = "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"
st.sidebar.title("Настройки")
AI_API_KEY = st.sidebar.text_input("OpenRouter API Key", type="password", value=DEFAULT_KEY)

# Дешёвая модель для изображений
VISION_MODEL = "google/gemini-2.0-flash-lite-001"
TEXT_MODEL = "google/gemini-2.0-flash-lite-001"

# --- Сжатые методические рекомендации для API ---
METHODOLOGY_PROMPT = """Оцени ЭОМ (электронный образовательный материал) по критериям:
АВТОМАТИЧЕСКИЕ: описание заполнено; указаны темы/дидактические единицы; >=6 фрагментов; экраны заполнены; шрифт>=22пт; заимствование<=20%; текст не выходит за слайд; названия и время фрагментов указаны.
НОРМАТИВНО-ПРАВОВЫЕ: соответствие ФГОС; защита детей (436-ФЗ); нет противоречий Конституции РФ; нет экстремизма; актуальные НПА; корректные карты/границы РФ; соответствие ГК РФ ст.1274,1259; гражданская идентичность.
НОРМАТИВНО-ТЕХНИЧЕСКИЕ: соответствие ТУ/регламентам; актуальные стандарты и СанПиН.
МЕТОДОЛОГИЧЕСКИЕ: последовательность и завершённость; логичность; практико-ориентированность; критерии оценивания; практические задания; современные тенденции; потребности рынка труда; новые технологии; междисциплинарность; экология; критическое мышление; активное обучение; деятельностный подход.
МЕТОДИЧЕСКИЕ: соответствие темам каркаса; корректная терминология; научная достоверность; однозначность формулировок; указание источников; корректные спецсимволы; нормы речи; соответствие названия содержанию; возрастное соответствие; доп.источники; полнота по ФРП; познавательная ценность; уникальность фрагментов.
ТЕХНИЧЕСКИЕ: нет водяных знаков/рекламы; чёткое аудио/видео; единообразие названий; единый стиль; пастельный фон; не перегружены; нет сторонних ссылок; контент по зонам (доска/планшет учителя/планшет ученика).
Верни JSON: {"compliance_rate":0-1,"structure":0-5,"clarity":0-5,"terminology":0-5,"completeness":0-5,"practice_orientation":0-5,"legal_compliance":0-5,"technical_quality":0-5,"subject":"","topic":"","annotation":"","material_type":"","issues":[],"recommendations":[]}"""


# --- INIT DB ---
def init_db():
    with engine.connect() as conn:
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS opit_files (
                id TEXT PRIMARY KEY,
                file_path TEXT,
                file_name TEXT,
                extension TEXT,
                file_size BIGINT,
                group_name TEXT,
                date_folder TEXT,
                text_content TEXT,
                text_length INTEGER,
                word_count INTEGER,
                sentence_count INTEGER,
                has_images BOOLEAN,
                images_count INTEGER,
                image_analysis TEXT,
                subject TEXT,
                topic TEXT,
                annotation TEXT,
                material_type TEXT,
                compliance_rate REAL,
                compliance_structure REAL,
                compliance_clarity REAL,
                compliance_terminology REAL,
                compliance_completeness REAL,
                practice_orientation REAL,
                legal_compliance REAL,
                technical_quality REAL,
                issues TEXT,
                recommendations TEXT,
                embedding TEXT,
                embedding_order INTEGER,
                has_previous BOOLEAN,
                has_next BOOLEAN,
                processed_at TIMESTAMP
            )
        '''))
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS opit_scan_summary (
                id SERIAL PRIMARY KEY,
                extension TEXT,
                file_count INTEGER,
                total_size BIGINT,
                scanned_at TIMESTAMP
            )
        '''))
        conn.execute(text("ALTER TABLE opit_files ADD COLUMN IF NOT EXISTS embedding TEXT"))
        conn.execute(text("ALTER TABLE opit_files ADD COLUMN IF NOT EXISTS embedding_order INTEGER"))
        conn.execute(text("ALTER TABLE opit_files ADD COLUMN IF NOT EXISTS has_previous BOOLEAN"))
        conn.execute(text("ALTER TABLE opit_files ADD COLUMN IF NOT EXISTS has_next BOOLEAN"))
        conn.commit()

init_db()


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Для расчёта тематических эмбеддингов нужен пакет sentence-transformers."
        ) from exc
    return SentenceTransformer('intfloat/multilingual-e5-small')


def process_embeddings7(df_input):
    if df_input.empty:
        df_empty = df_input.copy()
        df_empty['embedding'] = pd.Series(dtype='object')
        df_empty['embedding_order'] = pd.Series(dtype='int64')
        df_empty['has_previous'] = pd.Series(dtype='bool')
        df_empty['has_next'] = pd.Series(dtype='bool')
        return df_empty

    df_temp = df_input.copy()

    for col in ['topic', 'annotation', 'text_content', 'subject']:
        if col not in df_temp.columns:
            df_temp[col] = ""

    text_parts = (
        df_temp['topic'].fillna('').astype(str).str.strip() + " " +
        df_temp['annotation'].fillna('').astype(str).str.strip() + " " +
        df_temp['text_content'].fillna('').astype(str).str.strip()
    ).str.replace(r'\s+', ' ', regex=True).str.strip()
    texts = ["passage: " + (text if text else "пустой материал") for text in text_parts.tolist()]

    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    group_series = df_temp['subject'].fillna('').astype(str).str.strip()
    if 'group_name' in df_temp.columns:
        fallback_group = df_temp['group_name'].fillna('').astype(str).str.strip()
        group_series = group_series.mask(group_series.eq(''), fallback_group)
    df_temp['_embedding_group'] = group_series.mask(group_series.eq(''), 'Без предмета')

    df_temp['embedding_order'] = 0
    df_temp['has_previous'] = False
    df_temp['has_next'] = False

    for group_name in df_temp['_embedding_group'].unique():
        mask = df_temp['_embedding_group'] == group_name
        group_indices = df_temp.index[mask].to_numpy()
        group_embeddings = embeddings[mask.to_numpy()]
        n = len(group_indices)

        if n == 0:
            continue
        if n == 1:
            df_temp.loc[group_indices, 'embedding_order'] = 0
            df_temp.loc[group_indices, 'has_previous'] = False
            df_temp.loc[group_indices, 'has_next'] = False
            continue

        norms = np.linalg.norm(group_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        normalized = group_embeddings / norms
        distances = 1 - np.clip(normalized @ normalized.T, -1.0, 1.0)

        visited = np.zeros(n, dtype=bool)
        path = [0]
        visited[0] = True

        for _ in range(1, n):
            last = path[-1]
            candidates = np.where(~visited)[0]
            nearest = candidates[np.argmin(distances[last, candidates])]
            path.append(int(nearest))
            visited[nearest] = True

        order = np.argsort(path)
        df_temp.loc[group_indices, 'embedding_order'] = order.astype(int)
        df_temp.loc[group_indices, 'has_previous'] = order > 0
        df_temp.loc[group_indices, 'has_next'] = order < (n - 1)

    df_temp['embedding'] = [emb.tolist() for emb in embeddings]
    df_temp = df_temp.drop(columns=['_embedding_group'], errors='ignore')
    return df_temp


def refresh_thematic_features_in_db():
    df_db = pd.read_sql(
        """
        SELECT id, group_name, subject, topic, annotation, text_content
        FROM opit_files
        """,
        engine
    )

    if df_db.empty:
        return 0

    df_features = process_embeddings7(df_db)
    payload = []
    for _, row in df_features.iterrows():
        payload.append({
            "id": row["id"],
            "embedding": json.dumps(row["embedding"], ensure_ascii=False),
            "embedding_order": int(row["embedding_order"]),
            "has_previous": bool(row["has_previous"]),
            "has_next": bool(row["has_next"])
        })

    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE opit_files
                SET embedding = :embedding,
                    embedding_order = :embedding_order,
                    has_previous = :has_previous,
                    has_next = :has_next
                WHERE id = :id
            """),
            payload
        )

    return len(payload)


# --- LLM FUNCTIONS ---
def call_llm(prompt, model=None, expect_json=False):
    if not AI_API_KEY:
        return None
    model = model or TEXT_MODEL
    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                          headers=headers, json=data, timeout=60)
        if r.status_code != 200:
            return None
        content = r.json()['choices'][0]['message']['content'].strip()
        if expect_json:
            s, e = content.find('{'), content.rfind('}')
            if s != -1 and e != -1:
                return json.loads(content[s:e+1])
            return None
        return content
    except:
        return None


def analyze_image_via_api(image_bytes, prompt="Опиши содержимое изображения подробно."):
    if not AI_API_KEY:
        return "API ключ не указан"
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ]}]
    }
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content']
        return f"Ошибка {r.status_code}"
    except Exception as e:
        return str(e)


# --- FILE PARSING ---
def scan_opit_folder(root_path):
    """Рекурсивно сканирует папку ОПИТ и возвращает список файлов с метаданными."""
    files = []
    for dp, dn, fns in os.walk(root_path):
        for f in fns:
            fp = os.path.join(dp, f)
            rel = os.path.relpath(fp, root_path)
            parts = rel.split(os.sep)
            ext = os.path.splitext(f)[1].lower()
            try:
                sz = os.path.getsize(fp)
            except:
                sz = 0
            group_name = parts[2] if len(parts) > 2 else ""
            date_folder = parts[3] if len(parts) > 3 else ""
            files.append({
                'path': fp, 'rel_path': rel, 'name': f, 'ext': ext,
                'size': sz, 'group': group_name, 'date_folder': date_folder
            })
    return files


def get_extension_stats(files_list):
    """Статистика по расширениям."""
    from collections import Counter, defaultdict
    counts = Counter()
    sizes = defaultdict(int)
    for f in files_list:
        counts[f['ext']] += 1
        sizes[f['ext']] += f['size']
    rows = [{'Расширение': ext, 'Кол-во': counts[ext],
             'Размер (байт)': sizes[ext],
             'Размер (МБ)': round(sizes[ext] / 1048576, 2)}
            for ext in sorted(counts, key=lambda x: -counts[x])]
    return pd.DataFrame(rows)


def extract_text_docx(filepath):
    try:
        doc = DocxDocument(filepath)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        imgs = sum(1 for r in doc.inline_shapes) if hasattr(doc, 'inline_shapes') else 0
        return text, imgs
    except:
        return "", 0


def extract_text_pdf(filepath):
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text.strip(), 0  # PyPDF2 не извлекает изображения
    except:
        return "", 0


def extract_text_pptx(filepath):
    try:
        prs = Presentation(filepath)
        texts = []
        imgs = 0
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    texts.append(shape.text_frame.text)
                if shape.shape_type == 13:  # Picture
                    imgs += 1
        return "\n".join(texts), imgs
    except:
        return "", 0


def extract_text_doc(filepath):
    """Для .doc — пробуем через antiword или просто читаем бинарно текст."""
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        # Простая эвристика для .doc
        text = raw.decode('cp1251', errors='ignore')
        # Убираем непечатаемые
        text = re.sub(r'[^\x20-\x7E\u0400-\u04FF\n\r\t]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:5000], 0
    except:
        return "", 0


def extract_from_file(finfo):
    """Извлекает текст и кол-во изображений из файла."""
    ext = finfo['ext']
    path = finfo['path']
    if ext == '.docx':
        return extract_text_docx(path)
    elif ext == '.pdf':
        return extract_text_pdf(path)
    elif ext in ('.pptx',):
        return extract_text_pptx(path)
    elif ext == '.ppt':
        return extract_text_pptx(path)  # pptx lib иногда читает .ppt
    elif ext == '.doc':
        return extract_text_doc(path)
    elif ext == '.png':
        return "", 1
    else:
        return "", 0


def extract_images_from_pdf(filepath, max_images=3):
    """PyPDF2 не поддерживает извлечение изображений напрямую."""
    return []


def extract_images_from_pptx(filepath, max_images=3):
    images = []
    try:
        prs = Presentation(filepath)
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.shape_type == 13:
                    images.append(shape.image.blob)
                    if len(images) >= max_images:
                        return images
    except:
        pass
    return images


def analyze_file_images(finfo):
    """Анализирует изображения в файле через дешёвую vision модель."""
    ext = finfo['ext']
    path = finfo['path']
    results = []

    if ext == '.png':
        try:
            with open(path, 'rb') as f:
                img_bytes = f.read()
            r = analyze_image_via_api(img_bytes)
            results.append(r)
        except:
            pass
    elif ext == '.pdf':
        for img_bytes in extract_images_from_pdf(path, 2):
            r = analyze_image_via_api(img_bytes)
            results.append(r)
    elif ext in ('.pptx', '.ppt'):
        for img_bytes in extract_images_from_pptx(path, 2):
            r = analyze_image_via_api(img_bytes)
            results.append(r)

    return " | ".join(results) if results else ""


# --- SAVE TO DB ---
def save_file_to_db(finfo, text_content, images_count, image_analysis, llm_result):
    fid = hashlib.md5(finfo['rel_path'].encode('utf-8')).hexdigest()
    words = re.findall(r'\b\w+\b', text_content.lower()) if text_content else []
    sentences = [s for s in re.split(r'[.!?]+', text_content) if s.strip()] if text_content else []

    row = {
        "id": fid,
        "file_path": finfo['rel_path'],
        "file_name": finfo['name'],
        "extension": finfo['ext'],
        "file_size": finfo['size'],
        "group_name": finfo['group'],
        "date_folder": finfo['date_folder'],
        "text_content": (text_content or "")[:50000],
        "text_length": len(text_content) if text_content else 0,
        "word_count": len(words),
        "sentence_count": len(sentences),
        "has_images": images_count > 0,
        "images_count": images_count,
        "image_analysis": (image_analysis or "")[:5000],
        "subject": str(llm_result.get('subject', '')) if llm_result else '',
        "topic": str(llm_result.get('topic', '')) if llm_result else '',
        "annotation": str(llm_result.get('annotation', '')) if llm_result else '',
        "material_type": str(llm_result.get('material_type', '')) if llm_result else '',
        "compliance_rate": float(llm_result.get('compliance_rate', 0)) if llm_result else 0,
        "compliance_structure": float(llm_result.get('structure', 0)) if llm_result else 0,
        "compliance_clarity": float(llm_result.get('clarity', 0)) if llm_result else 0,
        "compliance_terminology": float(llm_result.get('terminology', 0)) if llm_result else 0,
        "compliance_completeness": float(llm_result.get('completeness', 0)) if llm_result else 0,
        "practice_orientation": float(llm_result.get('practice_orientation', 0)) if llm_result else 0,
        "legal_compliance": float(llm_result.get('legal_compliance', 0)) if llm_result else 0,
        "technical_quality": float(llm_result.get('technical_quality', 0)) if llm_result else 0,
        "issues": json.dumps(llm_result.get('issues', []), ensure_ascii=False) if llm_result else '[]',
        "recommendations": json.dumps(llm_result.get('recommendations', []), ensure_ascii=False) if llm_result else '[]',
        "embedding": None,
        "embedding_order": None,
        "has_previous": False,
        "has_next": False,
        "processed_at": datetime.now()
    }

    df = pd.DataFrame([row])
    
    # Явно указываем типы для SQLAlchemy, чтобы избежать ошибок приведения типов в Postgres (особенно для NULL значений)
    dtype_map = {
        "id": types.String,
        "file_path": types.Text,
        "file_name": types.Text,
        "extension": types.String(50),
        "file_size": types.BigInteger,
        "group_name": types.Text,
        "date_folder": types.Text,
        "text_content": types.Text,
        "text_length": types.Integer,
        "word_count": types.Integer,
        "sentence_count": types.Integer,
        "has_images": types.Boolean,
        "images_count": types.Integer,
        "image_analysis": types.Text,
        "subject": types.Text,
        "topic": types.Text,
        "annotation": types.Text,
        "material_type": types.Text,
        "compliance_rate": types.Float,
        "compliance_structure": types.Float,
        "compliance_clarity": types.Float,
        "compliance_terminology": types.Float,
        "compliance_completeness": types.Float,
        "practice_orientation": types.Float,
        "legal_compliance": types.Float,
        "technical_quality": types.Float,
        "issues": types.Text,
        "recommendations": types.Text,
        "embedding": types.Text,
        "embedding_order": types.Integer,
        "has_previous": types.Boolean,
        "has_next": types.Boolean,
        "processed_at": types.DateTime
    }

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM opit_files WHERE id = :fid"), {"fid": fid})
        df.to_sql("opit_files", conn, if_exists="append", index=False, dtype=dtype_map)
    return row


def save_scan_summary(ext_df):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM opit_scan_summary"))
        for _, row in ext_df.iterrows():
            conn.execute(text(
                "INSERT INTO opit_scan_summary (extension, file_count, total_size, scanned_at) "
                "VALUES (:ext, :cnt, :sz, :ts)"
            ), {"ext": row['Расширение'], "cnt": int(row['Кол-во']),
                "sz": int(row['Размер (байт)']), "ts": datetime.now()})


# --- STREAMLIT UI ---
st.title("📚 ОПИТ — Анализ образовательных материалов")

tab1, tab2, tab3 = st.tabs(["📊 Сканирование", "🔍 Обработка и анализ", "📈 Данные в БД"])

# --- TAB 1: SCAN ---
with tab1:
    st.header("Сканирование папки ОПИТ")
    st.write(f"**Путь:** `{OPIT_ROOT}`")

    if st.button("🔄 Сканировать папку", key="scan_btn"):
        with st.spinner("Сканирование..."):
            all_files = scan_opit_folder(OPIT_ROOT)
            st.session_state['all_files'] = all_files
            ext_df = get_extension_stats(all_files)
            st.session_state['ext_df'] = ext_df
            save_scan_summary(ext_df)

    if 'all_files' in st.session_state:
        all_files = st.session_state['all_files']
        ext_df = st.session_state['ext_df']

        st.subheader(f"Всего файлов: {len(all_files)}")
        st.dataframe(ext_df, width="stretch")

        total_mb = sum(f['size'] for f in all_files) / 1048576
        c1, c2, c3 = st.columns(3)
        c1.metric("Файлов", len(all_files))
        c2.metric("Расширений", len(ext_df))
        c3.metric("Общий размер", f"{total_mb:.1f} МБ")

        # Группы
        groups = set(f['group'] for f in all_files if f['group'])
        st.subheader(f"Группы ({len(groups)})")
        st.write(", ".join(sorted(groups)))

        # Файлы по группам
        with st.expander("Файлы по группам"):
            for g in sorted(groups):
                gf = [f for f in all_files if f['group'] == g]
                st.write(f"**{g}**: {len(gf)} файлов")

# --- TAB 2: PROCESS ---
with tab2:
    st.header("Обработка файлов и AI-анализ")

    if 'all_files' not in st.session_state:
        st.warning("Сначала выполните сканирование на вкладке «Сканирование».")
    else:
        all_files = st.session_state['all_files']
        parseable = [f for f in all_files if f['ext'] in ('.docx', '.pdf', '.pptx', '.ppt', '.doc', '.png')]

        st.write(f"Файлов для обработки: **{len(parseable)}**")

        # Фильтры
        col1, col2 = st.columns(2)
        with col1:
            sel_ext = st.multiselect("Расширения", ['.docx', '.pdf', '.pptx', '.ppt', '.doc', '.png'],
                                     default=['.docx', '.pdf', '.pptx', '.ppt', '.doc', '.png'])
        with col2:
            max_files = st.number_input("Макс. файлов", 1, len(parseable), min(50, len(parseable)))

        filtered = [f for f in parseable if f['ext'] in sel_ext][:max_files]

        analyze_images = st.checkbox("Анализировать изображения через Vision API (дешёвая модель)", value=False)
        run_llm = st.checkbox("Оценивать по методическим рекомендациям через LLM", value=True)
        force_reprocess = st.checkbox("Перезаписывать дубликаты (повторная обработка уже загруженных)", value=False)

        # --- Проверка дубликатов ---
        existing_ids = set()
        try:
            existing_df = pd.read_sql("SELECT id FROM opit_files", engine)
            existing_ids = set(existing_df['id'].tolist())
        except:
            pass

        if not force_reprocess:
            new_files = []
            skipped = 0
            for f in filtered:
                fid = hashlib.md5(f['rel_path'].encode('utf-8')).hexdigest()
                if fid in existing_ids:
                    skipped += 1
                else:
                    new_files.append(f)
            if skipped > 0:
                st.info(f"⏭️ Пропущено дубликатов (уже в БД): **{skipped}**. Новых для обработки: **{len(new_files)}**")
            filtered = new_files
        else:
            st.warning("⚠️ Режим перезаписи: все файлы будут обработаны заново.")

        st.write(f"К обработке: **{len(filtered)}** файлов")

        if st.button("🚀 Запустить обработку", key="process_btn"):
            results = []
            status = st.empty()
            if len(filtered) == 0:
                st.success("Нет новых файлов для обработки. Все уже в БД!")
            else:
                progress = st.progress(0)
                status = st.empty()

                for i, finfo in enumerate(filtered):
                    status.text(f"[{i+1}/{len(filtered)}] {finfo['name']}")

                    # Извлечение текста
                    txt, img_count = extract_from_file(finfo)

                    # Анализ изображений
                    img_analysis = ""
                    if analyze_images and img_count > 0:
                        img_analysis = analyze_file_images(finfo)

                    # LLM оценка
                    llm_result = None
                    if run_llm and txt and len(txt) > 50:
                        prompt = METHODOLOGY_PROMPT + f"\n\nМатериал (первые 3000 символов):\n{txt[:3000]}"
                        llm_result = call_llm(prompt, expect_json=True)

                    # Сохранение
                    try:
                        row = save_file_to_db(finfo, txt, img_count, img_analysis, llm_result)
                        results.append(row)
                    except Exception as e:
                        st.error(f"Ошибка сохранения {finfo['name']}: {e}")

                    progress.progress((i + 1) / len(filtered))

                if results:
                    try:
                        status.text("Пересчитываем тематические эмбеддинги и связи между материалами...")
                        updated_rows = refresh_thematic_features_in_db()
                        st.info(f"Тематические признаки обновлены для {updated_rows} записей.")
                    except Exception as e:
                        st.warning(f"Не удалось пересчитать тематические признаки: {e}")

            st.success(f"Обработано {len(results)} файлов!")
            status.text("Готово!")

# --- TAB 3: DB VIEW ---
with tab3:
    st.header("Данные в PostgreSQL")
    st.write(f"**БД:** `bd9` @ `192.168.103.73:5432`")

    if st.button("🔄 Загрузить данные из БД", key="load_db"):
        try:
            df = pd.read_sql(
                "SELECT * FROM opit_files ORDER BY subject NULLS LAST, embedding_order NULLS LAST, processed_at DESC",
                engine
            )
            st.session_state['db_data'] = df
        except Exception as e:
            st.error(f"Ошибка: {e}")

    if st.button("Пересчитать тематический порядок", key="refresh_embeddings"):
        try:
            updated_rows = refresh_thematic_features_in_db()
            st.success(f"Тематические признаки пересчитаны для {updated_rows} записей.")
            st.session_state['db_data'] = pd.read_sql(
                "SELECT * FROM opit_files ORDER BY subject NULLS LAST, embedding_order NULLS LAST, processed_at DESC",
                engine
            )
        except Exception as e:
            st.error(f"Ошибка пересчёта: {e}")

    if 'db_data' in st.session_state:
        df = st.session_state['db_data']
        st.subheader(f"Записей: {len(df)}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Всего записей", len(df))
        c2.metric("Расширений", df['extension'].nunique())
        c3.metric("Групп", df['group_name'].nunique())
        c4.metric("Ср. compliance", f"{df['compliance_rate'].mean():.2f}" if len(df) > 0 else "—")

        # По расширениям
        st.subheader("По расширениям")
        ext_stats = df.groupby('extension').agg(
            count=('id', 'count'),
            avg_compliance=('compliance_rate', 'mean'),
            total_size=('file_size', 'sum')
        ).reset_index()
        st.dataframe(ext_stats, width="stretch")

        # Сводка
        try:
            scan_df = pd.read_sql("SELECT * FROM opit_scan_summary ORDER BY extension", engine)
            if not scan_df.empty:
                st.subheader("Сводка сканирования")
                st.dataframe(scan_df, width="stretch")
        except:
            pass

        # Таблица файлов
        show_cols = ['file_name', 'extension', 'group_name', 'file_size',
                     'word_count', 'images_count', 'compliance_rate',
                     'subject', 'topic', 'material_type',
                     'embedding_order', 'has_previous', 'has_next']
        available = [c for c in show_cols if c in df.columns]
        st.subheader("Таблица файлов")
        st.dataframe(df[available], width="stretch")

        with st.expander("Полная таблица"):
            st.dataframe(df.drop(columns=['text_content', 'embedding'], errors='ignore'), width="stretch")



