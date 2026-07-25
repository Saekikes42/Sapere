from config import DATABASE_URL, OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import io
import PyPDF2
from docx import Document as DocxDocument
import joblib
import os
import json
import requests
import pandas as pd
from sqlalchemy import create_engine
from typing import List, Optional

AI_API_KEY = OPENROUTER_API_KEY
TEXT_MODEL = "google/gemini-2.5-flash"

app = FastAPI(title="Sapere API", description="API для оценки и маршрутизации учебных материалов (Модуль Г)")

engine = create_engine(DATABASE_URL)


models = {}
def load_models():
    model_types = ['label_conformance', 'label_parallel', 'label_sequential', 'label_difficulty']
    for mt in model_types:
        path = f"models/{mt}_latest.joblib"
        if os.path.exists(path):
            models[mt] = joblib.load(path)
        else:
            print(f"Warning: Model {path} not found.")

load_models()

class EvaluateRequest(BaseModel):
    text_content: str

class EvaluateResponse(BaseModel):
    conformance: str
    parallel: str
    sequential: str
    difficulty: str
    is_valid: bool = True
    explanation: str = ""

def process_evaluation_logic(text_content: str) -> EvaluateResponse:
    if not text_content:
        raise HTTPException(status_code=400, detail="Empty text_content")


    res = {}
    for mt in ['label_conformance', 'label_parallel', 'label_sequential', 'label_difficulty']:
        if mt in models:
            pred = models[mt].predict([text_content])[0]
            res[mt] = str(pred)
        else:
            res[mt] = "Неизвестно"


    prompt = f"""Ты — методист-аудитор. Пользователь прислал следующий текст:
{text_content[:2000]}  # берем первые 2000 символов для контекста

Наша локальная ML-система присвоила этому тексту следующие метки:
- Соответствие рекомендациям: {res.get('label_conformance', '')}
- Группа параллельного изучения: {res.get('label_parallel', '')}
- Траектория последовательного изучения: {res.get('label_sequential', '')}
- Сложность: {res.get('label_difficulty', '')}

Твоя задача:
1. Проверить, является ли текст осмысленным учебным или научным материалом (или это бессмысленный набор букв, спам, случайные символы вроде "sdfghjkl").
2. Если это осмысленный текст, напиши краткое методическое обоснование (1-2 предложения), почему наша ML-система могла присвоить ему именно такие метки.

Ответь строго в формате корректного JSON:
{{
  "is_valid": true_или_false,
  "explanation": "Твой комментарий..."
}}"""

    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": TEXT_MODEL, "messages": [{"role": "user", "content": prompt}]}

    is_valid = True
    explanation = "Обоснование недоступно."

    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=15)
        if r.status_code == 200:
            content = r.json()['choices'][0]['message']['content'].strip()
            s = content.find('{')
            e = content.rfind('}')
            if s != -1 and e != -1:
                ai_data = json.loads(content[s:e+1])
                is_valid = ai_data.get("is_valid", True)
                explanation = ai_data.get("explanation", "")
    except Exception as e:
        print(f"AI Audit Error: {e}")

    return EvaluateResponse(
        conformance=res.get('label_conformance', ''),
        parallel=res.get('label_parallel', ''),
        sequential=res.get('label_sequential', ''),
        difficulty=res.get('label_difficulty', ''),
        is_valid=is_valid,
        explanation=explanation
    )

def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    try:
        if ext == '.pdf':
            pdf = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for p in pdf.pages:
                text += (p.extract_text() or "") + "\n"
        elif ext == '.docx':
            d = DocxDocument(io.BytesIO(file_bytes))
            text = "\n".join([p.text for p in d.paragraphs])
        else:
            text = file_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Extraction error: {e}")
    return text

@app.post("/evaluate_file", response_model=EvaluateResponse)
async def evaluate_file(file: UploadFile = File(...)):
    content = await file.read()
    extracted_text = extract_text(content, file.filename)
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file")
    return process_evaluation_logic(extracted_text)

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_material(req: EvaluateRequest):
    return process_evaluation_logic(req.text_content)

class EstimateRequest(BaseModel):
    word_count: int
    images_count: int
    complexity: str
    practice_orientation: float

class EstimateResponse(BaseModel):
    time_min: float

@app.post("/estimate_time", response_model=EstimateResponse)
def estimate_time(req: EstimateRequest):
    BASE_READING_SPEED_WPM = 150
    VISUAL_ANALYSIS_MIN_PER_IMAGE = 3.0
    SETUP_TIME_MIN = 30.0

    difficulty_map = {
        'Низкая': 0.8,
        'Средняя': 1.0,
        'Высокая': 1.2
    }
    diff_multiplier = difficulty_map.get(req.complexity, 1.0)

    read_time = req.word_count / BASE_READING_SPEED_WPM
    visual_time = req.images_count * VISUAL_ANALYSIS_MIN_PER_IMAGE

    base_learning_time = read_time + visual_time
    practice_multiplier = 2.0 if req.practice_orientation > 0.5 else 1.0
    practice_time = base_learning_time * practice_multiplier

    review_time = (base_learning_time + practice_time) * 0.2

    total_time = (SETUP_TIME_MIN + base_learning_time + practice_time + review_time) * diff_multiplier
    return EstimateResponse(time_min=round(total_time, 2))

class TrajectoryMaterial(BaseModel):
    id: str
    subject: str
    topic: str
    est_time_min: float

class TrajectoryRequest(BaseModel):
    hours_per_week: float
    days_per_week: int
    materials: List[TrajectoryMaterial]

@app.post("/build_trajectory")
def build_trajectory(req: TrajectoryRequest):
    if req.days_per_week < 1 or req.days_per_week > 7:
        raise HTTPException(status_code=400, detail="days_per_week must be between 1 and 7")

    daily_budget_min = (req.hours_per_week * 60) / req.days_per_week

    weeks = []
    current_week = 1
    current_day = 1
    current_day_spent = 0
    current_week_days = []
    current_day_plan = []

    for mat in req.materials:
        time_left = mat.est_time_min
        part_num = 1

        while time_left > 0.1:
            available_in_day = daily_budget_min - current_day_spent
            if available_in_day <= 1:
                if current_day_plan:
                    current_week_days.append({
                        "День": current_day,
                        "План_объекты": current_day_plan,
                        "Занято_мин": current_day_spent
                    })
                current_day += 1
                current_day_spent = 0
                current_day_plan = []
                if current_day > req.days_per_week:
                    if current_week_days:
                        weeks.append({"Неделя": current_week, "Дни": current_week_days})
                    current_week += 1
                    current_day = 1
                    current_week_days = []
                continue

            take_time = min(time_left, available_in_day)
            is_partial = (take_time < mat.est_time_min)

            part_label = ""
            if is_partial:
                if time_left == mat.est_time_min:
                    part_label = "(Начало)"
                elif abs(take_time - time_left) < 1:
                    part_label = "(Завершение)"
                else:
                    part_label = f"(Продолжение {part_num})"

            current_day_plan.append({
                "subject": mat.subject,
                "topic": mat.topic,
                "time_min": take_time,
                "part_label": part_label
            })

            current_day_spent += take_time
            time_left -= take_time
            part_num += 1

            if time_left > 0.1:
                if current_day_plan:
                    current_week_days.append({
                        "День": current_day,
                        "План_объекты": current_day_plan,
                        "Занято_мин": current_day_spent
                    })
                current_day += 1
                current_day_spent = 0
                current_day_plan = []
                if current_day > req.days_per_week:
                    if current_week_days:
                        weeks.append({"Неделя": current_week, "Дни": current_week_days})
                    current_week += 1
                    current_day = 1
                    current_week_days = []

    if current_day_plan:
        current_week_days.append({
            "День": current_day,
            "План_объекты": current_day_plan,
            "Занято_мин": current_day_spent
        })
    if current_week_days:
        weeks.append({"Неделя": current_week, "Дни": current_week_days})

    return {"weeks": weeks}

class MaterialItem(BaseModel):
    id: str
    file_name: str
    subject: str
    topic: str
    complexity: str
    word_count: int
    images_count: int

@app.get("/materials", response_model=List[MaterialItem])
def get_materials():
    try:
        df = pd.read_sql("SELECT id, file_name, subject, topic, label_difficulty as complexity, text_length as word_count, images_count FROM opit_files", engine)
        df = df.fillna({
            "subject": "Неизвестно",
            "topic": "Без темы",
            "complexity": "Средняя",
            "word_count": 0,
            "images_count": 0
        })

        items = []
        for _, row in df.iterrows():
            items.append(MaterialItem(
                id=str(row['id']),
                file_name=str(row['file_name']),
                subject=str(row['subject']),
                topic=str(row['topic']),
                complexity=str(row['complexity']),
                word_count=int(row['word_count']),
                images_count=int(row['images_count'])
            ))
        return items
    except Exception as e:
        print(f"Error fetching materials: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/materials/{file_name}/download")
def download_material(file_name: str):
    file_path = os.path.join(os.path.dirname(__file__), "downloaded_docs", file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")
    return FileResponse(path=file_path, filename=file_name, media_type='application/octet-stream')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="127.0.0.1", port=8000, reload=True)
