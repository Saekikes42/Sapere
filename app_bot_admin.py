from config import DATABASE_URL, OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN
import os
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import requests
import asyncio
from PIL import Image


DB_URL = DATABASE_URL
engine = create_engine(DB_URL)
TOKEN = TELEGRAM_BOT_TOKEN


icon_path = "assets/icon_bot_admin.png"
icon = Image.open(icon_path) if os.path.exists(icon_path) else None

st.set_page_config(
    page_title="Sapere Intelligence - Управление Ботом",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    h1, h2, h3 { color: #f9a03f !important; font-weight: 700 !important; }
    h1 { font-size: 42px !important; margin-bottom: 30px !important; }

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
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value { font-size: 32px; font-weight: bold; color: #58a6ff; }
    .metric-label { font-size: 13px; color: #8b949e; text-transform: uppercase; }

    .stTextArea textarea {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        border-radius: 6px;
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
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #f9a03f !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Центр Управления Telegram-ботом")
st.caption("Аналитика аудитории и рассылки уведомлений в реальном времени")


@st.cache_data(ttl=10)
def load_users():
    try:
        return pd.read_sql("SELECT * FROM bot_users ORDER BY last_active DESC", engine)
    except Exception as e:
        return pd.DataFrame()

df_users = load_users()

if df_users.empty:
    st.warning("В базе данных пока нет пользователей бота (или таблица еще не создана). Запустите бота и напишите ему /start.")
    st.stop()


total_users = len(df_users)
now = datetime.now()
active_today = len(df_users[df_users['last_active'] >= (now - timedelta(days=1))])
active_week = len(df_users[df_users['last_active'] >= (now - timedelta(days=7))])

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_users}</div><div class='metric-label'>Всего пользователей</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{active_today}</div><div class='metric-label'>Активны за 24 часа</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{active_week}</div><div class='metric-label'>Активны за неделю</div></div>", unsafe_allow_html=True)


tabs = st.tabs(["Аудитория и Логи", "Центр рассылок", "Эмулятор Бота"])

with tabs[0]:
    st.subheader("Реестр пользователей")


    df_disp = df_users.copy()
    df_disp['joined_at'] = df_disp['joined_at'].dt.strftime('%Y-%m-%d %H:%M')
    df_disp['last_active'] = df_disp['last_active'].dt.strftime('%Y-%m-%d %H:%M')

    st.dataframe(
        df_disp[['user_id', 'username', 'first_name', 'joined_at', 'last_active']],
        width="stretch",
        hide_index=True
    )

    if st.button("Обновить данные", key="refresh_db"):
        st.cache_data.clear()
        st.rerun()

with tabs[1]:
    st.subheader("Массовая рассылка")
    st.info("Это сообщение будет отправлено всем пользователям из базы данных бота.")

    message_text = st.text_area("Текст рассылки", height=200, placeholder="Введите текст сообщения. Поддерживается Markdown (например, *жирный*, _курсив_).")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Отправить всем 🚀", key="btn_send_mass"):
            if not message_text.strip():
                st.error("Текст сообщения пуст!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                success_count = 0
                error_count = 0

                user_ids = df_users['user_id'].tolist()
                total = len(user_ids)

                for i, uid in enumerate(user_ids):
                    status_text.text(f"Отправка пользователю {uid} ({i+1}/{total})")

                    try:
                        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
                        payload = {
                            "chat_id": uid,
                            "text": message_text,
                            "parse_mode": "Markdown"
                        }
                        r = requests.post(url, json=payload, timeout=5)
                        if r.status_code == 200:
                            success_count += 1
                        else:
                            error_count += 1
                    except:
                        error_count += 1

                    progress_bar.progress((i + 1) / total)

                status_text.empty()
                st.success(f"Рассылка завершена! Успешно: {success_count}, Ошибок: {error_count}")

with tabs[2]:
    st.subheader("Интерактивный Эмулятор")
    st.info("Здесь вы можете протестировать логику бота без необходимости открывать Telegram. Эмулятор обращается к тем же API эндпоинтам.")

    if "emulator_chat" not in st.session_state:
        st.session_state["emulator_chat"] = [
            {"role": "assistant", "content": "Добро пожаловать в Sapere Bot! Выберите действие:"}
        ]
    if "em_cart" not in st.session_state:
        st.session_state["em_cart"] = []


    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state["emulator_chat"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


    st.write("**Доступные действия (Inline-кнопки):**")
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        if st.button("Оценить материал", key="em_eval"):
            st.session_state["emulator_chat"].append({"role": "user", "content": "[Нажата кнопка: Оценить материал]"})
            st.session_state["emulator_chat"].append({"role": "assistant", "content": "Пришлите текст материала для оценки:"})
            st.rerun()
    with col_a2:
        if st.button("Каталог материалов", key="em_cat"):
            st.session_state["emulator_chat"].append({"role": "user", "content": "[Нажата кнопка: Каталог материалов]"})
            try:
                r = requests.get("http://127.0.0.1:8000/materials", timeout=5)
                materials = r.json()
                if materials:
                    m = materials[0]
                    res = f"📚 **Каталог материалов**\n\n**Файл:** {m['file_name']}\n**Предмет:** {m['subject']}\n**Тема:** {m['topic']}\n\n*(В эмуляторе мы автоматически добавили этот первый материал в вашу корзину для тестов)*"
                    st.session_state["em_cart"].append(m)
                else:
                    res = "Каталог пуст. Создайте материалы через модуль Разметки."
            except Exception as e:
                res = f"Ошибка API: {e}"
            st.session_state["emulator_chat"].append({"role": "assistant", "content": res})
            st.rerun()
    with col_a3:
        if st.button("Моя корзина", key="em_cart_btn"):
            cart = st.session_state["em_cart"]
            if not cart:
                res = "Ваша корзина пуста. Добавьте материалы из каталога!"
            else:
                res = "🎒 **Ваши выбранные материалы:**\n\n"
                for i, m in enumerate(cart):
                    res += f"{i+1}. {m['subject']}: {m['topic']} ({m['complexity']})\n"
            st.session_state["emulator_chat"].append({"role": "user", "content": "[Нажата кнопка: Моя корзина]"})
            st.session_state["emulator_chat"].append({"role": "assistant", "content": res})
            st.rerun()
    with col_a4:
        if st.button("Построить траекторию", key="em_traj"):
            st.session_state["emulator_chat"].append({"role": "user", "content": "[Нажата кнопка: Построить траекторию]"})
            if not st.session_state["em_cart"]:
                st.session_state["emulator_chat"].append({"role": "assistant", "content": "Корзина пуста! Сначала добавьте материалы из каталога."})
            else:
                st.session_state["emulator_chat"].append({"role": "assistant", "content": "Сколько часов в неделю вы готовы учиться? (Напишите число, например 10)"})
            st.rerun()


    user_input = st.chat_input("Написать боту...")
    if user_input:
        st.session_state["emulator_chat"].append({"role": "user", "content": user_input})


        last_bot_msg = ""
        for m in reversed(st.session_state["emulator_chat"][:-1]):
            if m["role"] == "assistant":
                last_bot_msg = m["content"]
                break

        if "Пришлите текст" in last_bot_msg:

            try:
                r = requests.post("http://127.0.0.1:8000/evaluate", json={"text_content": user_input}, timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    if not data.get("is_valid", True):
                        res = "⚠️ Текст не похож на учебный материал."
                    else:
                        res = f"📊 *Оценка материала:*\nСоответствие: {data.get('conformance')}\nСложность: {data.get('difficulty')}\nПараллельно: {data.get('parallel')}\nПоследовательно: {data.get('sequential')}\n\n🤖 *Аудит ИИ:*\n{data.get('explanation', '')}"

                        payload = {"word_count": len(user_input.split()), "images_count": 0, "complexity": data.get('difficulty', 'Средняя'), "practice_orientation": 0.5}
                        r2 = requests.post("http://127.0.0.1:8000/estimate_time", json=payload)
                        if r2.status_code == 200:
                            res += f"\n\n⏱ Оценочное время изучения: {round(r2.json()['time_min']/60, 1)} ч."
                else:
                    res = "Ошибка сервера API (возможно бэкенд не запущен)."
            except Exception as e:
                res = f"Не удалось подключиться к API: {e}"
            st.session_state["emulator_chat"].append({"role": "assistant", "content": res})
        elif "Сколько часов" in last_bot_msg:
            try:
                hours = float(user_input)
                st.session_state["emulator_state"] = {"hours": hours}
                st.session_state["emulator_chat"].append({"role": "assistant", "content": "Сколько дней в неделю вы планируете учиться? (от 1 до 7)"})
            except:
                st.session_state["emulator_chat"].append({"role": "assistant", "content": "Пожалуйста, введите число."})
        elif "Сколько дней" in last_bot_msg:
            try:
                days = int(user_input)
                hours = st.session_state.get("emulator_state", {}).get("hours", 10)
                cart = st.session_state["em_cart"]


                mat_payloads = []
                for m in cart:
                    est_payload = {"word_count": m['word_count'], "images_count": m['images_count'], "complexity": m['complexity'], "practice_orientation": 0.5}
                    r = requests.post("http://127.0.0.1:8000/estimate_time", json=est_payload)
                    est_time = r.json()['time_min'] if r.status_code == 200 else 60
                    mat_payloads.append({"id": m['id'], "subject": m['subject'], "topic": m['topic'], "est_time_min": est_time})

                payload = {"hours_per_week": hours, "days_per_week": days, "materials": mat_payloads}
                r = requests.post("http://127.0.0.1:8000/build_trajectory", json=payload)
                if r.status_code == 200:
                    data = r.json()
                    res = "📌 *Ваша персональная траектория:*\n\n"
                    for w in data.get('weeks', []):
                        res += f"*Неделя {w['Неделя']}*\n"
                        for d in w['Дни']:
                            res += f"  🔹 День {d['День']}:\n"
                            for item in d['План_объекты']:
                                res += f"    - {item['subject']}: {item['topic']} {item.get('part_label', '')} ({round(item['time_min']/60, 1)} ч)\n"
                else:
                    res = "Ошибка при генерации расписания."

                st.session_state["emulator_chat"].append({"role": "assistant", "content": res})
                st.session_state["em_cart"] = []
            except ValueError:
                st.session_state["emulator_chat"].append({"role": "assistant", "content": "Пожалуйста, введите число от 1 до 7."})
            except Exception as e:
                st.session_state["emulator_chat"].append({"role": "assistant", "content": f"Ошибка API: {e}"})
        else:
            st.session_state["emulator_chat"].append({"role": "assistant", "content": "Я эмулятор. Нажимайте кнопки меню выше или следуйте инструкциям на экране."})

        st.rerun()
