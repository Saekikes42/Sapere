import streamlit as st
import psutil
import subprocess
import os
import sys
import time
from PIL import Image

logo_icon = Image.open("assets/sapere_logo.png") if os.path.exists("assets/sapere_logo.png") else None
st.set_page_config(page_title="Система Управления Агентами", page_icon=logo_icon, layout="wide")

# Применяем CSS стиль в точности как в остальных файлах
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Inter', -apple-system, sans-serif;
}
h1, h2, h3, h4, h5 { color: #f9a03f !important; font-weight: 700 !important; }

.stButton>button {
    background-color: #21262d; color: #f9a03f !important;
    border: 1px solid #30363d !important; border-radius: 6px !important;
    font-weight: 600 !important; width: 100%; transition: all 0.2s ease;
}
.stButton>button:hover { background-color: #30363d; border-color: #f9a03f !important; }

/* Стилизация контейнеров-карточек под карточки из проекта */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


st.title("Система Управления Агентами")
st.markdown("Централизованный хаб для развертывания и остановки системных агентов.")

SERVICES = {
    "app_moderation": {
        "name": "Агент: Ядро Модерации Контента",
        "icon_path": "assets/icon_moderation.png",
        "command": [sys.executable, "-m", "streamlit", "run", "app_moderation.py", "--server.port", "8502", "--server.headless", "true"],
        "script": "app_moderation.py",
        "url": "http://localhost:8502",
        "type": "Streamlit"
    },
    "app_analytics": {
        "name": "Агент: Аналитический Дашборд",
        "icon_path": "assets/icon_analytics.png",
        "command": [sys.executable, "app_analytics.py"],
        "script": "app_analytics.py",
        "url": "http://localhost:8050",
        "type": "Dash"
    },
    "app_clustering": {
        "name": "Агент: Кластеризация и Разметка",
        "icon_path": "assets/icon_clustering.png",
        "command": [sys.executable, "-m", "streamlit", "run", "app_clustering.py", "--server.port", "8503", "--server.headless", "true"],
        "script": "app_clustering.py",
        "url": "http://localhost:8503",
        "type": "Streamlit"
    },
    "app_modeling": {
        "name": "Агент: Предиктивное Моделирование",
        "icon_path": "assets/icon_modeling.png",
        "command": [sys.executable, "-m", "streamlit", "run", "app_modeling.py", "--server.port", "8504", "--server.headless", "true"],
        "script": "app_modeling.py",
        "url": "http://localhost:8504",
        "type": "Streamlit"
    },
    "backend_api": {
        "name": "Агент: API Шлюз",
        "icon_path": "assets/icon_api.png",
        "command": [sys.executable, "backend_api.py"],
        "script": "backend_api.py",
        "url": "http://127.0.0.1:8000/docs",
        "type": "FastAPI"
    },
    "telegram_bot": {
        "name": "Агент: Telegram Интерфейс",
        "icon_path": "assets/icon_bot.png",
        "command": [sys.executable, "telegram_bot.py"],
        "script": "telegram_bot.py",
        "url": None,
        "type": "Aiogram"
    },
    "app_bot_admin": {
        "name": "Центр Управления Ботом",
        "icon_path": "assets/icon_bot_admin.png",
        "command": [sys.executable, "-m", "streamlit", "run", "app_bot_admin.py", "--server.port", "8505", "--server.headless", "true"],
        "script": "app_bot_admin.py",
        "url": "http://localhost:8505",
        "type": "Streamlit"
    }
}

def is_running(script_name):
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline:
                for cmd in cmdline:
                    if cmd.endswith(script_name):
                        return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

def start_service(service_id):
    cfg = SERVICES[service_id]
    env = os.environ.copy()
    subprocess.Popen(cfg["command"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    time.sleep(1.5)

def stop_service(pid):
    try:
        proc = psutil.Process(pid)
        for child in proc.children(recursive=True):
            child.terminate()
        proc.terminate()
        time.sleep(1)
    except Exception as e:
        st.error(f"Ошибка остановки процесса (PID {pid}): {e}")

st.divider()

cols = st.columns(2)

for idx, (svc_id, svc_info) in enumerate(SERVICES.items()):
    col = cols[idx % 2]
    with col:
        with st.container(border=True):
            img_col, txt_col = st.columns([1, 4])
            with img_col:
                if os.path.exists(svc_info.get('icon_path', '')):
                    st.image(svc_info['icon_path'], width="stretch")
            with txt_col:
                st.subheader(svc_info['name'])
                st.caption(f"Архитектура: {svc_info['type']}")

            pid = is_running(svc_info["script"])

            status_col, action_col = st.columns([1, 1])

            with status_col:
                if pid:
                    st.success(f"СТАТУС: В СЕТИ (PID: {pid})")
                else:
                    st.error("СТАТУС: ОТКЛЮЧЕН")

            with action_col:
                if pid:
                    if st.button("Остановить Агента", key=f"stop_{svc_id}", width="stretch"):
                        stop_service(pid)
                        st.rerun()
                else:
                    if st.button("Развернуть Агента", key=f"start_{svc_id}", width="stretch"):
                        start_service(svc_id)
                        st.rerun()

            if svc_info["url"]:
                if pid:
                    st.markdown(f"**Точка доступа:** [{svc_info['url']}]({svc_info['url']})")
                else:
                    st.markdown(f"**Точка доступа:** `{svc_info['url']}`")
            else:
                st.markdown("**Точка доступа:** `Только Консоль`")

st.divider()

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Обновить Статус Системы", width="stretch"):
        st.rerun()
with col2:
    st.info("Системное уведомление: Разверните агента для инициализации базовых процессов. Интерфейсы станут доступны по указанным точкам доступа.")
