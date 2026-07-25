from config import DATABASE_URL, OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN
import asyncio
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile, InputMediaPhoto
import aiohttp
import os
import io
from sqlalchemy import create_engine, text
from datetime import datetime
from aiogram import BaseMiddleware

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch
from aiogram.types import BufferedInputFile
import matplotlib.pyplot as plt

TOKEN = TELEGRAM_BOT_TOKEN
API_URL = "http://127.0.0.1:8000"

logging.basicConfig(level=logging.INFO)
bot = Bot(token=TOKEN)
dp = Dispatcher()


DB_URL = DATABASE_URL
engine = create_engine(DB_URL)

def init_db():
    with engine.begin() as conn:
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS bot_users (
                user_id BIGINT PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                joined_at TIMESTAMP,
                last_active TIMESTAMP
            )
        '''))
init_db()

def update_user_activity(user):
    try:
        with engine.begin() as conn:
            conn.execute(text('''
                INSERT INTO bot_users (user_id, username, first_name, joined_at, last_active)
                VALUES (:uid, :uname, :fname, :now, :now)
                ON CONFLICT (user_id) DO UPDATE SET
                    username = EXCLUDED.username,
                    first_name = EXCLUDED.first_name,
                    last_active = EXCLUDED.last_active
            '''), {
                "uid": user.id,
                "uname": user.username or "",
                "fname": user.first_name or "",
                "now": datetime.now()
            })
    except Exception as e:
        logging.error(f"DB Error: {e}")

class ActivityMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        user = data.get("event_from_user")
        if user:
            update_user_activity(user)
        return await handler(event, data)

dp.message.middleware(ActivityMiddleware())
dp.callback_query.middleware(ActivityMiddleware())



user_carts = {}
user_trajectories = {}

def add_background(canvas, doc):
    canvas.saveState()
    canvas.setFillColorRGB(13/255.0, 17/255.0, 23/255.0)
    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1, stroke=0)
    canvas.restoreState()

def create_load_chart(weeks_data):
    x = [f"Нед {w['Неделя']}" for w in weeks_data]
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

def generate_simple_pdf(weeks_data, total_hours):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)

    try:
        pdfmetrics.registerFont(TTFont('Arial', r'C:\Windows\Fonts\arial.ttf'))
        pdfmetrics.registerFont(TTFont('Arial-Bold', r'C:\Windows\Fonts\arialbd.ttf'))
        font_main = 'Arial'
        font_bold = 'Arial-Bold'
    except:
        font_main = 'Helvetica'
        font_bold = 'Helvetica-Bold'

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CustomTitle', fontName=font_bold, fontSize=24, spaceAfter=20, alignment=1, textColor='#f9a03f'))
    styles.add(ParagraphStyle(name='CustomSummary', fontName=font_main, fontSize=14, spaceAfter=25, alignment=1, textColor='#c9d1d9'))
    styles.add(ParagraphStyle(name='CustomH1', fontName=font_bold, fontSize=18, spaceAfter=15, textColor='#58a6ff'))
    styles.add(ParagraphStyle(name='CustomDay', fontName=font_bold, fontSize=14, spaceAfter=10, textColor='#79c0ff', leftIndent=10))
    styles.add(ParagraphStyle(name='CustomItemTitle', fontName=font_bold, fontSize=12, spaceAfter=5, textColor='#ffffff', leftIndent=25))

    elements = []
    elements.append(Paragraph("Ваш ежедневный план обучения", styles['CustomTitle']))

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
                time_h = round(mat_item['time_min']/60, 1)

                subj = mat_item.get('subject', 'Материал')
                topic = mat_item.get('topic', 'Без названия')

                part_label = mat_item.get('part_label', '')
                if part_label: part_label = f" {part_label}"

                title = f"{idx+1}. {subj}: {topic}{part_label} — {time_h} ч"
                mat_elements.append(Paragraph(title, styles['CustomItemTitle']))

                elements.append(KeepTogether(mat_elements))
                elements.append(Spacer(1, 0.1*inch))

            elements.append(Spacer(1, 0.1*inch))

    doc.build(elements, onFirstPage=add_background, onLaterPages=add_background)
    buffer.seek(0)
    return buffer.getvalue()

class EvaluateState(StatesGroup):
    waiting_for_text = State()

class TrajectoryState(StatesGroup):
    waiting_for_hours = State()
    waiting_for_days = State()

def get_main_keyboard():
    kb = InlineKeyboardBuilder()
    kb.add(InlineKeyboardButton(text="Оценить материал", callback_data="eval_material"))
    kb.add(InlineKeyboardButton(text="Каталог материалов", callback_data="show_catalog_0"))
    kb.add(InlineKeyboardButton(text="Моя полка", callback_data="show_cart"))
    kb.add(InlineKeyboardButton(text="Построить траекторию", callback_data="build_traj"))
    kb.adjust(1)
    return kb.as_markup()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    if user_id not in user_carts:
        user_carts[user_id] = []
    photo = FSInputFile("icons/welcome.png")
    await message.answer_photo(photo=photo, caption="Добро пожаловать в Sapere Bot! Выберите действие:", reply_markup=get_main_keyboard())


@dp.callback_query(F.data.startswith("show_catalog_"))
async def show_catalog(callback_query: types.CallbackQuery):
    page = int(callback_query.data.split("_")[2])

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/materials") as resp:
            if resp.status != 200:
                await callback_query.answer("Ошибка при загрузке каталога.", show_alert=True)
                return
            materials = await resp.json()

    if not materials:
        await callback_query.message.answer("Каталог пуст.")
        return

    if page < 0: page = 0
    if page >= len(materials): page = len(materials) - 1

    mat = materials[page]

    text = f"📚 **Каталог материалов** ({page + 1}/{len(materials)})\n\n"
    text += f"**Файл:** {mat['file_name']}\n"
    text += f"**Предмет:** {mat['subject']}\n"
    text += f"**Тема:** {mat['topic']}\n"
    text += f"**Сложность:** {mat['complexity']}\n"
    text += f"**Слов:** {mat['word_count']} | **Изображений:** {mat['images_count']}\n"

    kb = InlineKeyboardBuilder()


    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"show_catalog_{page-1}"))
    if page < len(materials) - 1:
        nav_row.append(InlineKeyboardButton(text="Вперед ➡️", callback_data=f"show_catalog_{page+1}"))
    if nav_row:
        kb.row(*nav_row)


    kb.row(InlineKeyboardButton(text="📥 Скачать файл", callback_data=f"dl_{mat['id']}"))
    kb.row(InlineKeyboardButton(text="➕ В расписание", callback_data=f"add_{mat['id']}"))
    kb.row(InlineKeyboardButton(text="🏠 В меню", callback_data="menu"))

    try:
        media = InputMediaPhoto(media=FSInputFile("icons/catalog.png"), caption=text, parse_mode="Markdown")
        await callback_query.message.edit_media(media=media, reply_markup=kb.as_markup())
    except:
        try:
            await callback_query.message.delete()
        except:
            pass
        await callback_query.message.answer_photo(photo=FSInputFile("icons/catalog.png"), caption=text, parse_mode="Markdown", reply_markup=kb.as_markup())
    await bot.answer_callback_query(callback_query.id)

@dp.callback_query(F.data.startswith("dl_"))
async def download_file(callback_query: types.CallbackQuery):
    mat_id = callback_query.data[3:]
    await callback_query.answer("Загружаю файл...")

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/materials") as resp:
            materials = await resp.json()

    mat = next((m for m in materials if str(m['id']) == str(mat_id)), None)
    if not mat:
        await bot.send_message(callback_query.from_user.id, "Файл не найден в базе.")
        return

    file_name = mat['file_name']



    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/materials/{file_name}/download") as resp:
            if resp.status == 200:

                content = await resp.read()
                temp_path = f"temp_{file_name}"
                with open(temp_path, "wb") as f:
                    f.write(content)

                doc = FSInputFile(temp_path)
                await bot.send_document(callback_query.from_user.id, doc)
                os.remove(temp_path)
            else:
                await bot.send_message(callback_query.from_user.id, "Ошибка скачивания файла (возможно, он удален).")

@dp.callback_query(F.data.startswith("add_"))
async def add_to_cart(callback_query: types.CallbackQuery):
    mat_id = callback_query.data[4:]
    user_id = callback_query.from_user.id

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_URL}/materials") as resp:
            materials = await resp.json()

    mat = next((m for m in materials if str(m['id']) == str(mat_id)), None)
    if mat:
        if user_id not in user_carts:
            user_carts[user_id] = []
        user_carts[user_id].append(mat)
        await callback_query.answer(f"Добавлено: {mat['file_name']}", show_alert=True)
    else:
        await callback_query.answer("Материал не найден.", show_alert=True)


@dp.callback_query(F.data == "show_cart")
async def show_cart(callback_query: types.CallbackQuery):
    user_id = callback_query.from_user.id
    cart = user_carts.get(user_id, [])

    if not cart:
        text = "Ваша полка пуста. Добавьте материалы из каталога!"
        kb = InlineKeyboardBuilder()
        kb.add(InlineKeyboardButton(text="Каталог", callback_data="show_catalog_0"))
        kb.add(InlineKeyboardButton(text="🏠 В меню", callback_data="menu"))
        kb.adjust(1)
        try:
            await callback_query.message.delete()
        except:
            pass
        photo = FSInputFile("icons/shelf.png")
        await callback_query.message.answer_photo(photo=photo, caption=text, reply_markup=kb.as_markup())
        return

    text = "🎒 **Ваши выбранные материалы:**\n\n"
    for i, m in enumerate(cart):
        text += f"{i+1}. {m['subject']}: {m['topic']} ({m['complexity']})\n"

    kb = InlineKeyboardBuilder()
    kb.add(InlineKeyboardButton(text="Очистить", callback_data="clear_cart"))
    kb.add(InlineKeyboardButton(text="Построить траекторию", callback_data="build_traj"))
    kb.add(InlineKeyboardButton(text="🏠 В меню", callback_data="menu"))
    kb.adjust(1)

    try:
        await callback_query.message.delete()
    except:
        pass
    photo = FSInputFile("icons/cart.png")
    await callback_query.message.answer_photo(photo=photo, caption=text, parse_mode="Markdown", reply_markup=kb.as_markup())
    await bot.answer_callback_query(callback_query.id)

@dp.callback_query(F.data == "clear_cart")
async def clear_cart(callback_query: types.CallbackQuery):
    user_carts[callback_query.from_user.id] = []
    await callback_query.answer("Полка очищена", show_alert=True)
    await show_cart(callback_query)

@dp.callback_query(F.data == "menu")
async def back_to_menu(callback_query: types.CallbackQuery):
    try:
        await callback_query.message.delete()
    except:
        pass
    photo = FSInputFile("icons/welcome.png")
    await bot.send_photo(callback_query.from_user.id, photo=photo, caption="Выберите действие:", reply_markup=get_main_keyboard())
    await bot.answer_callback_query(callback_query.id)


@dp.callback_query(F.data == "eval_material")
async def process_eval_material(callback_query: types.CallbackQuery, state: FSMContext):
    await bot.answer_callback_query(callback_query.id)
    try:
        await callback_query.message.delete()
    except:
        pass
    photo = FSInputFile("icons/evaluate.png")
    await bot.send_photo(callback_query.from_user.id, photo=photo, caption="Пришлите текст или файл (PDF/DOCX) материала для оценки:")
    await state.set_state(EvaluateState.waiting_for_text)

async def _process_evaluation(message: types.Message, state: FSMContext = None):
    is_file = message.document is not None
    data_json = {}

    if is_file:
        ext = os.path.splitext(message.document.file_name)[1].lower()
        if ext not in ['.pdf', '.docx']:
            await message.answer("⚠️ Пожалуйста, отправьте файл в формате PDF или DOCX.", reply_markup=get_main_keyboard())
            if state: await state.clear()
            return

        await message.answer("Загружаю и анализирую файл...")
        file_info = await bot.get_file(message.document.file_id)
        file_bytes = io.BytesIO()
        await bot.download_file(file_info.file_path, destination=file_bytes)
        file_bytes.seek(0)

        try:
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field('file', file_bytes, filename=message.document.file_name)

                async with session.post(f"{API_URL}/evaluate_file", data=data) as resp:
                    if resp.status == 200:
                        data_json = await resp.json()
                    else:
                        await message.answer("Ошибка при обработке файла.")
                        if state: await state.clear()
                        return
                word_count = 2000
        except Exception as e:
            await message.answer("Ошибка соединения с API-сервером.")
            logging.error(f"Error: {e}")
            if state: await state.clear()
            return

    else:
        text = message.text
        if not text:
            await message.answer("Пожалуйста, отправьте текст.")
            return
        await message.answer("Анализирую текст...")
        word_count = len(text.split())

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{API_URL}/evaluate", json={"text_content": text}) as resp:
                    if resp.status == 200:
                        data_json = await resp.json()
                    else:
                        await message.answer("Ошибка при вызове API.")
                        if state: await state.clear()
                        return
        except Exception as e:
            await message.answer("Ошибка соединения с API-сервером.")
            logging.error(f"Error: {e}")
            if state: await state.clear()
            return

    if not data_json.get("is_valid", True):
        await message.answer("⚠️ Материал не похож на учебный.", parse_mode="Markdown", reply_markup=get_main_keyboard())
        if state: await state.clear()
        return

    res = f"📊 *Оценка материала:*\n" \
          f"Соответствие метод. рекомендациям: {data_json.get('conformance')}\n" \
          f"Сложность: {data_json.get('difficulty')}\n" \
          f"Можно изучать параллельно: {data_json.get('parallel')}\n" \
          f"Можно изучать последовательно: {data_json.get('sequential')}\n\n" \
          f"🤖 *Аудит ИИ:*\n{data_json.get('explanation', '')}"

    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "word_count": word_count,
                "images_count": 0,
                "complexity": data_json.get('difficulty', 'Средняя'),
                "practice_orientation": 0.5
            }
            async with session.post(f"{API_URL}/estimate_time", json=payload) as resp2:
                if resp2.status == 200:
                    t_data = await resp2.json()
                    res += f"\n\n⏱ Оценочное время изучения: {round(t_data['time_min']/60, 1)} ч."
    except Exception as e:
        logging.error(f"Estimate Error: {e}")

    await message.answer(res, parse_mode="Markdown", reply_markup=get_main_keyboard())
    if state: await state.clear()

@dp.message(EvaluateState.waiting_for_text)
async def evaluate_text_state(message: types.Message, state: FSMContext):
    await _process_evaluation(message, state)

@dp.message(F.document)
async def evaluate_document_direct(message: types.Message):
    await _process_evaluation(message, None)


@dp.callback_query(F.data == "build_traj")
async def process_build_traj(callback_query: types.CallbackQuery, state: FSMContext):
    user_id = callback_query.from_user.id
    cart = user_carts.get(user_id, [])

    if not cart:
        await callback_query.answer("Полка пуста! Сначала добавьте материалы из каталога.", show_alert=True)
        return

    await bot.answer_callback_query(callback_query.id)
    try:
        await callback_query.message.delete()
    except:
        pass
    photo = FSInputFile("icons/trajectory.png")
    await bot.send_photo(user_id, photo=photo, caption="Сколько часов в неделю вы готовы учиться? (Напишите число, например 10)")
    await state.set_state(TrajectoryState.waiting_for_hours)

@dp.message(TrajectoryState.waiting_for_hours)
async def traj_hours(message: types.Message, state: FSMContext):
    try:
        hours = float(message.text)
        await state.update_data(hours=hours)
        await message.answer("Сколько дней в неделю вы планируете учиться? (от 1 до 7)")
        await state.set_state(TrajectoryState.waiting_for_days)
    except ValueError:
        await message.answer("Пожалуйста, введите число.")

@dp.message(TrajectoryState.waiting_for_days)
async def traj_days(message: types.Message, state: FSMContext):
    try:
        days = int(message.text)
        data = await state.get_data()
        hours = data['hours']

        user_id = message.from_user.id
        cart = user_carts.get(user_id, [])

        await message.answer(f"Генерирую расписание для {len(cart)} материалов...")


        async with aiohttp.ClientSession() as session:
            mat_payloads = []
            for m in cart:
                est_payload = {
                    "word_count": m['word_count'],
                    "images_count": m['images_count'],
                    "complexity": m['complexity'],
                    "practice_orientation": 0.5
                }
                async with session.post(f"{API_URL}/estimate_time", json=est_payload) as r:
                    t_data = await r.json()
                    est_time = t_data['time_min']

                mat_payloads.append({
                    "id": m['id'],
                    "subject": m['subject'],
                    "topic": m['topic'],
                    "est_time_min": est_time
                })

            payload = {
                "hours_per_week": hours,
                "days_per_week": days,
                "materials": mat_payloads
            }

            async with session.post(f"{API_URL}/build_trajectory", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    user_trajectories[user_id] = {
                        "weeks": data['weeks'],
                        "total_hours": sum(sum(d['Занято_мин'] for d in w['Дни']) for w in data['weeks']) / 60.0
                    }

                    res = "📌 *Ваша персональная траектория:*\n\n"
                    for w in data['weeks']:
                        res += f"*Неделя {w['Неделя']}*\n"
                        for d in w['Дни']:
                            res += f"  🔹 День {d['День']}:\n"
                            for item in d['План_объекты']:
                                res += f"    - {item['subject']}: {item['topic']} {item.get('part_label', '')} ({round(item['time_min']/60, 1)} ч)\n"

                    kb = InlineKeyboardBuilder()
                    kb.add(InlineKeyboardButton(text="📥 Скачать PDF план", callback_data="download_traj_pdf"))
                    kb.add(InlineKeyboardButton(text="В главное меню", callback_data="menu"))
                    kb.adjust(1)

                    await message.answer(res, parse_mode="Markdown", reply_markup=kb.as_markup())
                else:
                    await message.answer("Ошибка при генерации расписания.", reply_markup=get_main_keyboard())

        await state.clear()
    except ValueError:
        await message.answer("Пожалуйста, введите число от 1 до 7.")

@dp.callback_query(F.data == "download_traj_pdf")
async def process_dl_traj_pdf(callback_query: types.CallbackQuery):
    user_id = callback_query.from_user.id
    traj_data = user_trajectories.get(user_id)

    if not traj_data:
        await callback_query.answer("План не найден. Пожалуйста, сгенерируйте его заново.", show_alert=True)
        return

    await callback_query.answer("Генерирую PDF...")
    try:
        pdf_bytes = generate_simple_pdf(traj_data["weeks"], traj_data["total_hours"])
        file = BufferedInputFile(pdf_bytes, filename="sapere_course_plan.pdf")

        await bot.send_document(chat_id=user_id, document=file, caption="Ваш персональный план обучения 🎓")
    except Exception as e:
        logging.error(f"Error generating PDF: {e}")
        await bot.send_message(user_id, "Извините, произошла ошибка при генерации PDF документа.")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
