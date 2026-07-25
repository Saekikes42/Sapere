from config import DATABASE_URL, OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN
import os
import json
import re
import datetime
import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go


DB_URL = DATABASE_URL
AI_API_KEY = OPENROUTER_API_KEY
COMPLETENESS_FILE = "subject_completeness.json"

engine = create_engine(DB_URL)


def get_data_from_db():
    try:
        df = pd.read_sql("SELECT * FROM opit_files", engine)
        if df.empty:
            return pd.DataFrame()

        df['is_generated'] = df['file_name'].str.startswith('AI_') | df['annotation'].str.contains('Сгенерировано', na=False)

        compliance_cols = [
            'compliance_rate', 'compliance_structure', 'compliance_clarity',
            'compliance_terminology', 'compliance_completeness',
            'practice_orientation', 'legal_compliance', 'technical_quality'
        ]
        for col in compliance_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        return df
    except Exception as e:
        print(f"Error fetching data from database: {e}")
        return pd.DataFrame()

def filter_dataframe(df, selected_subject, selected_material_type, selected_extension, selected_origin, selected_usage, selected_compliance_range):
    if df.empty:
        return df
    df_filtered = df.copy()

    if selected_subject:
        df_filtered = df_filtered[df_filtered['subject'] == selected_subject]

    if selected_material_type:
        df_filtered = df_filtered[df_filtered['material_type'] == selected_material_type]

    if selected_extension:
        df_filtered = df_filtered[df_filtered['extension'] == selected_extension]

    if selected_origin is not None and selected_origin != "":
        is_gen = (selected_origin == "ai")
        df_filtered = df_filtered[df_filtered['is_generated'] == is_gen]

    if selected_usage:
        df_filtered = df_filtered[df_filtered['usage_allowed'] == selected_usage]

    if selected_compliance_range:
        min_val, max_val = selected_compliance_range

        df_filtered = df_filtered[(df_filtered['compliance_rate'] >= min_val / 100.0) &
                                  (df_filtered['compliance_rate'] <= max_val / 100.0)]

    return df_filtered

def load_completeness_cache():
    if os.path.exists(COMPLETENESS_FILE):
        try:
            with open(COMPLETENESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading completeness cache: {e}")
    return {}

def save_completeness_cache(cache):
    try:
        with open(COMPLETENESS_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving completeness cache: {e}")


def check_completeness_via_ai(subject, existing_topics):
    prompt = f"""Ты - эксперт-методист. Для учебного предмета '{subject}' в базе данных есть следующие темы: {json.dumps(existing_topics, ensure_ascii=False)}.
Пожалуйста, оцени полноту покрытия тем для этого предмета. Перечисли, каких важных тем (разделов) не хватает для полноценного освоения этого предмета.
Ответь строго в формате JSON с ключами 'missing_topics' (список строк) и 'explanation' (строка). Не добавляй никакого другого текста, разметки markdown или объяснений вне JSON."""

    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "google/gemini-2.5-flash",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=45)
        if r.status_code == 200:
            content = r.json()['choices'][0]['message']['content'].strip()

            s = content.find('{')
            e = content.rfind('}')
            if s != -1 and e != -1:
                res = json.loads(content[s:e+1])
                missing_topics = res.get('missing_topics', [])
                explanation = res.get('explanation', '')


                existing_count = len(existing_topics)
                missing_count = len(missing_topics)
                total_count = existing_count + missing_count
                completeness = round((existing_count / total_count) * 100, 2) if total_count > 0 else 0.0

                return {
                    "existing_count": existing_count,
                    "existing_topics": existing_topics,
                    "missing_count": missing_count,
                    "missing_topics": missing_topics,
                    "completeness": completeness,
                    "explanation": explanation
                }
        else:
            print(f"OpenRouter API error: {r.status_code} {r.text}")
    except Exception as e:
        print(f"Exception during OpenRouter call: {e}")
    return None


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"],
    title="Sapere Intelligence - Аналитическая система",
    suppress_callback_exceptions=True
)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/png" href="/assets/icon_analytics.png">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


THEME_BG = "#0d1117"
THEME_CARD_BG = "#161b22"
THEME_BORDER = "#30363d"
THEME_TEXT = "#c9d1d9"
THEME_ACCENT = "#f9a03f"
THEME_ACCENT_ALT = "#58a6ff"


card_style = {
    "backgroundColor": THEME_CARD_BG,
    "border": f"1px solid {THEME_BORDER}",
    "borderRadius": "8px",
    "padding": "20px",
    "marginBottom": "20px",
    "color": THEME_TEXT
}

sidebar_style = {
    "backgroundColor": THEME_CARD_BG,
    "borderRight": f"1px solid {THEME_BORDER}",
    "padding": "25px 20px",
    "minHeight": "calc(100vh - 65px)",
    "color": THEME_TEXT
}




def build_login_elements():
    return dbc.Card(
        dbc.CardBody([
            html.Div(
                [
                    html.H1("Sapere", style={"color": THEME_ACCENT, "fontWeight": "700", "fontSize": "36px", "letterSpacing": "-0.02em", "marginBottom": "5px"}),
                    html.H5("Intelligence", style={"color": THEME_TEXT, "fontWeight": "400", "fontSize": "14px", "letterSpacing": "0.1em", "textTransform": "uppercase", "marginBottom": "20px"}),
                    html.P("Аналитическая система контроля качества", style={"color": "#8b949e", "fontSize": "13px", "marginBottom": "20px"})
                ],
                style={"textAlign": "center"}
            ),

            html.Hr(style={"borderColor": THEME_BORDER, "marginBottom": "20px"}),

            dbc.Row([

                dbc.Col([
                    html.Label("Логин:", style={"fontWeight": "600", "color": THEME_TEXT, "marginBottom": "6px"}),
                    dcc.Input(
                        id="login-username-input",
                        type="text",
                        placeholder="Введите логин",
                        style={
                            "width": "100%",
                            "backgroundColor": "#0d1117",
                            "color": "#fff",
                            "border": f"1px solid {THEME_BORDER}",
                            "borderRadius": "4px",
                            "padding": "8px 12px",
                            "marginBottom": "15px"
                        }
                    ),

                    html.Label("Пароль:", style={"fontWeight": "600", "color": THEME_TEXT, "marginBottom": "6px"}),
                    dcc.Input(
                        id="login-password-input",
                        type="password",
                        placeholder="Введите пароль",
                        style={
                            "width": "100%",
                            "backgroundColor": "#0d1117",
                            "color": "#fff",
                            "border": f"1px solid {THEME_BORDER}",
                            "borderRadius": "4px",
                            "padding": "8px 12px",
                            "marginBottom": "20px"
                        }
                    ),


                    dbc.Button(
                        "Войти в систему",
                        id="btn-login",
                        color="warning",
                        size="md",
                        style={"width": "100%", "fontWeight": "600", "backgroundColor": THEME_ACCENT, "color": "#0d1117", "border": "none", "borderRadius": "6px"}
                    ),


                    html.Div(
                        id="login-feedback",
                        style={"color": "#ff5252", "fontSize": "13px", "textAlign": "center", "marginTop": "12px", "fontWeight": "500"}
                    )
                ], width=7),


                dbc.Col([
                    html.Div([
                        html.H6("Подсказка по входу:", style={"color": THEME_ACCENT, "fontWeight": "600", "marginBottom": "12px"}),

                        html.Div([
                            html.Strong("Преподаватель", style={"fontSize": "13px", "color": THEME_TEXT}),
                            html.Div("Логин: teacher", style={"fontSize": "12px", "color": "#8b949e"}),
                            html.Div("Пароль: (не требуется)", style={"fontSize": "12px", "color": "#8b949e"}),
                        ], style={"marginBottom": "12px"}),

                        html.Div([
                            html.Strong("Методист", style={"fontSize": "13px", "color": THEME_TEXT}),
                            html.Div("Логин: methodist", style={"fontSize": "12px", "color": "#8b949e"}),
                            html.Div("Пароль: methodist", style={"fontSize": "12px", "color": "#8b949e"}),
                        ], style={"marginBottom": "12px"}),

                        html.Div([
                            html.Strong("Администратор", style={"fontSize": "13px", "color": THEME_TEXT}),
                            html.Div("Логин: admin", style={"fontSize": "12px", "color": "#8b949e"}),
                            html.Div("Пароль: admin", style={"fontSize": "12px", "color": "#8b949e"}),
                        ]),
                    ], style={"padding": "12px", "backgroundColor": "#0d1117", "borderRadius": "6px", "border": f"1px solid {THEME_BORDER}"})
                ], width=5, style={"borderLeft": f"1px solid {THEME_BORDER}", "paddingLeft": "15px"})
            ])
        ]),
        style={
            "width": "680px",
            "backgroundColor": THEME_CARD_BG,
            "border": f"1px solid {THEME_BORDER}",
            "borderRadius": "12px",
            "boxShadow": "0 10px 25px rgba(0,0,0,0.5)",
            "padding": "25px"
        }
    )


def build_dashboard_elements():
    return html.Div(
        [

            dbc.Navbar(
                dbc.Container(
                    [
                        dbc.NavbarBrand(
                            [
                                html.Span("Sapere ", style={"color": THEME_ACCENT, "fontWeight": "700", "fontSize": "22px"}),
                                html.Span("Intelligence - Аналитика", style={"color": THEME_TEXT, "fontWeight": "400", "fontSize": "20px"})
                            ],
                            href="#"
                        ),
                        html.Div(
                            [
                                html.Span(id="active-role-text", style={"color": THEME_TEXT, "marginRight": "20px", "fontSize": "14px", "fontWeight": "500", "border": f"1px solid {THEME_BORDER}", "padding": "5px 12px", "borderRadius": "15px", "backgroundColor": "#21262d"}),
                                dbc.Button("Выйти из системы", id="btn-logout", color="danger", size="sm", style={"fontWeight": "600", "borderRadius": "6px"})
                            ],
                            style={"display": "flex", "alignItems": "center"}
                        )
                    ],
                    fluid=True
                ),
                color=THEME_CARD_BG,
                dark=True,
                style={"borderBottom": f"1px solid {THEME_BORDER}", "height": "65px"}
            ),


            dbc.Row(
                [

                    dbc.Col(
                        [
                            html.Label("Предмет:", style={"fontWeight": "600", "color": THEME_TEXT, "marginBottom": "6px", "fontSize": "13px"}),
                            dcc.Dropdown(
                                id="subject-dropdown",
                                placeholder="Все предметы",
                                clearable=True,
                                style={"backgroundColor": "#0d1117", "color": "#fff", "marginBottom": "15px"}
                            ),

                            html.Label("Тип материала:", style={"fontWeight": "600", "color": THEME_TEXT, "marginBottom": "6px", "fontSize": "13px"}),
                            dcc.Dropdown(
                                id="material-type-dropdown",
                                placeholder="Все типы",
                                clearable=True,
                                style={"backgroundColor": "#0d1117", "color": "#fff", "marginBottom": "15px"}
                            ),

                            html.Label("Формат файла:", style={"fontWeight": "600", "color": THEME_TEXT, "marginBottom": "6px", "fontSize": "13px"}),
                            dcc.Dropdown(
                                id="extension-dropdown",
                                placeholder="Все форматы",
                                clearable=True,
                                style={"backgroundColor": "#0d1117", "color": "#fff", "marginBottom": "15px"}
                            ),

                            html.Label("Происхождение:", style={"fontWeight": "600", "color": THEME_TEXT, "marginBottom": "6px", "fontSize": "13px"}),
                            dcc.Dropdown(
                                id="origin-dropdown",
                                options=[
                                    {"label": "Все материалы", "value": ""},
                                    {"label": "Исходные материалы", "value": "original"},
                                    {"label": "Сгенерировано ИИ", "value": "ai"}
                                ],
                                value="",
                                clearable=False,
                                style={"backgroundColor": "#0d1117", "color": "#fff", "marginBottom": "15px"}
                            ),

                            html.Label("Допуск к использованию:", style={"fontWeight": "600", "color": THEME_TEXT, "marginBottom": "6px", "fontSize": "13px"}),
                            dcc.Dropdown(
                                id="usage-dropdown",
                                options=[
                                    {"label": "Все", "value": ""},
                                    {"label": "Допущен", "value": "yes"},
                                    {"label": "Не допущен", "value": "no"}
                                ],
                                value="",
                                clearable=False,
                                style={"backgroundColor": "#0d1117", "color": "#fff", "marginBottom": "15px"}
                            ),

                            html.Label("Оценка ФГОС (%):", style={"fontWeight": "600", "color": THEME_TEXT, "marginBottom": "6px", "fontSize": "13px"}),
                            html.Div([
                                dcc.RangeSlider(
                                    id="compliance-slider",
                                    min=0,
                                    max=100,
                                    step=5,
                                    value=[0, 100],
                                    marks={0: '0%', 50: '50%', 100: '100%'},
                                )
                            ], style={"marginBottom": "20px", "padding": "0 5px"}),
                            html.Hr(style={"borderColor": THEME_BORDER, "marginBottom": "25px"}),

                            dbc.Button(
                                "🔄 Синхронизировать БД",
                                id="btn-refresh-db",
                                color="warning",
                                style={"width": "100%", "fontWeight": "600", "backgroundColor": THEME_CARD_BG, "borderColor": THEME_BORDER, "color": THEME_ACCENT, "marginBottom": "10px"}
                            ),
                            html.Div(
                                id="last-updated-text",
                                style={"fontSize": "11px", "color": "#8b949e", "textAlign": "center"}
                            ),

                            html.Hr(style={"borderColor": THEME_BORDER, "marginTop": "25px", "marginBottom": "20px"}),

                            html.Div(
                                [
                                    html.Small("Система Sapere автоматически подгружает данные из Postgres и рассчитывает метрики качества материалов по стандартам ФГОС.", style={"color": "#8b949e", "fontSize": "12px", "lineHeight": "1.4"})
                                ],
                                style={"padding": "10px", "border": f"1px dashed {THEME_BORDER}", "borderRadius": "6px"}
                            )
                        ],
                        width=3,
                        style=sidebar_style
                    ),


                    dbc.Col(
                        [

                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody([
                                                html.H6("Всего материалов", className="text-muted", style={"fontSize": "12px", "textTransform": "uppercase"}),
                                                html.H3(id="stat-total-materials", style={"color": THEME_ACCENT, "fontWeight": "700", "marginBottom": "0"})
                                            ]),
                                            style=card_style
                                        ),
                                        width=3
                                    ),
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody([
                                                html.H6("Соответствие ФГОС", className="text-muted", style={"fontSize": "12px", "textTransform": "uppercase"}),
                                                html.H3(id="stat-avg-compliance", style={"color": THEME_ACCENT, "fontWeight": "700", "marginBottom": "0"})
                                            ]),
                                            style=card_style
                                        ),
                                        width=3
                                    ),
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody([
                                                html.H6("Сгенерировано ИИ", className="text-muted", style={"fontSize": "12px", "textTransform": "uppercase"}),
                                                html.H3(id="stat-generated-share", style={"color": THEME_ACCENT, "fontWeight": "700", "marginBottom": "0"})
                                            ]),
                                            style=card_style
                                        ),
                                        width=3
                                    ),
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody([
                                                html.H6("Заполненость тем", className="text-muted", style={"fontSize": "12px", "textTransform": "uppercase"}),
                                                html.H3(id="stat-avg-completeness", style={"color": THEME_ACCENT, "fontWeight": "700", "marginBottom": "0"})
                                            ]),
                                            style=card_style
                                        ),
                                        width=3
                                    ),
                                ],
                                style={"marginTop": "20px", "paddingLeft": "10px", "paddingRight": "10px"}
                            ),


                            dcc.Tabs(
                                id="dashboard-tabs",
                                value="tab-overview",
                                children=[
                                    dcc.Tab(
                                        label="Общий обзор покрытия",
                                        value="tab-overview",
                                        style={"backgroundColor": THEME_CARD_BG, "color": "#8b949e", "border": f"1px solid {THEME_BORDER}"},
                                        selected_style={"backgroundColor": "#21262d", "color": THEME_ACCENT, "borderTop": f"2px solid {THEME_ACCENT}", "borderLeft": f"1px solid {THEME_BORDER}", "borderRight": f"1px solid {THEME_BORDER}", "fontWeight": "600"}
                                    ),
                                    dcc.Tab(
                                        label="Анализ занятий и категорий",
                                        value="tab-lessons",
                                        style={"backgroundColor": THEME_CARD_BG, "color": "#8b949e", "border": f"1px solid {THEME_BORDER}"},
                                        selected_style={"backgroundColor": "#21262d", "color": THEME_ACCENT, "borderTop": f"2px solid {THEME_ACCENT}", "borderLeft": f"1px solid {THEME_BORDER}", "borderRight": f"1px solid {THEME_BORDER}", "fontWeight": "600"}
                                    ),
                                    dcc.Tab(
                                        label="Детализация требований",
                                        value="tab-requirements",
                                        style={"backgroundColor": THEME_CARD_BG, "color": "#8b949e", "border": f"1px solid {THEME_BORDER}"},
                                        selected_style={"backgroundColor": "#21262d", "color": THEME_ACCENT, "borderTop": f"2px solid {THEME_ACCENT}", "borderLeft": f"1px solid {THEME_BORDER}", "borderRight": f"1px solid {THEME_BORDER}", "fontWeight": "600"}
                                    ),
                                    dcc.Tab(
                                        label="Интеллектуальный ИИ-аудит",
                                        value="tab-ai-audit",
                                        style={"backgroundColor": THEME_CARD_BG, "color": "#8b949e", "border": f"1px solid {THEME_BORDER}"},
                                        selected_style={"backgroundColor": "#21262d", "color": THEME_ACCENT, "borderTop": f"2px solid {THEME_ACCENT}", "borderLeft": f"1px solid {THEME_BORDER}", "borderRight": f"1px solid {THEME_BORDER}", "fontWeight": "600"}
                                    ),
                                    dcc.Tab(
                                        label="Замечания и рекомендации",
                                        value="tab-issues",
                                        style={"backgroundColor": THEME_CARD_BG, "color": "#8b949e", "border": f"1px solid {THEME_BORDER}"},
                                        selected_style={"backgroundColor": "#21262d", "color": THEME_ACCENT, "borderTop": f"2px solid {THEME_ACCENT}", "borderLeft": f"1px solid {THEME_BORDER}", "borderRight": f"1px solid {THEME_BORDER}", "fontWeight": "600"}
                                    ),
                                ],
                                style={"marginBottom": "20px", "paddingLeft": "10px", "paddingRight": "10px"}
                            ),


                            dcc.Loading(
                                id="loading-content",
                                type="circle",
                                color=THEME_ACCENT,
                                children=html.Div(id="tab-content", style={"minHeight": "400px", "padding": "10px"})
                            )
                        ],
                        width=9,
                        style={"paddingRight": "25px"}
                    )
                ]
            )
        ]
    )


app.layout = dbc.Container(
    [

        dcc.Store(id="session-store", data={"logged_in": False, "role": "Преподаватель"}),
        dcc.Store(id="data-trigger", data=0),
        dcc.Interval(id="auto-refresh", interval=30000, n_intervals=0),


        html.Div(
            id="login-view-container",
            children=build_login_elements(),
            style={"display": "flex", "justifyContent": "center", "alignItems": "center", "minHeight": "100vh"}
        ),


        html.Div(
            id="dashboard-view-container",
            children=build_dashboard_elements(),
            style={"display": "none"}
        )
    ],
    fluid=True,
    style={"backgroundColor": THEME_BG, "minHeight": "100vh", "padding": "0px", "fontFamily": "'Inter', sans-serif"}
)




@app.callback(
    [Output("login-view-container", "style"),
     Output("dashboard-view-container", "style"),
     Output("active-role-text", "children")],
    [Input("session-store", "data")]
)
def toggle_views(session_data):
    if session_data.get("logged_in", False):
        role_label = session_data.get("role", "Преподаватель")
        return {"display": "none"}, {"display": "block"}, f"Доступ: {role_label}"
    return {"display": "flex", "justifyContent": "center", "alignItems": "center", "minHeight": "100vh"}, {"display": "none"}, ""



@app.callback(
    [Output("session-store", "data"),
     Output("login-feedback", "children")],
    [Input("btn-login", "n_clicks"),
     Input("btn-logout", "n_clicks")],
    [State("login-username-input", "value"),
     State("login-password-input", "value"),
     State("session-store", "data")]
)
def manage_session(n_login, n_logout, username, password, current_session):
    ctx = callback_context
    if not ctx.triggered:
        return current_session, ""

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "btn-login":
        if not n_login:
            return current_session, ""
        if not username:
            return current_session, "❌ Пожалуйста, введите логин"

        u = username.strip().lower()
        p = (password or "").strip()

        if u == "teacher":
            return {"logged_in": True, "role": "Преподаватель"}, ""
        elif u == "methodist":
            if p == "methodist":
                return {"logged_in": True, "role": "Методист"}, ""
            return current_session, "❌ Неверный пароль"
        elif u == "admin":
            if p == "admin":
                return {"logged_in": True, "role": "Администратор"}, ""
            return current_session, "❌ Неверный пароль"
        else:
            return current_session, "❌ Неизвестный логин"

    elif triggered_id == "btn-logout":
        if not n_logout:
            return current_session, ""
        return {"logged_in": False, "role": "Преподаватель"}, ""

    return current_session, ""



@app.callback(
    [Output("data-trigger", "data"),
     Output("last-updated-text", "children"),
     Output("subject-dropdown", "options"),
     Output("material-type-dropdown", "options"),
     Output("extension-dropdown", "options")],
    [Input("btn-refresh-db", "n_clicks"),
     Input("auto-refresh", "n_intervals"),
     Input("session-store", "data")],
    [State("data-trigger", "data")]
)
def sync_data(n_clicks, n_intervals, session_data, trigger_val):

    if not session_data.get("logged_in", False):
        return trigger_val, "", [], [], []

    df = get_data_from_db()
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    status_str = f"Обновлено: {now_str}"

    subject_options = []
    material_options = []
    extension_options = []

    if not df.empty:
        if 'subject' in df.columns:
            subjects = sorted([x for x in df['subject'].dropna().unique() if x.strip()])
            subject_options = [{"label": s, "value": s} for s in subjects]

        if 'material_type' in df.columns:
            mtypes = sorted([x for x in df['material_type'].dropna().unique() if x.strip()])
            material_options = [{"label": m, "value": m} for m in mtypes]

        if 'extension' in df.columns:
            exts = sorted([x for x in df['extension'].dropna().unique() if x.strip()])
            extension_options = [{"label": e.upper().replace('.', ''), "value": e} for e in exts]

    return trigger_val + 1, status_str, subject_options, material_options, extension_options




@app.callback(
    [Output("stat-total-materials", "children"),
     Output("stat-avg-compliance", "children"),
     Output("stat-generated-share", "children"),
     Output("stat-avg-completeness", "children")],
    [Input("data-trigger", "data"),
     Input("subject-dropdown", "value"),
     Input("material-type-dropdown", "value"),
     Input("extension-dropdown", "value"),
     Input("origin-dropdown", "value"),
     Input("usage-dropdown", "value"),
     Input("compliance-slider", "value")],
    [State("session-store", "data")]
)
def update_statistics(trigger, selected_subject, selected_material_type, selected_extension, selected_origin, selected_usage, selected_compliance_range, session_data):
    if not session_data.get("logged_in", False):
        return "0", "0.0%", "0.0%", "0.0%"

    df = get_data_from_db()
    completeness_cache = load_completeness_cache()

    if df.empty:
        return "0", "0.0%", "0.0%", "0.0%"

    df_filtered = filter_dataframe(
        df, selected_subject, selected_material_type, selected_extension,
        selected_origin, selected_usage, selected_compliance_range
    )

    total_materials = len(df_filtered)

    if 'compliance_rate' in df_filtered.columns and len(df_filtered) > 0:
        avg_fgos = df_filtered['compliance_rate'].mean() * 100
        avg_fgos_str = f"{avg_fgos:.1f}%"
    else:
        avg_fgos_str = "0.0%"

    if 'is_generated' in df_filtered.columns and len(df_filtered) > 0:
        gen_share = (df_filtered['is_generated'].sum() / len(df_filtered)) * 100
        gen_share_str = f"{gen_share:.1f}%"
    else:
        gen_share_str = "0.0%"


    if selected_subject:
        subj_comp = completeness_cache.get(selected_subject, {}).get("completeness", 0.0)
        avg_comp_str = f"{subj_comp:.1f}%"
    else:
        unique_filtered_subjects = df_filtered['subject'].dropna().unique() if not df_filtered.empty else []
        if len(unique_filtered_subjects) > 0:
            comps = [completeness_cache.get(s, {}).get("completeness", 0.0) for s in unique_filtered_subjects if s in completeness_cache]
            if comps:
                avg_comp_str = f"{np.mean(comps):.1f}%"
            else:
                avg_comp_str = "0.0%"
        else:
            if completeness_cache:
                avg_comp = np.mean([v.get("completeness", 0.0) for v in completeness_cache.values()])
                avg_comp_str = f"{avg_comp:.1f}%"
            else:
                avg_comp_str = "0.0%"

    return str(total_materials), avg_fgos_str, gen_share_str, avg_comp_str




@app.callback(
    Output("tab-content", "children"),
    [Input("dashboard-tabs", "value"),
     Input("subject-dropdown", "value"),
     Input("material-type-dropdown", "value"),
     Input("extension-dropdown", "value"),
     Input("origin-dropdown", "value"),
     Input("usage-dropdown", "value"),
     Input("compliance-slider", "value"),
     Input("data-trigger", "data")],
    [State("session-store", "data")]
)
def render_tab_content(active_tab, selected_subject, selected_material_type, selected_extension, selected_origin, selected_usage, selected_compliance_range, trigger, session_data):
    if not session_data.get("logged_in", False):
        return html.Div()

    df = get_data_from_db()
    completeness_cache = load_completeness_cache()
    current_role = session_data.get("role", "Преподаватель")


    if active_tab == "tab-issues" and current_role == "Преподаватель":
        return dbc.Alert(
            "⚠️ Доступ ограничен. Эта вкладка доступна только пользователям с ролью 'Методист' или 'Администратор'. Пожалуйста, войдите в систему с соответствующей ролью.",
            color="danger", style={"marginTop": "20px"}
        )

    if active_tab == "tab-ai-audit" and current_role == "Преподаватель":
        return dbc.Alert(
            "⚠️ Доступ ограничен. Проведение ИИ-аудита полноты тем доступно только с ролью 'Методист' или 'Администратор'. Пожалуйста, войдите в систему с соответствующей ролью.",
            color="danger", style={"marginTop": "20px"}
        )

    if df.empty:
        return html.Div(
            [html.H4("В базе данных нет записей. Пожалуйста, запустите модерацию в основном приложении.")],
            style={"textAlign": "center", "padding": "50px", "color": THEME_TEXT}
        )


    df_filtered = filter_dataframe(
        df, selected_subject, selected_material_type, selected_extension,
        selected_origin, selected_usage, selected_compliance_range
    )

    if df_filtered.empty and active_tab != "tab-ai-audit":
        return html.Div(
            [html.H5("Нет материалов, соответствующих выбранным фильтрам.", style={"color": "#ff5252", "textAlign": "center", "padding": "80px"})],
            style={"color": THEME_TEXT}
        )


    if active_tab == "tab-overview":

        comp_rows = []
        for s_name, data in completeness_cache.items():
            comp_rows.append({
                "Subject": s_name,
                "Completeness": data.get("completeness", 0.0),
                "Existing": data.get("existing_count", 0),
                "Missing": data.get("missing_count", 0)
            })

        fig1 = go.Figure()
        if comp_rows:
            comp_df = pd.DataFrame(comp_rows)
            if selected_subject:
                comp_df = comp_df[comp_df['Subject'] == selected_subject]

            comp_df = comp_df.sort_values(by="Completeness", ascending=True)
            colors = [THEME_ACCENT if s == selected_subject else THEME_ACCENT_ALT for s in comp_df['Subject']]

            fig1.add_trace(go.Bar(
                y=comp_df['Subject'],
                x=comp_df['Completeness'],
                orientation='h',
                marker_color=colors,
                name="Полнота тем (%)",
                text=comp_df['Completeness'].apply(lambda x: f" {x:.1f}%"),
                textposition='outside',
                textfont=dict(color="#ffffff", size=11, family="Inter")
            ))

            avg_comp = np.mean([v.get("completeness", 0.0) for v in completeness_cache.values()])
            fig1.add_vline(
                x=avg_comp,
                line_dash="dash",
                line_color="#ff5252",
                annotation_text=f"Среднее по системе: {avg_comp:.1f}%",
                annotation_position="bottom right",
                annotation_font=dict(color="#ff5252", size=12, family="Inter")
            )

            fig1.update_layout(
                title=dict(text="Заполненность покрытия тем по предметам (относительно среднего)", font=dict(color=THEME_TEXT, size=16, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title=dict(text="Полнота тем, %", font=dict(color="#8b949e")), gridcolor="#21262d", tickfont=dict(color="#8b949e"), range=[0, 110]),
                yaxis=dict(title="", tickfont=dict(color="#c9d1d9"), automargin=True),
                height=500,
                margin=dict(l=20, r=40, t=60, b=40)
            )
        else:
            fig1.update_layout(title="Нет данных по покрытию тем. Запустите ИИ-аудит.", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)


        val_counts = df_filtered['is_generated'].value_counts()
        labels = ["Исходные материалы", "Сгенерировано ИИ"]
        values = [val_counts.get(False, 0), val_counts.get(True, 0)]

        fig2_global = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker=dict(colors=['#58a6ff', '#f9a03f']),
            textinfo='percent+label',
            textfont=dict(color='#ffffff', size=12, family="Inter")
        )])
        fig2_global.update_layout(
            title=dict(text="Доля ИИ-генерации (Сводно)", font=dict(color=THEME_TEXT, size=15, family="Inter")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        subj_gen_counts = df_filtered.groupby(['subject', 'is_generated']).size().unstack(fill_value=0)
        if True not in subj_gen_counts.columns:
            subj_gen_counts[True] = 0
        if False not in subj_gen_counts.columns:
            subj_gen_counts[False] = 0

        subj_gen_counts['total'] = subj_gen_counts[True] + subj_gen_counts[False]
        subj_gen_counts = subj_gen_counts.sort_values(by='total', ascending=True)

        fig2_bar = go.Figure()
        fig2_bar.add_trace(go.Bar(
            y=subj_gen_counts.index,
            x=subj_gen_counts[False],
            name="Исходные материалы",
            orientation='h',
            marker_color='#58a6ff'
        ))
        fig2_bar.add_trace(go.Bar(
            y=subj_gen_counts.index,
            x=subj_gen_counts[True],
            name="Сгенерировано ИИ",
            orientation='h',
            marker_color='#f9a03f'
        ))

        fig2_bar.update_layout(
            title=dict(text="Доля сгенерированных материалов по предметам", font=dict(color=THEME_TEXT, size=15, family="Inter")),
            barmode='stack',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title=dict(text="Количество материалов", font=dict(color="#8b949e")), gridcolor="#21262d", tickfont=dict(color="#8b949e")),
            yaxis=dict(tickfont=dict(color="#c9d1d9"), automargin=True),
            legend=dict(font=dict(color=THEME_TEXT)),
            height=400,
            margin=dict(l=20, r=20, t=50, b=40)
        )

        return html.Div([
            dbc.Row([
                dbc.Col(
                    dbc.Card(dbc.CardBody(dcc.Graph(figure=fig1, config={"displayModeBar": False})), style={"backgroundColor": THEME_CARD_BG, "border": f"1px solid {THEME_BORDER}"}),
                    width=12
                )
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Card(dbc.CardBody(dcc.Graph(figure=fig2_global, config={"displayModeBar": False})), style={"backgroundColor": THEME_CARD_BG, "border": f"1px solid {THEME_BORDER}"}),
                    width=4
                ),
                dbc.Col(
                    dbc.Card(dbc.CardBody(dcc.Graph(figure=fig2_bar, config={"displayModeBar": False})), style={"backgroundColor": THEME_CARD_BG, "border": f"1px solid {THEME_BORDER}"}),
                    width=8
                )
            ], style={"marginTop": "20px"})
        ])


    elif active_tab == "tab-lessons":
        lesson_col = 'material_type' if 'material_type' in df_filtered.columns else 'extension'
        df_filtered[lesson_col] = df_filtered[lesson_col].fillna("Не определен")

        lesson_counts = df_filtered[lesson_col].value_counts()

        fig3_pie = go.Figure(data=[go.Pie(
            labels=lesson_counts.index,
            values=lesson_counts.values,
            hole=.4,
            textinfo='percent+label',
            textfont=dict(color='#ffffff', size=11, family="Inter")
        )])

        fig3_pie.update_layout(
            title=dict(text=f"Типы занятий: {selected_subject or 'Все предметы (Сводно)'}", font=dict(color=THEME_TEXT, size=15, family="Inter")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(color=THEME_TEXT)),
            height=350,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        cat_averages = {}
        if 'compliance_structure' in df_filtered.columns and 'compliance_clarity' in df_filtered.columns:
            cat_averages["Автоматические требования (Структура / Ясность)"] = np.mean([
                df_filtered['compliance_structure'].mean(),
                df_filtered['compliance_clarity'].mean()
            ]) * 20

        if 'legal_compliance' in df_filtered.columns:
            cat_averages["Нормативно-правовые требования"] = df_filtered['legal_compliance'].mean() * 20

        if 'technical_quality' in df_filtered.columns:
            cat_averages["Нормативно-технические требования"] = df_filtered['technical_quality'].mean() * 20

        if 'practice_orientation' in df_filtered.columns:
            cat_averages["Методологические требования (Практика)"] = df_filtered['practice_orientation'].mean() * 20

        if 'compliance_completeness' in df_filtered.columns and 'compliance_terminology' in df_filtered.columns:
            cat_averages["Методические требования (Термины / Полнота)"] = np.mean([
                df_filtered['compliance_completeness'].mean(),
                df_filtered['compliance_terminology'].mean()
            ]) * 20

        fig4_bar = go.Figure()
        if cat_averages:
            categories = list(cat_averages.keys())
            scores = list(cat_averages.values())

            fig4_bar.add_trace(go.Bar(
                x=scores,
                y=categories,
                orientation='h',
                marker_color='#f9a03f',
                text=[f" {s:.1f}%" for s in scores],
                textposition='outside',
                textfont=dict(color="#ffffff", size=11, family="Inter")
            ))

            fig4_bar.update_layout(
                title=dict(text="Обеспеченность требований метод. рекомендаций по категориям", font=dict(color=THEME_TEXT, size=15, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title=dict(text="Уровень обеспеченности, %", font=dict(color="#8b949e")), gridcolor="#21262d", tickfont=dict(color="#8b949e"), range=[0, 110]),
                yaxis=dict(tickfont=dict(color="#c9d1d9"), automargin=True),
                height=350,
                margin=dict(l=20, r=40, t=50, b=40)
            )

        return html.Div([
            dbc.Row([
                dbc.Col(
                    dbc.Card(dbc.CardBody(dcc.Graph(figure=fig3_pie, config={"displayModeBar": False})), style={"backgroundColor": THEME_CARD_BG, "border": f"1px solid {THEME_BORDER}"}),
                    width=5
                ),
                dbc.Col(
                    dbc.Card(dbc.CardBody(dcc.Graph(figure=fig4_bar, config={"displayModeBar": False})), style={"backgroundColor": THEME_CARD_BG, "border": f"1px solid {THEME_BORDER}"}),
                    width=7
                )
            ])
        ])


    elif active_tab == "tab-requirements":
        criteria_cols = {
            'compliance_structure': 'Структура',
            'compliance_clarity': 'Ясность',
            'compliance_terminology': 'Термины',
            'compliance_completeness': 'Полнота',
            'practice_orientation': 'Практика',
            'legal_compliance': 'Правовые',
            'technical_quality': 'Технические'
        }

        existing_cols = [col for col in criteria_cols.keys() if col in df.columns]

        fig5 = go.Figure()
        if existing_cols and selected_subject:
            subj_df = df_filtered

            diff_data = []
            for col in existing_cols:
                name = criteria_cols[col]
                subj_avg = subj_df[col].mean() * 20
                global_avg = df[col].mean() * 20
                diff = subj_avg - global_avg
                diff_data.append({
                    "Критерий": name,
                    "Субъект_Среднее": subj_avg,
                    "Глобальное_Среднее": global_avg,
                    "Отклонение": diff
                })

            diff_df = pd.DataFrame(diff_data).sort_values(by="Отклонение", ascending=True)
            colors = ['#ff5252' if val < 0 else '#4caf50' for val in diff_df['Отклонение']]

            fig5.add_trace(go.Bar(
                x=diff_df['Отклонение'],
                y=diff_df['Критерий'],
                orientation='h',
                marker_color=colors,
                text=diff_df['Отклонение'].apply(lambda x: f" {x:+.1f}%"),
                textposition='outside',
                textfont=dict(color="#ffffff", size=10, family="Inter")
            ))

            fig5.update_layout(
                title=dict(text=f"Отклонение критериев '{selected_subject}' от среднего по системе", font=dict(color=THEME_TEXT, size=15, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title=dict(text="Отклонение (в процентных пунктах)", font=dict(color="#8b949e")), gridcolor="#21262d", tickfont=dict(color="#8b949e")),
                yaxis=dict(tickfont=dict(color="#c9d1d9")),
                height=350,
                margin=dict(l=20, r=40, t=50, b=40)
            )
        else:
            fig5.update_layout(
                title="Выберите конкретный предмет в боковой панели для анализа отклонений",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350
            )

        fig6 = go.Figure()
        if existing_cols:
            original_means = []
            generated_means = []
            names = []

            df_original = df_filtered[df_filtered['is_generated'] == False]
            df_generated = df_filtered[df_filtered['is_generated'] == True]

            for col in existing_cols:
                names.append(criteria_cols[col])
                original_means.append(df_original[col].mean() * 20 if not df_original.empty else 0.0)
                generated_means.append(df_generated[col].mean() * 20 if not df_generated.empty else 0.0)

            fig6.add_trace(go.Bar(
                x=names,
                y=original_means,
                name="Исходные материалы",
                marker_color='#58a6ff'
            ))
            fig6.add_trace(go.Bar(
                x=names,
                y=generated_means,
                name="Сгенерировано ИИ",
                marker_color='#f9a03f'
            ))

            fig6.update_layout(
                title=dict(text="Качество материалов: Исходные vs Сгенерированные ИИ", font=dict(color=THEME_TEXT, size=15, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickfont=dict(color="#c9d1d9")),
                yaxis=dict(title=dict(text="Оценка соответствия, %", font=dict(color="#8b949e")), gridcolor="#21262d", tickfont=dict(color="#8b949e"), range=[0, 110]),
                legend=dict(font=dict(color=THEME_TEXT)),
                barmode='group',
                height=350,
                margin=dict(l=20, r=20, t=50, b=40)
            )

        return html.Div([
            dbc.Row([
                dbc.Col(
                    dbc.Card(dbc.CardBody(dcc.Graph(figure=fig5, config={"displayModeBar": False})), style={"backgroundColor": THEME_CARD_BG, "border": f"1px solid {THEME_BORDER}"}),
                    width=6
                ),
                dbc.Col(
                    dbc.Card(dbc.CardBody(dcc.Graph(figure=fig6, config={"displayModeBar": False})), style={"backgroundColor": THEME_CARD_BG, "border": f"1px solid {THEME_BORDER}"}),
                    width=6
                )
            ])
        ])


    elif active_tab == "tab-ai-audit":
        if not selected_subject:
            return html.Div(
                [html.H5("Пожалуйста, выберите конкретный предмет в боковой панели, чтобы провести проверку покрытия или просмотреть детали.", style={"color": "#8b949e", "textAlign": "center", "padding": "50px"})]
            )

        subj_comp_data = completeness_cache.get(selected_subject, {})
        existing_topics = sorted(list(set(df[df['subject'] == selected_subject]['topic'].dropna().unique())))

        completeness_value = subj_comp_data.get("completeness", None)
        missing_topics = subj_comp_data.get("missing_topics", [])
        explanation = subj_comp_data.get("explanation", "Нет объяснения.")

        card_content = []
        if completeness_value is not None:
            card_content = [
                html.Div([
                    html.H4(f"Результаты аудита для предмета '{selected_subject}'", style={"color": THEME_ACCENT, "fontWeight": "600", "marginBottom": "20px"}),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Метрики покрытия:", style={"fontWeight": "600", "color": THEME_TEXT}),
                            html.P([html.Strong("Заполненность: "), html.Span(f"{completeness_value}%", style={"color": THEME_ACCENT, "fontWeight": "bold", "fontSize": "18px"})]),
                            html.P([html.Strong("Изучено тем: "), f"{subj_comp_data.get('existing_count', 0)} ед."]),
                            html.P([html.Strong("Не хватает тем: "), f"{subj_comp_data.get('missing_count', 0)} ед."]),
                        ], width=4),
                        dbc.Col([
                            html.H5("Существующие темы в базе данных:", style={"fontWeight": "600", "color": THEME_TEXT}),
                            html.Ul([html.Li(t, style={"fontSize": "13px"}) for t in subj_comp_data.get("existing_topics", [])], style={"maxHeight": "150px", "overflowY": "auto"})
                        ], width=4),
                        dbc.Col([
                            html.H5("Недостающие темы (ИИ):", style={"fontWeight": "600", "color": "#ff5252"}),
                            html.Ul([html.Li(t, style={"fontSize": "13px"}) for t in missing_topics], style={"maxHeight": "150px", "overflowY": "auto"})
                        ], width=4),
                    ]),
                    html.Hr(style={"borderColor": THEME_BORDER}),
                    html.H5("Методическое заключение ИИ:", style={"fontWeight": "600", "color": THEME_TEXT}),
                    html.P(explanation, style={"fontStyle": "italic", "fontSize": "14px", "lineHeight": "1.5"})
                ])
            ]
        else:
            card_content = [
                html.Div([
                    html.H4(f"Анализ предмета '{selected_subject}'", style={"color": THEME_TEXT}),
                    html.P("Данные о полноте покрытия этого предмета отсутствуют в кэше. Пожалуйста, запустите ИИ-аудит."),
                    html.P([html.Strong("Найдено существующих тем в базе данных: "), f"{len(existing_topics)} тем."])
                ])
            ]

        admin_action = html.Div([
            html.Hr(style={"borderColor": THEME_BORDER}),
            dbc.Button(
                "🚀 Запустить ИИ-аудит полноты тем (OpenRouter)",
                id="btn-run-ai-audit",
                color="danger",
                style={"fontWeight": "600", "marginRight": "10px"}
            ),
            dcc.Loading(
                id="loading-audit",
                type="dot",
                children=html.Div(id="audit-result-text", style={"marginTop": "10px", "color": THEME_ACCENT})
            )
        ]) if current_role == "Администратор" else html.Div([
            html.Hr(style={"borderColor": THEME_BORDER}),
            html.Small("⚠️ Только администраторы имеют право запускать повторный ИИ-анализ через OpenRouter.")
        ])

        return dbc.Card(
            dbc.CardBody(card_content + [admin_action]),
            style={"backgroundColor": THEME_CARD_BG, "border": f"1px solid {THEME_BORDER}", "color": THEME_TEXT}
        )


    elif active_tab == "tab-issues":
        issues_rows = []
        for _, row in df_filtered.iterrows():
            f_name = row['file_name']
            subj = row['subject'] or "—"
            rate = f"{row['compliance_rate'] * 100:.0f}%" if 'compliance_rate' in row else "0%"
            allowed = "ДА" if row.get('usage_allowed') == 'yes' else "НЕТ"

            iss_list = []
            rec_list = []
            try:
                iss_list = json.loads(row['issues']) if 'issues' in row and row['issues'] else []
            except:
                if 'issues' in row and row['issues']:
                    iss_list = [row['issues']]

            try:
                rec_list = json.loads(row['recommendations']) if 'recommendations' in row and row['recommendations'] else []
            except:
                if 'recommendations' in row and row['recommendations']:
                    rec_list = [row['recommendations']]

            iss_text = "; ".join(iss_list) if isinstance(iss_list, list) else str(iss_list)
            rec_text = "; ".join(rec_list) if isinstance(rec_list, list) else str(rec_list)

            issues_rows.append({
                "Имя файла": f_name,
                "Предмет": subj,
                "Оценка ФГОС": rate,
                "Допущен": allowed,
                "Замечания": iss_text if iss_text else "Замечаний нет",
                "Методические рекомендации": rec_text if rec_text else "Рекомендаций нет"
            })

        issues_df = pd.DataFrame(issues_rows)

        table = dash_table.DataTable(
            data=issues_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in issues_df.columns],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#161b22",
                "color": THEME_ACCENT,
                "fontWeight": "bold",
                "border": f"1px solid {THEME_BORDER}"
            },
            style_cell={
                "backgroundColor": "#0d1117",
                "color": THEME_TEXT,
                "border": f"1px solid {THEME_BORDER}",
                "padding": "10px",
                "fontSize": "13px",
                "textAlign": "left",
                "whiteSpace": "normal",
                "height": "auto",
            },
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{Допущен} = "НЕТ"',
                        'column_id': 'Допущен'
                    },
                    'color': '#ff5252',
                    'fontWeight': 'bold'
                },
                {
                    'if': {
                        'filter_query': '{Допущен} = "ДА"',
                        'column_id': 'Допущен'
                    },
                    'color': '#4caf50',
                    'fontWeight': 'bold'
                }
            ]
        )

        download_section = html.Div([
            dbc.Button(
                "📥 Скачать отчет по замечаниям (CSV)",
                id="btn-download-issues",
                color="info",
                style={"marginTop": "20px", "fontWeight": "600"}
            ),
            dcc.Download(id="download-issues-csv")
        ])

        return html.Div([
            html.H4("Журнал замечаний и методических рекомендаций", style={"color": THEME_ACCENT, "marginBottom": "20px"}),
            table,
            download_section
        ])

    return html.Div("Вкладка в разработке")



@app.callback(
    Output("audit-result-text", "children"),
    [Input("btn-run-ai-audit", "n_clicks")],
    [State("subject-dropdown", "value"),
     State("session-store", "data")]
)
def run_ai_completeness_audit(n_clicks, selected_subject, session_data):
    if not n_clicks or not selected_subject:
        return ""

    current_role = session_data.get("role", "Преподаватель")
    if current_role != "Администратор":
        return "❌ Ошибка доступа: Анализ может запускать только Администратор."

    df = get_data_from_db()
    if df.empty:
        return "❌ База данных пуста."

    existing_topics = sorted(list(set(df[df['subject'] == selected_subject]['topic'].dropna().unique())))
    if not existing_topics:
        return f"❌ Для предмета '{selected_subject}' не найдено существующих тем в базе данных."

    audit_res = check_completeness_via_ai(selected_subject, existing_topics)
    if audit_res:
        cache = load_completeness_cache()
        cache[selected_subject] = audit_res
        save_completeness_cache(cache)
        return f"✓ Успешно завершено! Заполненность темы: {audit_res['completeness']}%."
    else:
        return "❌ Не удалось выполнить запрос к OpenRouter. Проверьте соединение или API ключ."



@app.callback(
    Output("download-issues-csv", "data"),
    [Input("btn-download-issues", "n_clicks")],
    [State("subject-dropdown", "value"),
     State("material-type-dropdown", "value"),
     State("extension-dropdown", "value"),
     State("origin-dropdown", "value"),
     State("usage-dropdown", "value"),
     State("compliance-slider", "value")]
)
def download_issues_report(n_clicks, selected_subject, selected_material_type, selected_extension, selected_origin, selected_usage, selected_compliance_range):
    if not n_clicks:
        return None

    df = get_data_from_db()
    if df.empty:
        return None

    df_filtered = filter_dataframe(
        df, selected_subject, selected_material_type, selected_extension,
        selected_origin, selected_usage, selected_compliance_range
    )

    issues_rows = []
    for _, row in df_filtered.iterrows():
        f_name = row['file_name']
        subj = row['subject'] or "—"
        rate = f"{row['compliance_rate'] * 100:.0f}%" if 'compliance_rate' in row else "0%"
        allowed = "ДА" if row.get('usage_allowed') == 'yes' else "НЕТ"

        iss_list = []
        rec_list = []
        try:
            iss_list = json.loads(row['issues']) if 'issues' in row and row['issues'] else []
        except:
            if 'issues' in row and row['issues']:
                iss_list = [row['issues']]

        try:
            rec_list = json.loads(row['recommendations']) if 'recommendations' in row and row['recommendations'] else []
        except:
            if 'recommendations' in row and row['recommendations']:
                rec_list = [row['recommendations']]

        iss_text = "; ".join(iss_list) if isinstance(iss_list, list) else str(iss_list)
        rec_text = "; ".join(rec_list) if isinstance(rec_list, list) else str(rec_list)

        issues_rows.append({
            "Имя файла": f_name,
            "Предмет": subj,
            "Оценка соответствия ФГОС": rate,
            "Допуск к использованию": allowed,
            "Критические замечания": iss_text if iss_text else "Замечаний нет",
            "Методические рекомендации": rec_text if rec_text else "Рекомендаций нет"
        })

    res_df = pd.DataFrame(issues_rows)
    return dcc.send_data_frame(res_df.to_csv, "sapere_issues_report.csv", index=False, encoding="utf-8-sig")



if __name__ == '__main__':

    app.run(host='127.0.0.1', port=8050, debug=True)
