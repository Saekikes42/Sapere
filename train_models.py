from config import DATABASE_URL, OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN
import os
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def prepare_all_models():
    os.makedirs('models', exist_ok=True)

    engine = create_engine(DATABASE_URL)
    query = "SELECT file_name, text_content, compliance_rate, label_parallel_id, label_sequential_id, label_difficulty FROM opit_files"
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print("Database error:", e)
        return

    df['text_content'] = df['text_content'].fillna('')
    X = df['text_content']

    targets = {
        'compliance_rate': '100',
        'label_parallel_id': 'P0',
        'label_sequential_id': 'S0',
        'label_difficulty': 'Средняя'
    }

    for target_col, default_val in targets.items():
        y = df[target_col].fillna(default_val)
        y = y.astype(str)

        if len(y.unique()) <= 1:
            print(f"Warning: Only 1 class for {target_col}. Appending dummy data for training.")
            X = pd.concat([X, pd.Series(['dummy text 1', 'dummy text 2'])])
            if target_col == 'label_difficulty':
                y = pd.concat([y, pd.Series(['Низкая', 'Высокая'])])
            elif target_col == 'compliance_rate':
                y = pd.concat([y, pd.Series(['0', '50'])])
            else:
                y = pd.concat([y, pd.Series(['P1', 'P2'])])

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('clf', LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'))
        ])

        print(f"Training {target_col}...")
        pipeline.fit(X, y)


        save_name = 'label_conformance' if target_col == 'compliance_rate' else target_col

        save_name = save_name.replace('_id', '')

        joblib_path = f"models/{save_name}_latest.joblib"
        joblib.dump(pipeline, joblib_path)
        print(f"Saved {joblib_path}")

if __name__ == "__main__":
    prepare_all_models()
