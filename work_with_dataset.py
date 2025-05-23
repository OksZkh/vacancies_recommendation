import pandas as pd
import re
from pymorphy3 import MorphAnalyzer
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from html import unescape

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Скачиваем стоп-слова...")
    nltk.download('stopwords')

morph = MorphAnalyzer()
stopwords_ru = set(stopwords.words('russian'))

def clean_text(text):
    """Очистка и лемматизация текста"""
    if not isinstance(text, str):
        return ""

    # Удаляем HTML-теги
    #text = re.sub(r'<[^>]+>', '', text)
    # Удаляем спецсимволы, оставляем буквы и цифры
    words = re.findall(r'[а-яёa-z0-9]+', text.lower(), flags=re.IGNORECASE)
    # Лемматизация с обработкой исключений
    cleaned_words = []
    for word in words:
        if word not in stopwords_ru and len(word) > 2:
            try:
                parsed = morph.parse(word)
                if parsed:
                    cleaned_words.append(parsed[0].normal_form)
            except:
                cleaned_words.append(word)
    return " ".join(cleaned_words)

def load_df():
    try:
        df = pd.read_csv("vacancies_2020.csv", nrows=10000)  # Ограничение для MVP
        print(f"Загружено {len(df)} вакансий")
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        # Создаем тестовые данные
        data = {
            "name": ["Аналитик данных", "Программист Python", "Менеджер проектов"],
            "description": [
                "Анализ больших данных, построение отчетов, SQL, Python",
                "Разработка веб-приложений на Python и Django",
                "Управление IT проектами, Agile, Scrum"
            ],
            "key_skills": ["SQL\nPython\nTableau", "Python\nDjango\nPostgreSQL", "Agile\nScrum\nУправление"]
        }
        df = pd.DataFrame(data)

    # Предобработка текста
    df["full_text"] = (
        df["name"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df.get("key_skills", "").fillna("")
    )
    df["text_clean"] = df["full_text"].apply(clean_text)
    df["name_clean"] = df["name"].apply(clean_text)
    return df

# Векторизация
def craate_model():
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return model
def create_embeddings(model, df):
    embeddings = model.encode(df["name_clean"].tolist())
    return embeddings

# Функция для удаления HTML-тегов
def remove_html_tags(text):
    """Удаляет HTML-теги и нормализует спецсимволы"""
    # Удаляем HTML-теги с помощью регулярного выражения
    clean_text = re.sub('<.*?>', '', text)
    # Восстанавливаем спецсимволы HTML
    return unescape(clean_text)

