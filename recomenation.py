import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from work_with_dataset import clean_text, craate_model, create_embeddings, remove_html_tags,load_df

#Загружаем данные
df = load_df()

model = craate_model()

embeddings = create_embeddings(model=model, df=df)
# Функция рекомендаций

def recommend_vacancies(query, skills=None, top_k=50, min_salary=None, experience=None):

    query_clean = clean_text(query)
    if not query_clean:
        return pd.DataFrame({"Сообщение": ["Не удалось обработать запрос"]})

    query_embedding = model.encode([query_clean])
    sim_scores = cosine_similarity(query_embedding, embeddings)[0]

    results = df.copy()
    results["similarity"] = sim_scores

    results["key_skills"] = results["key_skills"].fillna("").astype(str).apply(lambda x: x.strip().split("\n"))
    # Фильтрация
    if min_salary:
        results["salary_from"] = pd.to_numeric(results["salary_from"], errors="coerce")
        results = results[results["salary_from"] >= min_salary]
    if experience:
        results = results[results["experience_name"] == experience]
    if skills:
        # Очищаем поле description от HTML-разметки
        results["cleaned_description"] = results["description"].apply(remove_html_tags)
        regex_pattern = rf'\b(?:{("|".join(map(re.escape, skills)))})\b'
        # Предполагаем, что results["key_skills"] — это список строк (список навыков)
        results = results[results["key_skills"].apply(
            lambda vacancy_skills: any(skill in vacancy_skills for skill in skills)
        ) | (results["description"].str.contains(regex_pattern, case=False))]
    return results.sort_values("similarity", ascending=False).head(top_k)

# Предобработка зарплаты
def format_salary(row):
    salary_from = row["salary_from"]
    salary_to = row["salary_to"]
    currency = row.get("salary_currency", "₽")  # если нет — по умолчанию рубли

    # Обработка NaN
    if pd.isna(salary_from) and pd.isna(salary_to):
        return "Не указано"
    elif pd.notna(salary_from) and pd.notna(salary_to):
        return f"{int(salary_from)} - {int(salary_to)} {currency}"
    elif pd.notna(salary_from):
        return f"от {int(salary_from)} {currency}"
    elif pd.notna(salary_to):
        return f"до {int(salary_to)} {currency}"

# Поиск уникальных скиллов в датасете 
def skills_un():
    unique_skills = (
    df['key_skills']              # Берем нужный столбец
      .dropna()                   # Исключаем пропуски (NaN)
      .str.split('\n')            # Разбиваем каждую строку на список по переносу строки
      .explode()                  # Преобразуем списки в отдельные строки
      .str.strip()                # Чистим от лишних пробелов
      .unique()                   # Оставляем только уникальные значения
      .tolist()                   # Преобразуем в список
    )
    return unique_skills

# Предобработка контактов
def format_contacts(row):
    salary = row['contacts']
    if pd.isna(salary):
        return "Не указано"
    else:
        return salary

# Функция для вывода найденных вакансий по занным данным
def search_jobs(query,  min_salary=None, experience=None,skills=None, num_vacancies=None):
    '''
    query = наименование вакансии,  
    min_salary = минимальная зарплата,
    experience = опыт работы,
    skills = скиллы,
    num_vacancies=количество вакансий для вывода
    '''
    if experience == 'Любой':
        experience = None
    results = recommend_vacancies(query, skills=skills, min_salary=min_salary, experience=experience, top_k=num_vacancies)
        # Генерируем HTML-таблицу
    df = results[["name", "description", "key_skills","experience_name", "salary_from", "salary_to", "contacts", "employer_name","alternate_url"]]
    df["name"] = df.apply(
        lambda row: f'<a href="{row["alternate_url"]}" target="_blank">{row["name"]}</a>', axis=1
    )
    df["Salary"] = df.apply(format_salary, axis=1)
    df["_Contacts_"] = df.apply(format_contacts, axis=1)
    
    df_clear = df[["name", "description","experience_name", "key_skills", "Salary", "_Contacts_", "employer_name"]]

    df_clear["key_skills"] = df["key_skills"].apply(
    lambda skills: "<ul>" + "".join(f"<li>{s}</li>" for s in skills) + "</ul>" if isinstance(skills, list) else ""
)

    html = """
    <style>
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #000000; }
        a { color: #007bff; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
    """
    df_clear = df_clear.rename(columns={
    "name": "Название вакансии",
    "description": "Описание",
    "key_skills": "Ключевые навыки",
    "_Contacts_": "Контакты",
    "employer_name":"Наименование огранизации",
    "Salary":"Зарплата",
    "alternate_url":"ССылка",
    "experience_name":"Опыт"
    })
    html += df_clear.to_html(escape=False, index=False)
    return html

# Функция предобработки ошибки, если не введено название вакансии
def process_query(query,min_salary, experience, skills, num_vacancies):
    if not query:
        raise gr.Error("Необходимо ввести наименование вакансии!")
    else:
        return search_jobs(query,min_salary, experience, skills, num_vacancies) 
    
