import gradio as gr
from recomenation import process_query, skills_un
# Создание интерфейса вручную
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 Система рекомендации вакансий")

    with gr.Row():
        with gr.Column():
            query = gr.Textbox(label="Наименование вакансии", placeholder="Например: аналитик данных python")
            min_salary = gr.Number(label="Минимальная зарплата (необязательно)", value=None)
            experience = gr.Dropdown(
                label="Опыт работы",
                choices=["Любой", "Нет опыта", "От 1 года до 3 лет", "От 3 до 6 лет"]
            )
            skills = gr.Dropdown(
                label="Ключевые навыки",
                choices=skills_un(),
                multiselect=True
            )
            num_vacancies = gr.Dropdown(
                label="Количество вакансий",
                choices=[5, 10, 20, 50, 100, 150]
            )
            submit_btn = gr.Button("Поиск")

    gr.Examples(
        examples=[
            ["аналитик данных"],
            ["python разработчик"],
            ["project manager"]
        ],
        inputs=[query],
        label="Примеры для быстрого поиска"
    )
    results = gr.HTML()
    
    submit_btn.click(fn=process_query, inputs=[query, min_salary, experience, skills, num_vacancies], outputs=results)

demo.launch(debug=True)