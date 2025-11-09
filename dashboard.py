import streamlit as st
import requests
import pandas as pd
import altair as alt
import json

API_URL = "http://127.0.0.1:8000"

st.title("ML API Dashboard")

# Получение списка классов моделей
def get_model_classes():
    r = requests.get(f"{API_URL}/get_model_classes")
    if r.ok:
        return json.loads(r.text)['model_classes']
    else:
        st.error("Ошибка получения списка моделей")
        return []

# Получение списка созданных моделей и их гиперпараметров (если есть такой эндпоинт, иначе храните локально)
# Для примера возьмём локальное хранение id и гиперпараметры пока
model_classes = get_model_classes()

st.sidebar.header("Выбор операции")

action = st.sidebar.selectbox("Выберите действие", [
    "Получить список доступных классов моделей",
    "Создать модель",
    "Обучить модель",
    "Создать и обучить модель",  # new action
    "Получить список моделей",  # new action
    "Предсказать",
    "Удалить модель",
    "Просмотр состояния сервиса"
])
# Локальное хранение для выбора id пока (реализуйте запросы к API для списка моделей и параметров, если есть)
model_ids = []

if action == "Получить список доступных классов моделей":
    st.header("Список классов моделей")
    r = requests.get(f"{API_URL}/get_model_classes")
    if r.ok:
        model_classes = json.loads(r.text)['model_classes']
        if model_classes:
            st.write("Доступные классы моделей:")
            for cls in model_classes:
                st.write(f"- {cls}")
                print(f"- {cls}")
        else:
            st.write("Список классов моделей пуст")
    else:
        st.error("Ошибка при получении классов моделей")

elif action == "Создать модель":
    st.header("Создание модели")

    model_type = st.selectbox("Выберите тип модели", model_classes)
    st.write("Введите гиперпараметры (в формате JSON):")
    hyperparams_str = st.text_area("hyperparameters", '{"learning_rate": 0.1, "n_estimators": 100, "num_leaves": 31}')

    if st.button("Создать"):
        try:
            hyperparams = eval(hyperparams_str)  # лучше использовать безопасный JSON парсер
            data = {"model_type": model_type, "hyperparameters": hyperparams}
            r = requests.post(f"{API_URL}/create_model", json=data)
            if r.ok:
                st.success(f"Модель создана: {r.json()}")
                model_ids.append(r.json().get('model_id'))
            else:
                st.error(f"Ошибка при создании модели: {r.text}")
        except Exception as e:
            st.error(f"Некорректный формат гиперпараметров: {e}")

elif action == "Обучить модель":
    st.header("Обучение модели")

    model_id = st.number_input("Введите ID модели", min_value=0, step=1)

    uploaded_X = st.file_uploader("Загрузите CSV с X (train)", type=["csv"])
    uploaded_Y = st.file_uploader("Загрузите CSV с Y (train)", type=["csv"])

    if st.button("Обучить") and uploaded_X and uploaded_Y:
        X = pd.read_csv(uploaded_X)
        Y = pd.read_csv(uploaded_Y)
        data = {
            "model_id": model_id,
            "X_data": X.to_json(orient='records'),
            "Y_data": Y.to_json(orient='records')
        }
        r = requests.post(f"{API_URL}/train_model", json=data)
        if r.ok:
            st.success(f"Обучение завершено: {r.json()}")
        else:
            st.error(f"Ошибка обучения: {r.text}")

elif action == "Создать и обучить модель":
    st.header("Создать и обучить модель")
    model_type = st.selectbox("Выберите тип модели", model_classes)
    st.write("Введите гиперпараметры (в формате JSON):")
    hyperparams_str = st.text_area("hyperparameters", '{"learning_rate": 0.1, "n_estimators": 100, "num_leaves": 31}')
    uploaded_X = st.file_uploader("Загрузите CSV с X (train)", type=["csv"])
    uploaded_Y = st.file_uploader("Загрузите CSV с Y (train)", type=["csv"])
    if st.button("Создать и обучить") and uploaded_X and uploaded_Y:
        try:
            hyperparams = eval(hyperparams_str)  # recommend safe json parsing
            X = pd.read_csv(uploaded_X)
            Y = pd.read_csv(uploaded_Y)
            data = {
                "model_type": model_type,
                "hyperparameters": hyperparams,
                "X_data": X.to_json(orient='records'),
                "Y_data": Y.to_json(orient='records')
            }
            r = requests.post(f"{API_URL}/train_new_model", json=data)
            if r.ok:
                st.success(f"Модель создана и обучена: {r.json()}")
            else:
                st.error(f"Ошибка при создании и обучении модели: {r.text}")
        except Exception as e:
            st.error(f"Некорректный формат гиперпараметров или данных: {e}")

elif action == "Предсказать":
    st.header("Предсказание")

    model_id = st.number_input("Введите ID модели", min_value=0, step=1)

    uploaded_X = st.file_uploader("Загрузите CSV с X для предсказания", type=["csv"])

    if st.button("Предсказать") and uploaded_X:
        X = pd.read_csv(uploaded_X)
        data = {
            "model_id": model_id,
            "X_data": X.to_json(orient='records')
        }
        r = requests.post(f"{API_URL}/predict", json=data)
        if r.ok:
            pred = r.json().get("prediction", [])
            st.success("Предсказание получено")
            st.write(pred)

            # Визуализация предсказаний
            df_pred = pd.DataFrame({"prediction": pred})
            chart = alt.Chart(df_pred.reset_index()).mark_line().encode(
                x="index",
                y="prediction"
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.error(f"Ошибка предсказания: {r.text}")

elif action == "Удалить модель":
    st.header("Удаление модели")

    model_id = st.number_input("Введите ID модели для удаления", min_value=0, step=1)

    if st.button("Удалить"):
        r = requests.delete(f"{API_URL}/delete_model/{model_id}")
        if r.ok:
            st.success(f"Удалено: {r.json()}")
        else:
            st.error(f"Ошибка удаления: {r.text}")

elif action == "Просмотр состояния сервиса":
    st.header("Статус сервиса")

    r = requests.get(f"{API_URL}/healthcheck")
    if r.ok:
        info = r.json()
        st.success(f"Статус: {info.get('status')}")
        st.write(f"Количество моделей: {info.get('model_count')}")
    else:
        st.error("Не удалось получить статус сервиса")

elif action == "Получить список моделей":
    st.header("Список моделей")
    r = requests.get(f"{API_URL}/get_models")
    if r.ok:
        models_info = r.json().get("models_ids_hyperparameters", [])
        if models_info:
            for model_id, hyperparams in models_info:
                st.write(f"ID: {model_id} | Гиперпараметры: {hyperparams}")
        else:
            st.write("Нет сохранённых моделей")
    else:
        st.error("Ошибка получения списка моделей")