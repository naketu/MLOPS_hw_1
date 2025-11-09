# fastapi import
from fastapi import FastAPI, HTTPException

# models import
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error, roc_auc_score

# other dependancies
from pydantic import BaseModel
from typing import Dict
import json
import pandas as pd
import logging

# setting up a logger
logger = logging.getLogger("ml_api_logger")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("ml_api.log", mode="a")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# request types
class ModelRequest(BaseModel):
    """
        Запрос на создание модели с гиперпараметрами.
    """
    model_type: str
    hyperparameters: Dict[str, int| float | str | None]

class TrainRequest(BaseModel):
    """
        Запрос на обучение модели с ID модели и данными:
        X (регрессоры), Y (зависимая пременная).
    """
    model_id: float
    X_data: str
    Y_data: str

class CreateAndTrainRequest(BaseModel):
    """
        Запрос на создание и обучение модели с гиперпараметрами и данными:
        X (регрессоры), Y (зависимая пременная).
    """
    model_type: str
    hyperparameters: Dict[str, int| float | str | None]
    X_data: str
    Y_data: str

class PredictRequest(BaseModel):
    """
        Запрос на получение предсказаний модели по ID модели, 
        на данных X (регрессоры).
    """
    model_id: float
    X_data: str

# basic containers
next_model_id = 0
models = {}
model_hyperparameters = {}

# API functions
app = FastAPI()

# 1.1 create a model record with hyperparameters
@app.post("/create_model")
async def create_model(request: ModelRequest):
    """
    Создание модели с указанными гиперпараметрами.
    Возвращает ID созданной модели.
    """
    global next_model_id

    logger.info(f"Создание модели типа {request.model_type} с параметрами: {request.hyperparameters}")

    try:
        model_hyperparameters[next_model_id] = request
        
        if request.model_type == "LGBMRegressor":
            models[next_model_id] = lgb.LGBMRegressor(**request.hyperparameters)
        elif request.model_type == "LGBMClassifier":
            models[next_model_id] = lgb.LGBMClassifier(**request.hyperparameters)
        else:
            error_message = "Создание модели невозможно, "
            error_message += "так как был выбран неподдерживаемый класс модели.\n"
            error_message += "Доступные классы можно посмотреть с помошью запроса "
            error_message += ".../get_model_classes"

            logger.error(f"Создание модели завершилось с ошибкой:\n {error_message}")
            raise HTTPException(status_code=500, detail=str(error_message))
        
        next_model_id += 1

        return {"model_id": next_model_id - 1,
                "model_class": request.model_type,
                "hyperparameters": request.hyperparameters}
    
    except Exception as e:
        logger.error(f"Создание модели завершилось с ошибкой: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 1.2 train existing model
@app.post("/train_model")
async def train_model(request: TrainRequest):
    """
    Обучение заранее созданной модели.
    Возвращает ID созданной модели, метрику качества.
    """
    logger.info(f"Обучение модели {request.model_id}")

    try:
        model_id = request.model_id
        dataset_X = pd.read_json(request.X_data)
        dataset_Y = pd.read_json(request.Y_data)

        model_instance = models[model_id]
        model_instance.fit(dataset_X, dataset_Y)

        y_prediction = model_instance.predict(dataset_X)

        if type(model_instance) == lgb.LGBMRegressor:
            metric = 'MAPE'
            value = mean_absolute_percentage_error(dataset_Y, y_prediction)

        elif type(model_instance) == lgb.LGBMClassifier:
            metric = 'ROC AUC'
            value = roc_auc_score(dataset_Y, y_prediction)

        return {"model_id": model_id, 
                "status": "trained", 
                "train metric": f"{metric}: {value}"}
    
    except Exception as e:
        logger.error(f"Обучение модели {request.model_id} завершилось с ошибкой: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 1.3 train new model
@app.post("/train_new_model")
async def train_model(request: CreateAndTrainRequest):
    """
    Создание модели с указанными гиперпараметрами и ее обучение.
    На вход требует тип модели, гиперпараметры, регрессоры и зависимую переменную.
    Возвращает ID созданной модели, метрику качества.
    """

    global next_model_id

    logger.info(f"Создание и обучение модели ID {next_model_id}")
    logger.info(f"Создание модели типа {request.model_type} с параметрами: {request.hyperparameters}")
    
    try:
        # save model

        model_hyperparameters[next_model_id] = {
            'model_type': request.model_type
        }
        
        if request.model_type == "LGBMRegressor":
            models[next_model_id] = lgb.LGBMRegressor(**request.hyperparameters)
        elif request.model_type == "LGBMClassifier":
            models[next_model_id] = lgb.LGBMClassifier(**request.hyperparameters)
        else:
            error_message = "Создание модели невозможно, "
            error_message += "так как был выбран неподдерживаемый класс модели.\n"
            error_message += "Доступные классы можно посмотреть с помошью запроса "
            error_message += ".../get_model_classes"

            logger.error(f"Создание модели завершилось с ошибкой:\n {error_message}")
            raise HTTPException(status_code=500, detail=str(error_message))
        
        model_id = next_model_id
        next_model_id += 1

    except Exception as e:
        logger.error(f"Создание модели завершилось с ошибкой: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"Обучение модели {model_id}")

    try:
        # train model
        dataset_X = pd.read_json(request.X_data)
        dataset_Y = pd.read_json(request.Y_data)

        model_instance = models[model_id]
        model_instance.fit(dataset_X, dataset_Y)

        y_prediction = model_instance.predict(dataset_X)

        if type(model_instance) == lgb.LGBMRegressor:
            metric = 'MAPE'
            value = mean_absolute_percentage_error(dataset_Y, y_prediction)

        elif type(model_instance) == lgb.LGBMClassifier:
            metric = 'ROC AUC'
            value = roc_auc_score(dataset_Y, y_prediction)

        return {"model_id": model_id, "status": "trained", "train metric": f"{metric}: {value}",}
    
    except Exception as e:
        logger.error(f"Обучение модели завершилось с ошибкой: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 2. get all available models
@app.get("/get_model_classes")
async def get_model_classes():
    """
    Перечисление доступных классов модели
    """
    logger.info(f"Запрос на перечисление классов модели")
    return {"model_classes": ["LGBMRegressor", "LGBMClassifier"]}

# additional function - get all saved models and their types
@app.get("/get_models")
async def get_models():
    """
    Перечисление всех сохраненных моделей и гиперпараметров
    """
    logger.info(f"Запрос на перечисление моделей")
    model_ids = list(models.keys())
    model_types = [model_hyperparameters[x] for x in model_ids]
    return {"models_ids_hyperparameters": list(zip(model_ids, model_types))}

# 3. get model prediction
@app.post("/predict")
async def get_model_prediction(request: PredictRequest):
    """
    Выполняет предсказание модели по ID модели на данных X.
    Возвращает предсказания.
    """
    logger.info(f"Выполняется инференс модели")

    try:
        model_id = request.model_id
        X_data = pd.read_json(request.X_data)

        model_instance = models[model_id]
        prediction = model_instance.predict(X_data)

        return {"model_id": request.model_id, "prediction": prediction.tolist()}
    
    except Exception as e:
        logger.error(f"Инференс модели модели завершился с ошибкой: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 4. deleting a model
@app.delete("/delete_model/{model_id}")
async def delete_model(model_id: int):
    """
    Удаляет модель и её гиперпараметры по ID.
    Возвращает статус удаления.
    """
    global models, model_hyperparameters

    logger.info(f"Выполняется удаление модели {model_id}")
    
    if model_id not in models:
        return {"status": "failed", "result": f"Модели с id {model_id} не существует"}
    
    models.pop(model_id)
    model_hyperparameters.pop(model_id)
    
    return {"status": "success", "result": f"Model {model_id} and its hyperparameters were deleted"}

# 5. Check the APP status
@app.get("/healthcheck")
async def healthcheck():
    """
    Эндпоинт для проверки состояния сервиса.
    Возвращает статус и количество зарегистрированных моделей.
    """
    try:
        model_count = len(models)
        logger.info(f"Healthcheck запрошен. Всего моделей: {model_count}")
        return {"status": "ok", "model_count": model_count}
    except Exception as e:
        logger.error(f"Ошибка в healthcheck: {str(e)}")
        return {"status": "error", "message": str(e)}