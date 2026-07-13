# ML/AI Engineer Foundations

Небольшой практический набор базовых задач для подготовки к junior-позициям ML/AI/LLM Engineer. Материалы организованы как Python-проект: каждая тема вынесена в отдельный модуль, все эксперименты воспроизводимы, а основные сценарии покрыты тестами.

## Что входит в проект

| Модуль | Задача | Основные навыки |
|---|---|---|
| `regression.py` | Прогноз числового показателя | train/test split, Pipeline, масштабирование, RidgeCV, MAE, RMSE, R² |
| `classification.py` | Выявление злокачественных опухолей | Logistic Regression, дисбаланс классов, F1, ROC-AUC |
| `ensembles.py` | Сравнение деревьев и ансамблей | Decision Tree, Random Forest, Gradient Boosting, cross-validation |
| `dimensionality.py` | Сжатие признакового пространства | StandardScaler, PCA, explained variance, классификация |
| `text_retrieval.py` | Поиск релевантного текстового фрагмента | TF-IDF, N-граммы, cosine similarity, базовая retrieval-логика для RAG |

## Почему выбраны эти темы

Для junior ML/AI Engineer важно понимать полный базовый цикл: подготовить данные, разделить выборку, собрать pipeline, обучить baseline, выбрать метрики и проверить обобщающую способность. Для LLM-направления добавлена retrieval-задача: она показывает механику индексации и поиска контекста до перехода к embeddings и векторным базам.

## Структура

```text
ml-ai-engineer-foundations/
├── src/
│   ├── regression.py
│   ├── classification.py
│   ├── ensembles.py
│   ├── dimensionality.py
│   ├── text_retrieval.py
│   └── run_all.py
├── tests/
│   └── test_foundations.py
├── pyproject.toml
├── requirements.txt
└── SOURCE.md
```

## Установка и запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_all
```

Проверка проекта:

```bash
pytest
```

## Данные

Регрессия использует встроенный датасет Diabetes, классификация и ансамбли — Breast Cancer Wisconsin, PCA — Wine. Они загружаются из `scikit-learn`, поэтому отдельные файлы с данными не нужны. Тексты для retrieval-примера написаны специально для этого проекта.

## Что демонстрирует проект

- воспроизводимое разбиение данных через `random_state`;
- отсутствие утечки при масштабировании благодаря `Pipeline`;
- подбор регуляризации внутри `RidgeCV`;
- выбор метрик под тип задачи;
- кросс-валидацию для устойчивого сравнения моделей;
- снижение размерности без обработки test-данных до обучения;
- простую архитектуру поиска контекста для LLM-приложения.

## Возможные улучшения

- добавить конфигурацию экспериментов и логирование в MLflow;
- сохранять обученные модели через `joblib`;
- добавить подбор гиперпараметров через `RandomizedSearchCV`;
- заменить TF-IDF на sentence embeddings и подключить векторную базу;
- обернуть retrieval-модуль в FastAPI;
- добавить Docker и CI-проверку тестов.

