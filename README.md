📖 О проекте

Данное репозиторий содержит реализацию курсовой работы по дисциплине Машинное обучение. Целью проекта является разработка и сравнительный анализ моделей многоклассовой классификации вида транспорта на основе демографических и поведенческих признаков, а также развёртывание интерактивного веб‑интерфейса с помощью Streamlit.

Основные этапы:

Разведочный анализ данных (EDA)

Предобработка и создание признаков (Feature Engineering)

Обучение моделей CatBoost, XGBoost, LightGBM

Ансамблирование (VotingClassifier)

AutoML‑оценка с LazyPredict

Интеграция SHAP и LIME для интерпретации

Веб‑приложение на Streamlit

🚀 Установка и запуск

Клонируйте репозиторий:
```
git clone https://github.com/username/MLCourseWorkVariant6.git
cd MLCourseWorkVariant6
```
Установите зависимости:
```
pip install -r requirements.txt
```
Запустите приложение:
```
streamlit run app.py
```
Откройте в браузере: http://localhost:8501.

📁 Структура каталогов
```
├── app.py                  # Основной скрипт Streamlit
├── requirements.txt        # Список зависимостей
├── Dataset_2_DATA.csv      # Исходные данные
├── README.md               # Описание проекта
└── report.md               # Текст курсовой работы
```


🛠️ Использованные технологии

Python 3.10+

Streamlit

Pandas, NumPy

Scikit-learn, CatBoost, XGBoost, LightGBM

SHAP, LIME

LazyPredict

📚 Ссылки

Streamlit documentation: https://docs.streamlit.io
