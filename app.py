import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pickle
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(page_title="РГР_Предсказание_цены_алмазов", layout="wide")

# ---------------------- СТРАНИЦА 1: О РАЗРАБОТЧИКЕ ----------------------
def page_about():
    st.title(" О разработчике")
    col1, col2 = st.columns([1, 2])
    with col1:
        # Замените на путь к вашему фото или уберите
        st.image("фото_отчёт.jpg", caption="Фото студента")
    with col2:
        st.markdown("""
        **ФИО:** Шакенова Жанслу Жанатовна  
                    

        **Группа:** МО-241

                    
        **Тема РГР:** Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных  
                    

        **Цель:** Предсказание цены алмазов на основе их характеристик
        """)

# ---------------------- СТРАНИЦА 2: О ДАТАСЕТЕ ----------------------
def page_data_info():
    st.title("Информация о наборе данных")
    st.markdown("""
    **Датасет:** Diamonds (набор данных о бриллиантах)

    **Предметная область:** Ювелирные изделия – определение рыночной стоимости алмаза по физическим и качественным характеристикам.

    **Признаки (8 колонок):**
    - `carat` – вес алмаза в каратах (числовой)
    - `cut` – качество огранки (категории: Fair, Good, Very Good, Premium, Ideal)
    - `color` – цвет (от J (худший) до D (лучший))
    - `clarity` – чистота (от I1 (худший) до IF (лучший))
    - `depth` – глубина в процентах (числовой)
    - `table` – ширина стола в процентах (числовой)
    - `volume` – объём (вычислен как x*y*z)
    - `price` – цена в долларах США (целевая переменная)

    **Предобработка:**
    - Удалены выбросы по глубине, столу и объёму.
    - Категориальные признаки закодированы one-hot.
    - Числовые признаки масштабированы (StandardScaler).
    """)

# ---------------------- СТРАНИЦА 3: ВИЗУАЛИЗАЦИИ ----------------------
def page_visuals():
    st.title(" Анализ зависимостей в данных")
    df = pd.read_csv('diamonds_clean.csv')
    
    st.subheader("1. Распределение цены")
    fig, ax = plt.subplots()
    ax.hist(df['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title('Распределение цены алмаза')
    st.pyplot(fig)
    
    st.subheader("2. Зависимость цены от веса (carat)")
    fig, ax = plt.subplots()
    ax.scatter(df['carat'], df['price'], alpha=0.3, s=1)
    ax.set_xlabel('carat')
    ax.set_ylabel('price')
    ax.set_title('Price vs Carat')
    st.pyplot(fig)
    
    st.subheader("3. Зависимость цены от объема (volume)")
    fig, ax = plt.subplots()
    ax.scatter(df['volume'], df['price'], alpha=0.3, s=1)
    ax.set_xlabel('volume (mm³)')
    ax.set_ylabel('price')
    ax.set_title('Price vs Volume')
    st.pyplot(fig)
    
    st.subheader("4. Boxplot цены по огранке (cut) – упрощенно")
    # Упростим: покажем среднюю цену по cut
    cut_means = df.groupby('cut')['price'].mean().sort_values()
    fig, ax = plt.subplots()
    cut_means.plot(kind='bar', ax=ax, color='lightgreen')
    ax.set_title('Средняя цена по Cut')
    ax.set_ylabel('Средняя цена ($)')
    st.pyplot(fig)

# ---------------------- СТРАНИЦА 4: ИНФЕРЕНС ----------------------
# Загрузка моделей
@st.cache_resource
def load_models():
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    lr = joblib.load('linear_regression.joblib')
    xgb = joblib.load('xgboost.joblib')
    cat = CatBoostRegressor()
    cat.load_model('catboost_model.cbm')
    rf = joblib.load('random_forest.joblib')
    stack = joblib.load('stacking_model.joblib')
    nn = joblib.load('neural_net.joblib')
    models = {
        "Linear Regression": lr,
        "XGBoost": xgb,
        "CatBoost": cat,
        "Random Forest": rf,
        "Stacking (best)": stack,
        "Neural Network": nn
    }
    return preprocessor, models

def page_inference():
    st.title(" Предсказание цены алмаза")
    st.markdown("Введите характеристики алмаза или загрузите CSV-файл с несколькими примерами.")

    preprocessor, models = load_models()

    # Чёткое сопоставление числа -> текст (стандартный порядок для diamonds)
    # Если в вашем датасете порядок другой – замените списки
    cut_labels = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color_labels = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    clarity_labels = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

    # Обратные словари: текст -> число
    cut_map = {label: i for i, label in enumerate(cut_labels)}
    color_map = {label: i for i, label in enumerate(color_labels)}
    clarity_map = {label: i for i, label in enumerate(clarity_labels)}

    model_name = st.sidebar.selectbox("Выберите модель", list(models.keys()))
    model = models[model_name]

    input_method = st.radio("Способ ввода:", ["Ручной ввод", "Загрузить CSV"])

    if input_method == "Ручной ввод":
        col1, col2 = st.columns(2)
        with col1:
            carat = st.number_input("Вес (карат)", min_value=0.2, max_value=5.0, value=0.7, step=0.1)
            cut_text = st.selectbox("Огранка (cut)", cut_labels)
            color_text = st.selectbox("Цвет (color)", color_labels)
            clarity_text = st.selectbox("Чистота (clarity)", clarity_labels)
        with col2:
            depth = st.number_input("Глубина (%)", min_value=50.0, max_value=70.0, value=61.5, step=0.5)
            table = st.number_input("Стол (%)", min_value=55.0, max_value=70.0, value=57.0, step=0.5)
            volume = st.number_input("Объём (мм³)", min_value=20.0, max_value=500.0, value=100.0, step=10.0)

        if st.button("Рассчитать цену"):
            # Преобразуем текст в число
            cut_code = cut_map[cut_text]
            color_code = color_map[color_text]
            clarity_code = clarity_map[clarity_text]

            input_df = pd.DataFrame({
                'carat': [carat],
                'cut': [cut_code],
                'color': [color_code],
                'clarity': [clarity_code],
                'depth': [depth],
                'table': [table],
                'volume': [volume]
            })
            processed = preprocessor.transform(input_df)
            pred = model.predict(processed)
            price = pred[0]

            metric_map = {
                "Linear Regression": "0.8942",
                "XGBoost": "0.9467",
                "CatBoost": "0.9464",
                "Random Forest": "0.9386",
                "Stacking (best)": "0.9478",
                "Neural Network": "0.9093"
            }
            st.success(f"**Предсказанная цена алмаза:** ${price:,.2f} USD")
            st.info(f"Модель: {model_name}, R² = {metric_map.get(model_name, '?')}")

    else:  # Загрузка CSV
        uploaded_file = st.file_uploader("Загрузите CSV-файл с колонками: carat, cut, color, clarity, depth, table, volume", type=["csv"])
        if uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)
            # Если колонки текстовые, преобразуем в числа
            if df_input['cut'].dtype == object:
                df_input['cut'] = df_input['cut'].map(cut_map)
                df_input['color'] = df_input['color'].map(color_map)
                df_input['clarity'] = df_input['clarity'].map(clarity_map)
                if df_input[['cut', 'color', 'clarity']].isnull().any().any():
                    st.error("В загруженном файле неизвестные категории. Используйте стандартные названия.")
                    return
            # Если уже числа, оставляем как есть
            required = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'volume']
            if all(col in df_input.columns for col in required):
                processed = preprocessor.transform(df_input)
                predictions = model.predict(processed)
                df_input['predicted_price'] = predictions
                st.dataframe(df_input.style.format({'predicted_price': '${:,.2f}'}))
                csv = df_input.to_csv(index=False).encode('utf-8')
                st.download_button("Скачать результаты CSV", csv, "predictions.csv", "text/csv")
            else:
                st.error(f"Файл должен содержать колонки: {required}")



# ---------------------- НАВИГАЦИЯ ----------------------
def main():
    st.sidebar.title("Навигация")
    pages = {
        "1. О разработчике": page_about,
        "2. Описание датасета": page_data_info,
        "3. Визуализации": page_visuals,
        "4. Предсказание цены": page_inference
    }
    choice = st.sidebar.radio("Перейти", list(pages.keys()))
    pages[choice]()

if __name__ == "__main__":
    main()