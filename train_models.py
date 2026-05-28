import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor 
from sklearn.metrics import r2_score
import joblib
import pickle

# 1. Загрузка
df = pd.read_csv('diamonds_clean.csv')
print(f"Исходный размер: {df.shape}")

# Уменьшаем датасет до 15 000 строк 
df = df.sample(n=15000, random_state=42)
print(f"Уменьшенный размер: {df.shape}")

X = df.drop('price', axis=1)
y = df['price']

numeric = ['carat', 'depth', 'table', 'volume']
categorical = ['cut', 'color', 'clarity']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical)
])

X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Модели с экономными параметрами
lr = LinearRegression()
xgb = XGBRegressor(random_state=42, n_jobs=1, n_estimators=80)
cat = CatBoostRegressor(verbose=0, random_state=42, iterations=80)
rf = RandomForestRegressor(random_state=42, n_estimators=30, max_depth=10, n_jobs=1)
stack = StackingRegressor([('rf', rf), ('xgb', xgb), ('cat', cat)], final_estimator=LinearRegression(), cv=3)
nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42, early_stopping=True)

# Обучение
lr.fit(X_train, y_train)
xgb.fit(X_train, y_train)
cat.fit(X_train, y_train)
rf.fit(X_train, y_train)
stack.fit(X_train, y_train)
nn.fit(X_train, y_train)

# Сохранение
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
joblib.dump(lr, 'linear_regression.joblib')
joblib.dump(xgb, 'xgboost.joblib')
cat.save_model('catboost_model.cbm')
joblib.dump(rf, 'random_forest.joblib')
joblib.dump(stack, 'stacking_model.joblib')
joblib.dump(nn, 'neural_net.joblib')

print("\n Все модели сохранены")

# Оценка R²
print("\nR² на тесте:")
for name, model in [('LR', lr), ('XGB', xgb), ('CatBoost', cat), ('RF', rf), ('Stack', stack), ('NN', nn)]:
    pred = model.predict(X_test)
    print(f"{name:10} R² = {r2_score(y_test, pred):.4f}")