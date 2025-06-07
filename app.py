#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from lazypredict.Supervised import LazyClassifier

import shap
import lime
import lime.lime_tabular

# Настройка стиля графиков
sns.set(style="whitegrid", font_scale=1.1)

# Функции
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def preprocess(df):
    # Преобразование типов и заполнение пропусков
    for col in ['HHAUTO_N', 'HHPERS', 'KAFSTV', 'KVERTTIJD', 'HHBRUTOINK2_w5', 'N_KIND']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include='object').columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df.drop_duplicates(inplace=True)
    # Кодирование и создание признаков
    df_enc = pd.get_dummies(df, drop_first=True)
    df_enc['HAS_CAR'] = (df_enc.get('HHAUTO_N', 0) > 0).astype(int)
    df_enc['HAS_CHILDREN'] = (df_enc.get('N_KIND', 0) > 0).astype(int)
    df_enc['KAFSTV_LOG'] = np.log1p(df_enc.get('KAFSTV', 0))
    df_enc['INCOME_NORM'] = StandardScaler().fit_transform(df_enc[['HHBRUTOINK2_w5']])
    return df_enc

@st.cache_data
def split_data(df, target='KHVM'):
    le = LabelEncoder()
    y = le.fit_transform(df[target])
    X = df[[c for c in df.columns if c != target]]
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

@st.cache_data
def train_base_models(X_train, y_train):
    # Обучение трех моделей
    cat = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, random_state=42, verbose=0)
    cat.fit(X_train, y_train)
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                        use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X_train, y_train)
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, random_state=42)
    lgbm.fit(X_train, y_train)
    return {'CatBoost': cat, 'XGBoost': xgb, 'LightGBM': lgbm}

@st.cache_data
def evaluate_models(_models, X_test, y_test):
    metrics = {}
    for name, model in _models.items():
        preds = model.predict(X_test)
        metrics[name] = (accuracy_score(y_test, preds), f1_score(y_test, preds, average='macro'))
    return metrics

# Основная функция приложения

def main():
    st.title("ML Coursework Streamlit Demo")
    uploaded = st.file_uploader("Загрузите CSV с данными", type="csv")
    if not uploaded:
        st.info("Пожалуйста, загрузите CSV для продолжения.")
        return

    df = load_data(uploaded)
    st.subheader("Исходные данные")
    st.dataframe(df.head())

    if st.checkbox("Показать Pandas Profiling отчет"):
        profile = ProfileReport(df, title="Pandas Profiling", minimal=True)
        html = profile.to_html()
        components.html(html, height=600, scrolling=True)

    df_proc = preprocess(df)
    st.subheader("Распределение целевой переменной")
    st.bar_chart(df_proc['KHVM'].value_counts())

    X_train, X_test, y_train, y_test = split_data(df_proc)
    st.write(f"Размер train: {X_train.shape}, test: {X_test.shape}")

    models = train_base_models(X_train, y_train)
    metrics = evaluate_models(models, X_test, y_test)
    st.subheader("Метрики базовых моделей")
    st.write(metrics)

    # Ансамбль
    ensemble = VotingClassifier(estimators=[(n, m) for n, m in models.items()],
                                voting='soft', n_jobs=-1)
    ensemble.fit(X_train, y_train)
    ens_preds = ensemble.predict(X_test)
    ens_metrics = {'Ensemble': (accuracy_score(y_test, ens_preds),
                                 f1_score(y_test, ens_preds, average='macro'))}
    st.subheader("Ансамбль (soft voting)")
    st.write(ens_metrics)

    # AutoML
    if st.checkbox("Запустить AutoML (LazyPredict)"):
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        auto_models, _ = clf.fit(X_train, X_test, y_train, y_test)
        st.subheader("Топ-5 моделей AutoML по F1 Score")
        st.dataframe(auto_models.sort_values('F1 Score', ascending=False).head(5))

    # SHAP
    if st.checkbox("Показать SHAP summary для LightGBM"):
        explainer = shap.TreeExplainer(models['LightGBM'])
        shap_values = explainer.shap_values(X_train)
        fig = shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        st.pyplot(fig)

    # LIME
    if st.checkbox("Показать LIME объяснения для трех примеров"):
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns.tolist(),
            class_names=[str(c) for c in sorted(set(y_train))],
            mode='classification'
        )
        for idx in [0,1,2]:
            exp = explainer_lime.explain_instance(
                data_row=X_test.iloc[idx].values,
                predict_fn=models['LightGBM'].predict_proba,
                num_features=10
            )
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
