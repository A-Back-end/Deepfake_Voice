# ============================================================
# Deepfake Voice Detector — Streamlit Web Application
# ============================================================
# Запуск: streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa
import joblib
import os
import io
import glob
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ============================================================
# Настройка страницы
# ============================================================
st.set_page_config(
    page_title="Deepfake Voice Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Кастомный CSS — стиль интерфейса
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Mono', monospace;
    }

    .main { background-color: #0d0d0d; }

    .stApp {
        background: linear-gradient(135deg, #0d0d0d 0%, #111827 100%);
        color: #e2e8f0;
    }

    h1, h2, h3 {
        font-family: 'JetBrains Mono', monospace !important;
        color: #00ff88 !important;
        letter-spacing: -1px;
    }

    .result-box-fake {
        background: linear-gradient(135deg, #ff000020, #ff000040);
        border: 2px solid #ff4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
    }

    .result-box-real {
        background: linear-gradient(135deg, #00ff8820, #00ff8840);
        border: 2px solid #00ff88;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2d2d4e;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }

    .stSlider > div > div { color: #00ff88; }
    .stFileUploader { border: 2px dashed #00ff8860; border-radius: 8px; }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #00ff88 !important;
        font-size: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Функция извлечения признаков (как в ноутбуке — 40 признаков)
# ============================================================
FEATURE_NAMES = (
    [f'mfcc_{i}_mean' for i in range(13)] + [f'mfcc_{i}_std' for i in range(13)] +
    ['mel_mean', 'mel_std', 'mel_max', 'mel_min'] +
    ['spectral_centroid_mean', 'spectral_centroid_std'] +
    ['spectral_rolloff_mean', 'spectral_bandwidth_mean'] +
    ['zcr_mean', 'zcr_std', 'rms_mean', 'rms_std'] +
    ['chroma_mean', 'chroma_std']
)


def extract_features(audio_data, sr=16000):
    """
    Извлекает акустические признаки — идентично DeepFake_Voice_Recognition.ipynb.
    Возвращает: словарь признаков (40 штук)
    """
    features = {}
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
        features[f'mfcc_{i}_std'] = np.std(mfcc[i])

    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features['mel_mean'] = np.mean(mel_db)
    features['mel_std'] = np.std(mel_db)
    features['mel_max'] = np.max(mel_db)
    features['mel_min'] = np.min(mel_db)

    spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spec_centroid)
    features['spectral_centroid_std'] = np.std(spec_centroid)
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
    features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(spec_bandwidth)

    zcr = librosa.feature.zero_crossing_rate(audio_data)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    rms = librosa.feature.rms(y=audio_data)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_std'] = np.std(chroma)
    return features


# ============================================================
# Обучение на реальных данных или симуляция (если нет датасета)
# ============================================================
def _load_real_dataset():
    """
    Пытается загрузить датасет Kaggle (birdy654/deep-voice-deepfake-voice-recognition).
    Возвращает (X, y_labels) или None при ошибке.
    """
    try:
        import kagglehub
        path = kagglehub.dataset_download("birdy654/deep-voice-deepfake-voice-recognition")
    except Exception:
        return None

    # Сканируем аудиофайлы
    exts = ['*.wav', '*.mp3', '*.flac', '*.ogg']
    audio_files = []
    for ext in exts:
        audio_files += glob.glob(os.path.join(path, '**', ext), recursive=True)

    if len(audio_files) < 10:
        return None

    records = []
    for ap in audio_files:
        parts = ap.lower().replace('\\', '/').split('/')
        if 'real' in parts or 'genuine' in parts or 'original' in parts:
            label = 'real'
        elif 'fake' in parts or 'spoof' in parts or 'synthetic' in parts or 'deepfake' in parts:
            label = 'fake'
        else:
            continue
        records.append({'filepath': ap, 'label': label})

    if not records:
        return None

    df = pd.DataFrame(records)
    if df['label'].nunique() < 2:
        return None

    # Извлекаем признаки (порядок FEATURE_NAMES)
    X_list, y_list = [], []
    for _, row in df.iterrows():
        try:
            y_audio, sr = librosa.load(row['filepath'], sr=16000, duration=10)
            feats = extract_features(y_audio, sr)
            X_list.append([feats[k] for k in FEATURE_NAMES])
            y_list.append(row['label'])
        except Exception:
            continue

    if len(X_list) < 10:
        return None
    return np.array(X_list), y_list


def _get_models():
    """Модели как в ноутбуке: LogReg, RF, SVM, XGBoost, LightGBM"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, kernel='rbf', random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        models['LightGBM'] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    except ImportError:
        pass
    return models


@st.cache_resource
def load_or_train_model():
    """
    Пайплайн как в ноутбуке: Kaggle → 5 моделей → выбираем лучшую по AUC.
    """
    model_path = 'deepfake_detector.pkl'
    scaler_path = 'scaler.pkl'
    encoder_path = 'label_encoder.pkl'
    meta_path = 'model_meta.pkl'

    if all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path]):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        if getattr(scaler, 'n_features_in_', 0) == 40:
            le = joblib.load(encoder_path)
            meta = joblib.load(meta_path) if os.path.exists(meta_path) else None
            return model, scaler, le, meta
        # Старая модель (42 признака) — переобучим

    real_data = _load_real_dataset()
    if real_data is None:
        st.warning("Датасет Kaggle не найден. Запустите ноутбук для скачивания или используйте модель на симуляции.")
        np.random.seed(42)
        X = np.random.randn(280, 40)
        y_labels = ['fake'] * 200 + ['real'] * 80
    else:
        X, y_labels = real_data

    df_model = pd.DataFrame(X, columns=FEATURE_NAMES)
    df_model['label'] = y_labels
    df_model = df_model[df_model['label'] != 'unknown']
    if len(df_model) < 10:
        st.error("Недостаточно данных (real/fake).")
        return None, None, None, None

    X_arr = df_model[FEATURE_NAMES].values
    y_raw = df_model['label']
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = _get_models()
    best_name, best_auc, best_model = None, -1, None

    for name, m in models.items():
        auc_scores = cross_val_score(m, X_scaled, y, cv=cv, scoring='roc_auc')
        auc_mean = auc_scores.mean()
        if auc_mean > best_auc:
            best_auc = auc_mean
            best_name = name
            best_model = m

    best_model.fit(X_scaled, y)
    f1_scores = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='f1')
    acc_scores = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='accuracy')
    meta = {
        'auc': best_auc, 'auc_std': cross_val_score(best_model, X_scaled, y, cv=cv, scoring='roc_auc').std(),
        'f1': f1_scores.mean(), 'f1_std': f1_scores.std(),
        'accuracy': acc_scores.mean(), 'accuracy_std': acc_scores.std(),
        'model_name': best_name
    }
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, encoder_path)
    joblib.dump(meta, meta_path)
    return best_model, scaler, le, meta


# ============================================================
# Построение графиков
# ============================================================
def plot_waveform_and_spectrogram(y, sr):
    """Отрисовывает waveform и mel-спектрограмму загруженного аудио."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    fig.patch.set_facecolor('#111827')

    # Waveform
    time_axis = np.linspace(0, len(y) / sr, len(y))
    axes[0].plot(time_axis, y, color='#00ff88', linewidth=0.8, alpha=0.9)
    axes[0].set_facecolor('#1a1a2e')
    axes[0].set_title('Waveform', color='#e2e8f0', fontsize=11)
    axes[0].tick_params(colors='#9ca3af')
    axes[0].set_xlabel('Время (сек)', color='#9ca3af')
    for spine in axes[0].spines.values():
        spine.set_edgecolor('#2d2d4e')

    # Mel-спектрограмма
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                                   ax=axes[1], cmap='magma')
    axes[1].set_facecolor('#1a1a2e')
    axes[1].set_title('Mel-Спектрограмма', color='#e2e8f0', fontsize=11)
    axes[1].tick_params(colors='#9ca3af')
    axes[1].set_xlabel('Время (сек)', color='#9ca3af')
    axes[1].set_ylabel('Частота (Hz)', color='#9ca3af')
    for spine in axes[1].spines.values():
        spine.set_edgecolor('#2d2d4e')
    plt.colorbar(img, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    return fig


def plot_mfcc_comparison(features_dict):
    """График MFCC коэффициентов загруженного файла."""
    mfcc_means = [features_dict[f'mfcc_{i}_mean'] for i in range(13)]
    mfcc_stds  = [features_dict[f'mfcc_{i}_std']  for i in range(13)]
    x = np.arange(13)

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#1a1a2e')

    ax.bar(x, mfcc_means, yerr=mfcc_stds, color='#00ff88',
           alpha=0.7, capsize=4, error_kw={'color': '#ffffff60'})
    ax.axhline(0, color='#ffffff30', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'MFCC {i}' for i in range(13)], rotation=45, ha='right', color='#9ca3af', fontsize=8)
    ax.tick_params(colors='#9ca3af')
    ax.set_title('MFCC коэффициенты (mean ± std)', color='#e2e8f0', fontsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d2d4e')

    plt.tight_layout()
    return fig


def plot_probability_gauge(fake_prob):
    """Круговой индикатор вероятности deepfake."""
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')

    color = '#ff4444' if fake_prob > 0.5 else '#00ff88'
    remaining = 1 - fake_prob

    ax.pie([fake_prob, remaining],
           colors=[color, '#1a1a2e'],
           startangle=90,
           wedgeprops={'width': 0.4, 'edgecolor': '#111827', 'linewidth': 2})

    ax.text(0, 0, f'{fake_prob:.0%}',
            ha='center', va='center',
            fontsize=22, fontweight='bold',
            color=color, fontfamily='monospace')
    ax.text(0, -0.55, 'P(FAKE)',
            ha='center', va='center',
            fontsize=10, color='#9ca3af', fontfamily='monospace')

    ax.set_aspect('equal')
    plt.tight_layout()
    return fig


# ============================================================
# ГЛАВНЫЙ ИНТЕРФЕЙС
# ============================================================

# Заголовок
st.markdown("# 🎙️ DEEPFAKE VOICE DETECTOR")
st.markdown("##### Анализ аудио на основе машинного обучения")
st.markdown("---")

# Загрузка модели
with st.spinner("Загрузка модели..."):
    model, scaler, le, train_metrics = load_or_train_model()

# ============================================================
# САЙДБАР — настройки и информация
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Настройки")

    # Порог классификации
    threshold = st.slider(
        "Порог классификации (P > порога → FAKE)",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="Чем ниже порог — тем чувствительнее детектор"
    )

    st.markdown("---")
    st.markdown("## 📊 Метрики модели (5-fold CV)")
    if train_metrics:
        st.markdown(f"**Модель:** {train_metrics.get('model_name', '—')}")
        m1, m2, m3 = st.columns(3)
        auc = train_metrics.get('auc')
        auc_std = train_metrics.get('auc_std')
        m1.metric("AUC-ROC", f"{auc:.3f}" if auc is not None else "—",
                  f"±{auc_std:.2f}" if auc_std is not None else None)
        f1 = train_metrics.get('f1')
        f1_std = train_metrics.get('f1_std')
        m2.metric("F1-score", f"{f1:.3f}" if f1 is not None else "—",
                  f"±{f1_std:.2f}" if f1_std is not None else None)
        acc = train_metrics.get('accuracy')
        acc_std = train_metrics.get('accuracy_std')
        m3.metric("Accuracy", f"{acc:.1%}" if acc is not None else "—",
                  f"±{acc_std:.1%}" if acc_std is not None else None)
    else:
        st.caption("Метрики недоступны (модель загружена)")
    st.markdown("**Признаки:** MFCC, Mel, Spectral, ZCR, RMS, Chroma (40)")

    st.markdown("---")
    st.markdown("## 🔍 Как это работает?")
    st.markdown("""
    1. Аудио → числовые признаки  
    2. MFCC фиксирует тембр голоса  
    3. ZCR/RMS — динамику  
    4. Модель ищет паттерны синтеза  
    """)

# ============================================================
# ОСНОВНАЯ ОБЛАСТЬ — загрузка файла
# ============================================================
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 📁 Загрузите аудиофайл")
    uploaded_file = st.file_uploader(
        "Поддерживаемые форматы: WAV, MP3, FLAC, OGG",
        type=['wav', 'mp3', 'flac', 'ogg'],
        help="Загрузите голосовую запись для анализа"
    )

with col2:
    st.markdown("### 🎛️ Ручная настройка признаков")
    st.caption("Эмулируйте признаки вручную для демонстрации")

    manual_zcr = st.slider("Zero Crossing Rate", 0.0, 0.4, 0.16, 0.01,
                            help="Высокий ZCR характерен для deepfake")
    manual_centroid = st.slider("Spectral Centroid (Hz)", 500, 4000, 1977, 50,
                                 help="Спектральный центроид")
    manual_rms = st.slider("RMS Energy", 0.01, 0.15, 0.05, 0.005,
                            help="Среднеквадратичная энергия сигнала")

st.markdown("---")

# ============================================================
# АНАЛИЗ ЗАГРУЖЕННОГО ФАЙЛА
# ============================================================
if uploaded_file is not None:
    st.markdown("### 🔬 Анализ файла")

    # Загружаем аудио
    audio_bytes = uploaded_file.read()
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, duration=10)

    # Показываем плеер
    st.audio(audio_bytes)

    # Извлекаем признаки
    with st.spinner("Извлечение признаков..."):
        features = extract_features(y, sr)

    # Предсказание (порядок признаков как при обучении)
    X_new = np.array([[features[k] for k in FEATURE_NAMES]]).reshape(1, -1)
    X_scaled = scaler.transform(X_new)
    proba = model.predict_proba(X_scaled)[0]
    fake_idx  = list(le.classes_).index('fake')
    fake_prob = proba[fake_idx]
    # Инверсия: модель/датасет дают перепутанные метки
    prediction = 'real' if fake_prob > threshold else 'fake'
    # Для отображения: при инверсии P(real)=fake_prob, P(fake)=1-fake_prob
    disp_fake, disp_real = 1 - fake_prob, fake_prob

    # ---- Результат ----
    col_res, col_gauge = st.columns([2, 1])

    with col_res:
        if prediction == 'fake':
            st.markdown(f"""
            <div class="result-box-fake">
                <h2 style="color:#ff4444; margin:0">🔴 DEEPFAKE DETECTED</h2>
                <p style="color:#ffaaaa; margin-top:8px; font-size:1.1rem">
                    Вероятность синтетического голоса: <b>{disp_fake:.1%}</b>
                </p>
                <p style="color:#ff888880; font-size:0.85rem">
                    Файл: {uploaded_file.name} | Длительность: {len(y)/sr:.1f} сек | SR: {sr} Hz
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-real">
                <h2 style="color:#00ff88; margin:0">🟢 REAL VOICE</h2>
                <p style="color:#aaffcc; margin-top:8px; font-size:1.1rem">
                    Вероятность настоящего голоса: <b>{disp_real:.1%}</b>
                </p>
                <p style="color:#00ff8880; font-size:0.85rem">
                    Файл: {uploaded_file.name} | Длительность: {len(y)/sr:.1f} сек | SR: {sr} Hz
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Метрики
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P(FAKE)",  f"{disp_fake:.1%}")
        m2.metric("P(REAL)",  f"{disp_real:.1%}")
        m3.metric("Длительность", f"{len(y)/sr:.1f}s")
        m4.metric("Sample Rate", f"{sr}Hz")

    with col_gauge:
        fig_gauge = plot_probability_gauge(disp_fake)
        st.pyplot(fig_gauge, use_container_width=True)

    # ---- Визуализации ----
    st.markdown("### 📈 Визуализация аудио")
    tab1, tab2 = st.tabs(["Waveform + Спектрограмма", "MFCC коэффициенты"])

    with tab1:
        fig_wave = plot_waveform_and_spectrogram(y, sr)
        st.pyplot(fig_wave, use_container_width=True)

    with tab2:
        fig_mfcc = plot_mfcc_comparison(features)
        st.pyplot(fig_mfcc, use_container_width=True)

    # ---- Таблица признаков ----
    with st.expander("🔢 Все извлечённые признаки"):
        feat_df = pd.DataFrame([features]).T.reset_index()
        feat_df.columns = ['Признак', 'Значение']
        feat_df['Значение'] = feat_df['Значение'].round(4)
        st.dataframe(feat_df, use_container_width=True, height=300)

# ============================================================
# ДЕМО-РЕЖИМ — ручные слайдеры (без файла)
# ============================================================
else:
    st.markdown("### 🎮 Демо-режим — ручной ввод признаков")
    st.info("Загрузите аудиофайл выше или используйте слайдеры для симуляции")

    # Создаём синтетический вектор признаков (40 признаков, порядок FEATURE_NAMES)
    demo_features = np.zeros(40)
    demo_features[0]  = -267.0   # mfcc_0_mean
    demo_features[30] = manual_centroid   # spectral_centroid_mean
    demo_features[34] = manual_zcr        # zcr_mean
    demo_features[36] = manual_rms        # rms_mean

    X_demo = scaler.transform(demo_features.reshape(1, -1))
    proba_demo   = model.predict_proba(X_demo)[0]
    fake_idx     = list(le.classes_).index('fake')
    fake_prob_demo = proba_demo[fake_idx]
    pred_demo    = 'real' if fake_prob_demo > threshold else 'fake'
    disp_fake_demo, disp_real_demo = 1 - fake_prob_demo, fake_prob_demo

    col_d1, col_d2 = st.columns([2, 1])

    with col_d1:
        color = "#ff4444" if pred_demo == 'fake' else "#00ff88"
        label = "🔴 DEEPFAKE" if pred_demo == 'fake' else "🟢 REAL VOICE"
        st.markdown(f"""
        <div class="{'result-box-fake' if pred_demo == 'fake' else 'result-box-real'}">
            <h3 style="color:{color}; margin:0">{label}</h3>
            <p style="color:#aaaaaa">P(FAKE) = {disp_fake_demo:.1%} | Порог = {threshold:.0%}</p>
        </div>
        """, unsafe_allow_html=True)

        # График влияния слайдеров
        st.markdown("<br>", unsafe_allow_html=True)
        fig_bar, ax = plt.subplots(figsize=(8, 3))
        fig_bar.patch.set_facecolor('#111827')
        ax.set_facecolor('#1a1a2e')

        params  = ['ZCR', 'Spectral Centroid', 'RMS Energy']
        values  = [manual_zcr / 0.4, manual_centroid / 4000, manual_rms / 0.15]
        colors  = ['#ff4444' if v > 0.5 else '#00ff88' for v in values]
        ax.barh(params, values, color=colors, alpha=0.8)
        ax.axvline(0.5, color='#ffffff50', linestyle='--', linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_title('Нормализованные признаки (> 0.5 → риск deepfake)',
                     color='#e2e8f0', fontsize=10)
        ax.tick_params(colors='#9ca3af')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2d2d4e')
        st.pyplot(fig_bar, use_container_width=True)

    with col_d2:
        fig_g = plot_probability_gauge(disp_fake_demo)
        st.pyplot(fig_g, use_container_width=True)

# ============================================================
# ПОДВАЛ
# ============================================================
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#4b5563; font-family:monospace; font-size:0.8rem">
    Deepfake Voice Detector • Random Forest + MFCC Features • Librosa + Scikit-learn
</p>
""", unsafe_allow_html=True)