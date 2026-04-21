import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# =========================
# Load Model Files
# =========================
@st.cache_resource
def load_model():
    with open("model/ecg_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, scaler, encoder

model, scaler, encoder = load_model()

# =========================
# Feature Extraction (same as Flask)
# =========================
def extract_features(signal):
    signal = np.array(signal, dtype=float)
    feats = []
    feats.append(np.mean(signal))
    feats.append(np.std(signal))
    feats.append(np.max(signal))
    feats.append(np.min(signal))
    feats.append(np.max(signal) - np.min(signal))
    feats.append(np.sqrt(np.mean(signal ** 2)))
    feats.extend(np.percentile(signal, [10, 25, 50, 75, 90]).tolist())

    diff1 = np.diff(signal)
    feats.append(np.mean(np.abs(diff1)))
    feats.append(np.std(diff1))

    diff2 = np.diff(diff1)
    feats.append(np.mean(np.abs(diff2)))

    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal))
    total_power = np.sum(fft_vals ** 2) + 1e-12
    feats.append(np.sum(freqs * fft_vals ** 2) / total_power)
    feats.extend(fft_vals[:10].tolist())

    n_segs = 5
    seg_len = len(signal) // n_segs
    for i in range(n_segs):
        seg = signal[i * seg_len:(i + 1) * seg_len]
        feats.append(np.std(seg))
        feats.append(np.max(seg) - np.min(seg))

    crossings = np.where(np.diff(np.sign(signal)))[0]
    feats.append(len(crossings))

    peaks = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))[0]
    feats.append(len(peaks))

    if len(peaks) > 1:
        feats.append(np.std(np.diff(peaks)))
    else:
        feats.append(0.0)

    return np.array(feats, dtype=float)

# =========================
# Preprocessing
# =========================
def preprocess_signal(signal_values, window_size=187):
    signal = np.array(signal_values, dtype=float)

    if len(signal) > window_size:
        signal = signal[:window_size]
    elif len(signal) < window_size:
        signal = np.pad(signal, (0, window_size - len(signal)), mode='edge')

    mean = signal.mean()
    std = signal.std() + 1e-8
    signal = (signal - mean) / std

    return signal

# =========================
# UI
# =========================
st.set_page_config(page_title="PulseSense-AI", layout="centered")

st.title("🫀 PulseSense-AI: ECG Arrhythmia Classifier")
st.write("Upload ECG CSV file (187 values) to predict arrhythmia.")

# =========================
# File Upload
# =========================
file = st.file_uploader("📂 Upload ECG CSV", type=["csv"])

# =========================
# Prediction
# =========================
if file:
    try:
        df = pd.read_csv(file, header=None)
        signal_values = df.values.flatten().tolist()

        st.subheader("📊 ECG Signal Preview")
        st.write(f"Total Samples: {len(signal_values)}")

        # Plot raw signal
        fig, ax = plt.subplots()
        ax.plot(signal_values[:187])
        ax.set_title("ECG Signal (First 187 Samples)")
        st.pyplot(fig)

        if st.button("🔍 Predict"):
            with st.spinner("Analyzing ECG signal..."):

                signal = preprocess_signal(signal_values)
                features = extract_features(signal).reshape(1, -1)
                features_scaled = scaler.transform(features)

                proba = model.predict_proba(features_scaled)[0]
                pred_idx = np.argmax(proba)

                label = encoder.classes_[pred_idx]
                confidence = proba[pred_idx]

                st.success(f"✅ Prediction: {label}")
                st.info(f"📈 Confidence: {confidence*100:.2f}%")

                # Probability Chart
                prob_df = pd.DataFrame({
                    "Class": encoder.classes_,
                    "Probability": proba
                })

                st.subheader("📊 Class Probabilities")
                st.bar_chart(prob_df.set_index("Class"))

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("💡 PulseSense-AI | Streamlit Deployment Ready")