# 🎧 SonicSync – AI-Powered Equalizer & Genre Classifier

![SonicSync Architecture](./path-to/Screenshot%202025-04-18%20at%2011.12.45 PM.png)

## Why AI in an Equalizer?

We all love listening to music. As the saying goes:

> _“Music is like food — you don’t need to know how it’s made, as long as it’s tasty!”_

But just like food, everyone has a unique taste. Some prefer it spicy, others salty — the same applies to music. One song can sound different and better to each person with the right **Equalizer** (EQ) settings.

### So why use AI?

The world of equalizers is vast and technical. Not everyone is an audio expert, but everyone deserves a great listening experience. Our tool bridges that gap.

---

## 🎯 Goal

To provide users with a **seamless, high-quality, and personalized music listening experience** by:

- Accurately classifying music genres
- Auto-tuning equalizer settings based on genre, user preferences, and headphones

---

## 💡 Motivation

The rise of multimedia has flooded us with digital music, but:

- The experience isn’t tailored for every user
- Manual EQ tweaking is not beginner-friendly

Our tool solves this by combining genre classification with automatic EQ adjustment.

---

## ⚙️ Challenges

- Accurately classifying music across a variety of genres and formats
- Dealing with different devices and headphone signatures
- Understanding both **spatial** (spectrogram) and **temporal** (sequence) audio features

---

## 🧠 Our Solution

- A **hybrid multi-model approach** combining:
  - CNNs for spatial audio features
  - BiLSTM for temporal dynamics
  - Random Forest for robust ensemble predictions
- Personalized EQ adjustments via **AutoEQ** based on:
  - Predicted genre
  - User’s headphones
  - Individual preferences

---

## 🔮 Future Work

- **Ensemble learning**: Combine outputs from multiple classifiers
- **Advanced Architectures**: Explore transformers and attention-based models
- **Mobile Application**: Real-time music classification and EQ on-the-go
- **Command-Line Support**: For power users and automation
- **Extended Features**: Playlist recommendations, live audio stream support, mood-based suggestions, etc.

---

## 🚀 How to Run This Project

### Backend Setup

1. Navigate to the backend directory:

   ```bash
   cd webapp
   ```

2. Install Python dependencies:

   ```bash
   python -m pip install -U -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```

The FastAPI backend will run on: `http://localhost:8000`

---

### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd ui
   ```

2. Install frontend dependencies:

   ```bash
   npm install
   ```

3. Start the frontend server:
   ```bash
   npm start
   ```

Visit: `http://localhost:3000` in your browser.

---

## ✨ Tech Stack

- **Frontend**: React.js (UI for genre display & EQ)
- **Backend**: FastAPI
- **ML Models**: Keras, PyTorch, RandomForest, Librosa
- **Audio Analysis**: MFCCs, Spectrograms
- **Equalizer Logic**: AutoEQ (custom frequency response modeling)

---

## 📂 Project Structure

```
SonicSync/
│
|
├── main.py               # Backend - FastAPI
├── genre_classifier.py
├── ensemble_model.py
├── model.pkl
|── ...
│
├── ui/                   # Frontend - React.js
│   └── ...
│
├── data/
│   ├── audio/            # Audio samples
│   ├── entries.json
│   └── ...
│
├── temp_audio/           # Uploaded files (runtime)
└── requirements.txt
```

---

## 🎵 Experience the magic of AI + Audio

Get accurate genre predictions and auto-tuned audio—just upload a track and let SonicSync do the rest 🎶
