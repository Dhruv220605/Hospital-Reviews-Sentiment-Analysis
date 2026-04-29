# 🏥 Hospital Reviews Sentiment Analysis

> Deep Learning powered sentiment analysis using **ALBERT Transformer** to classify hospital reviews as **Positive**, **Neutral**, or **Negative** — with an interactive Streamlit web app.

---

## 📸 Project Preview

| Predict Page | Model Evaluation | Dataset Explorer |
|---|---|---|
| Type any review and get instant sentiment | Confusion matrix + accuracy charts | Browse and filter all 999 reviews |

---

## 🧠 How It Works

```
User types review → Tokenizer converts to numbers → ALBERT model reads it → Outputs sentiment
```

1. **Text Input** — User enters a hospital review
2. **Tokenization** — ALBERT tokenizer converts text to numerical tokens
3. **Prediction** — Fine-tuned ALBERT model classifies the sentiment
4. **Output** — Returns Positive 😊, Neutral 😐, or Negative 😠

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total Reviews | 999 |
| Positive | 333 |
| Neutral | 333 |
| Negative | 333 |
| Balance | ✅ Perfectly balanced |

---

## 🤖 Model

| Property | Value |
|---|---|
| Base Model | ALBERT (A Lite BERT) by Google |
| Framework | TensorFlow 2.15 |
| Task | 3-class text classification |
| Epochs | 50 |
| Batch Size | 32 |
| Max Token Length | 256 |
| Learning Rate | 1e-6 |

---

## 🗂️ Project Structure

```
Hospital-Reviews-Sentiment-Analysis/
│
├── 📁 datasets/
│   └── processed_hospital_reviews.csv   # 999 hospital reviews
│
├── 📁 models/
│   ├── config.json                       # ALBERT model configuration
│   └── tf_model.h5                       # Trained model weights (download separately)
│
├── 📄 streamlit_app.py                   # Main web application
├── 📄 prediction.py                      # Sentiment prediction script
├── 📄 confusion_matrix.py               # Model evaluation + confusion matrix
├── 📄 visualizing_result.py             # Training accuracy/loss charts
├── 📄 reading_dataset.py                # Dataset loading and splitting
└── 📄 README.md
```

---

## ⚙️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| TensorFlow 2.15 | Deep learning framework |
| HuggingFace Transformers 4.35 | ALBERT model |
| Streamlit | Web application |
| Scikit-learn | Evaluation metrics |
| Matplotlib + Seaborn | Visualizations |
| Pandas + NumPy | Data processing |

---

## 🚀 How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/Dhruv220605/Hospital-Reviews-Sentiment-Analysis.git
cd Hospital-Reviews-Sentiment-Analysis
```

### Step 2 — Install dependencies
```bash
py -3.11 -m pip install tensorflow==2.15.0 wrapt==1.14.1 transformers==4.35.0 streamlit pandas numpy scikit-learn matplotlib seaborn sentencepiece
```

### Step 3 — Add model weights
Download `tf_model.h5` and place it inside the `models/` folder.
```
models/
    config.json     ✅ already in repo
    tf_model.h5     ← place here
```

### Step 4 — Run the web app
```bash
py -3.11 -m streamlit run streamlit_app.py
```

Open browser at **http://localhost:8501**

---

## 📄 Run Individual Scripts

```bash
# Load and explore dataset
py -3.11 reading_dataset.py

# Run predictions on sample reviews
py -3.11 prediction.py

# Generate accuracy/loss charts
py -3.11 visualizing_result.py

# Generate confusion matrix + classification report
py -3.11 confusion_matrix.py
```

---

## 📈 Results

| Metric | Score |
|---|---|
| Positive Recall | 100% |
| Negative Recall | 100% |
| Neutral Recall | ~73% |
| Overall Accuracy | ~69% |

> **Note:** Neutral class is harder to classify as neutral language often overlaps with positive and negative — a known challenge in 3-class sentiment analysis.

---

## 🖥️ App Features

- 🔍 **Predict Sentiment** — Single review prediction with color coded result
- 📋 **Bulk Predict** — Paste multiple reviews, get all results at once
- 📊 **Model Evaluation** — Confusion matrix, accuracy, precision, recall, F1 score
- 📁 **Dataset Explorer** — Browse, filter and search all 999 reviews
- 🎨 **Dark Mode UI** — Clean dark themed interface

---

## 👨‍💻 Author

**Dhruv**
- GitHub: [@Dhruv220605](https://github.com/Dhruv220605)

---

## 📝 License

This project is for educational purposes.
