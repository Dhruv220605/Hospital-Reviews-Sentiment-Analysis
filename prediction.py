import numpy as np
import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification, AlbertConfig

# ── Config ───────────────────────────────────────────────────────────────────
MAX_LENGTH  = 256
MODEL_PATH  = "models/tf_model.h5"
CONFIG_PATH = "models/config.json"

# ── Label mapping ─────────────────────────────────────────────────────────────
label_mapping = {"Positive": 0, "Neutral": 1, "Negative": 2}

# ── Load tokenizer (uses cache if already downloaded) ─────────────────────────
print("Loading ALBERT tokenizer...")
try:
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", local_files_only=True)
    print("Tokenizer loaded from cache!")
except Exception:
    print("Downloading tokenizer (one time only)...")
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    print("Tokenizer downloaded!")

# ── Load model from LOCAL config.json + tf_model.h5 (no internet needed) ──────
print("Loading model from local files...")
config = AlbertConfig.from_pretrained(CONFIG_PATH, num_labels=len(label_mapping))
model  = TFAlbertForSequenceClassification(config)

# Build model with dummy input first (required before loading weights)
dummy = tokenizer("test", return_tensors="tf", padding="max_length", max_length=16)
model(dummy)

# Load your trained weights
model.load_weights(MODEL_PATH)
print("Model loaded successfully!\n")

# ── Reviews to predict ────────────────────────────────────────────────────────
new_texts = [
    'Facilities are clean',
    'The service was excellent',
    'Staff were rude and unhelpful',
    'The experience was fine overall',
    'The quality of service was good',
    'The experience was disappointing'
]

# ── Tokenize ──────────────────────────────────────────────────────────────────
new_encodings = tokenizer(
    new_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors='tf'
)

input_ids      = new_encodings['input_ids'].numpy()
attention_mask = new_encodings['attention_mask'].numpy()

# ── Predict ───────────────────────────────────────────────────────────────────
predictions      = model.predict([input_ids, attention_mask])
logits           = predictions.logits
predicted_labels = tf.argmax(logits, axis=1).numpy()

predicted_sentiments = [
    list(label_mapping.keys())[list(label_mapping.values()).index(label)]
    for label in predicted_labels
]

# ── Print results ─────────────────────────────────────────────────────────────
print("=" * 55)
print("           SENTIMENT PREDICTIONS")
print("=" * 55)
for text, sentiment in zip(new_texts, predicted_sentiments):
    emoji = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}[sentiment]
    print(f"{emoji} {sentiment:10s} → {text}")
print("=" * 55)