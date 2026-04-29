import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification, AlbertConfig
import tensorflow as tf

MAX_LENGTH  = 256
BATCH       = 32
MODEL_PATH  = "models/tf_model.h5"
CONFIG_PATH = "models/config.json"
FILE_PATH   = "datasets/processed_hospital_reviews.csv"
class_names = ['Positive', 'Neutral', 'Negative']
label_mapping = {"Positive": 0, "Neutral": 1, "Negative": 2}

print("Loading dataset...")
df = pd.read_csv(FILE_PATH)

# FIX: use to_numpy() instead of .values to avoid pyarrow error
_, test_texts, _, test_labels = train_test_split(
    df['text'].to_numpy(),
    df['sentiment'].to_numpy(),
    test_size=0.2,
    random_state=42
)

y_true = np.array([label_mapping[label] for label in test_labels])

print("Loading tokenizer...")
try:
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", local_files_only=True)
    print("Tokenizer loaded from cache!")
except Exception:
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

print("Loading model from local files...")
config = AlbertConfig.from_pretrained(CONFIG_PATH, num_labels=len(label_mapping))
model  = TFAlbertForSequenceClassification(config)
dummy  = tokenizer("test", return_tensors="tf", padding="max_length", max_length=16)
model(dummy)
model.load_weights(MODEL_PATH)
print("Model loaded!\n")

print("Running predictions on test set...")
encodings = tokenizer(
    list(test_texts),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors='tf'
)
input_ids      = encodings['input_ids'].numpy()
attention_mask = encodings['attention_mask'].numpy()

all_logits = []
for i in range(0, len(input_ids), BATCH):
    preds = model.predict([input_ids[i:i+BATCH], attention_mask[i:i+BATCH]], verbose=0)
    all_logits.append(preds.logits)

all_logits     = np.concatenate(all_logits, axis=0)
y_pred_classes = np.argmax(all_logits, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Value", fontsize=10)
plt.ylabel("Actual Value",    fontsize=10)
plt.title("Confusion Matrix — ALBERT Hospital Sentiment", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n")
print("=" * 55)
print("         CLASSIFICATION REPORT")
print("=" * 55)
print(classification_report(y_true, y_pred_classes, target_names=class_names))
print("✅ Confusion matrix saved as confusion_matrix.png")