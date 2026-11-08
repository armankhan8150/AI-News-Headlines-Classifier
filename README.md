# AI News Headlines Classifier

A fine-tuned BERT model that classifies news headlines into 4 categories: World, Sports, Business, and Technology, achieving **94.83% accuracy** on the AG News benchmark dataset, with a Streamlit web interface for real-time classification.

---

## 🏆 Model Performance

| Metric | Score |
|---|---|
| **Test Accuracy** | **94.83%** |
| Dataset | AG News (120K train / 7.6K test) |
| Base Model | `bert-base-uncased` |
| Fine-tuning Epochs | 3 |
| Batch Size | 16 |

### Per-Class Accuracy

| Category | Precision | Recall |
|---|---|---|
| World | 95.1% | 94.7% |
| Sports | 97.8% | 98.1% |
| Business | 92.4% | 93.0% |
| Technology | 94.2% | 93.5% |

---

## 🧠 How It Works

```
Raw Headline Text
    │
    ▼
┌──────────────────────┐
│   BERT Tokenizer      │  ──→  Tokenizes text, adds [CLS] / [SEP] tokens
│   (bert-base-uncased) │       Pads/truncates to max_length=128
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│   Fine-tuned BERT     │  ──→  Transformer encoder (12 layers)
│   + Classification    │       [CLS] token → Linear(768, 4) → Softmax
│     Head              │
└──────────────────────┘
    │
    ▼
Predicted Category + Confidence Score
(World / Sports / Business / Technology)
```

### Key Design Decisions
- **BERT over traditional ML**: contextual embeddings capture semantic meaning in short headlines far better than TF-IDF + logistic regression
- **bert-base-uncased**: sufficient for news classification without the compute cost of BERT-large; uncased handles mixed-case headlines naturally
- **HuggingFace Trainer API**: streamlined fine-tuning with built-in gradient accumulation, evaluation loops, and checkpoint saving
- **Confidence scores displayed**: softmax probabilities shown alongside predictions so users can gauge model certainty

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Base Model | `bert-base-uncased` (HuggingFace) |
| Framework | PyTorch + HuggingFace Transformers |
| Training API | HuggingFace Trainer |
| Dataset | AG News (via `datasets` library) |
| Web UI | Streamlit |
| Language | Python 3.10+ |

---

## Features

- 🤖 **Fine-tuned BERT**: not just a wrapper; custom classification head trained on AG News
- 📊 **94.83% test accuracy**: validated on the standard AG News test split
- 🔢 **Confidence scores**: softmax probabilities for all 4 classes
- 🖥 **Streamlit UI**: paste any headline and classify in real time
- 📦 **Reproducible training**: full training script with fixed seeds and logged metrics

---

## Project Structure

```
├── train.py                # Fine-tuning script (HuggingFace Trainer)
├── predict.py              # Inference pipeline
├── app.py                  # Streamlit web interface
├── model/                  # Saved fine-tuned model weights
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer/
├── notebooks/
│   └── training_analysis.ipynb
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- GPU recommended for training (CPU works for inference)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/armankhan8150/AI-News-Headlines-Classifier.git
cd AI-News-Headlines-Classifier

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
# Model saved to ./model/
```

### Run the Web App

```bash
streamlit run app.py
```

---

## Example Predictions

| Headline | Predicted Category | Confidence |
|---|---|---|
| "Fed raises interest rates by 25 basis points" | Business | 97.2% |
| "Real Madrid wins Champions League final" | Sports | 99.1% |
| "OpenAI releases new reasoning model" | Technology | 95.8% |
| "UN Security Council holds emergency session" | World | 93.4% |

---

## Training Details

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

---

## Roadmap

- [ ] Multi-label classification support
- [ ] Support for full article text (not just headlines)
- [ ] Model distillation for faster inference
- [ ] REST API endpoint

---

## Author

**Arman Malik** — AI Engineer
[LinkedIn](https://linkedin.com/in/arman-malik) · [GitHub](https://github.com/armankhan8150)

---

## License

MIT License: see [LICENSE](LICENSE) for details.
