# 💬 BERT-Powered Chatbot

A simple yet intelligent **Q&A chatbot** built with **Streamlit** and **BERT** (Bidirectional Encoder Representations from Transformers). It uses **cosine similarity** on BERT embeddings to match user queries to predefined Q&A pairs — perfect for demos, learning, or extending into a full FAQ bot.

---

## ✨ Features

- **BERT-based semantic matching** (not keyword-based!)
- Real-time chat interface using **Streamlit**
- Precomputed embeddings for fast inference
- Threshold-based fallback (`> 0.5` similarity)
- Caching with `@st.cache_resource` for efficient model loading

---

## 📦 Tech Stack

| Technology         | Purpose |
|--------------------|--------|
| `streamlit`        | Interactive web UI |
| `transformers`     | BERT tokenizer & model |
| `torch`            | Deep learning backend |
| `scikit-learn`     | Cosine similarity |
| `numpy`            | Array operations |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bert-chatbot.git
cd bert-chatbot

## Install Dependencies
pip install -r requirements.txt

**requirements.txt should include:**
streamlit
transformers
torch
scikit-learn

Run the App

pip install -r requirements.txt

🎯 How It Works

BERT Model (bert-base-uncased) generates embeddings for:
All predefined questions
User input

Cosine similarity finds the closest match
If similarity > 0.5 → return answer
Else → "I'm not sure how to respond to that."

🔧 Customize Q&A Pairs
Edit the qa_pairs dictionary in app.py:
pythonqa_pairs = {
    "Your question here": "Your answer here",
    "Another Q": "Another A",
}
