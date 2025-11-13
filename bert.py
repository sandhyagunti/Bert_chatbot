import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Load BERT ----------------
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model

tokenizer, model = load_bert_model()

# ---------------- Q/A Pairs ----------------
qa_pairs = {
    "Hi": "Hello there! 👋",
    "How are you?": "I’m doing great, thanks for asking! How about you?",
    "What is your name?": "I’m your friendly chatbot buddy 🤖",
    "What can you do?": "I can chat, tell jokes, and maybe even make you smile 🙂",
    "Tell me something interesting.": "Did you know octopuses have three hearts? 🐙❤️❤️❤️",
    "What is AI?": "AI is the ability of machines to simulate human intelligence.",
    "What is data science?": "Data Science is analyzing data to make smart decisions.",
    "What is machine learning?": "Machine Learning teaches computers to learn from data.",
    "What is deep learning?": "Deep Learning uses multi-layered neural networks.",
    "Tell me a joke.": "Why don’t skeletons fight each other? They don’t have the guts. 💀😂",
}

# ---------------- BERT Embeddings ----------------
def get_embeddings(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).numpy()

precomputed = {q: get_embeddings(q) for q in qa_pairs}

# ---------------- Chatbot Logic ----------------
def chatbot_reply(text):
    user_emb = get_embeddings(text)
    scores = {q: cosine_similarity(user_emb, precomputed[q])[0][0] for q in qa_pairs}
    best = max(scores, key=scores.get)
    return qa_pairs[best] if scores[best] > 0.5 else "I'm not sure how to respond to that."

# ---------------- UI ----------------
st.set_page_config(page_title="BERT Chatbot 💬", layout="centered")

st.title("💬 BERT Chatbot")
st.write("Chat with the BERT-powered assistant below!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    reply = chatbot_reply(user_input)
    st.session_state.messages.append({"role": "assistant", "content": reply})

    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(reply)
