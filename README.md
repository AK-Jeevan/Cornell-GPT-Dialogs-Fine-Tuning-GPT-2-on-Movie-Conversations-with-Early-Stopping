# CornellGPT‑Dialogs 🎬🤖
Fine‑tuning GPT‑2 on the **Cornell Movie Dialogs dataset** with Hugging Face Transformers.  
This project demonstrates conversational AI training with **early stopping** to prevent overfitting.

---

## ✨ Features
- Load and preprocess the Cornell Movie Dialogs dataset
- Tokenize text using GPT‑2 tokenizer
- Fine‑tune GPT‑2 with Hugging Face `Trainer`
- Apply **EarlyStoppingCallback** for efficient training
- Generate sample dialogue responses

---

## 📂 Dataset
We use the [Cornell Movie Dialogs dataset](https://huggingface.co/datasets/cornell_movie_dialog), which contains:
- 220,000+ conversational exchanges
- Extracted from movie scripts
- Ideal for chatbot training and dialogue modeling

---

## ⚙️ Installation

### Clone the repo
git clone https://github.com/your-username/CornellGPT-Dialogs.git
cd CornellGPT-Dialogs

### Install dependencies
pip install -r requirements.txt

#### Dependencies:

transformers

datasets

torch

## 📊 Results
Faster convergence with early stopping

Reduced overfitting on small dialogue dataset

Generates conversational responses after fine‑tuning

## 📌 Future Work
Add custom metrics (e.g., perplexity) for monitoring

Experiment with larger context windows

Fine‑tune on multi‑domain dialogue datasets

## 📜 License
MIT License. Free to use and modify. 
