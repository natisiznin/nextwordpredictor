# 🧠 Next Word Predictor using LSTM

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

An interactive, deep-learning-powered natural language processing (NLP) web application that generates real-time next-word predictions based on user input. Built from scratch using a **Recurrent Neural Network (RNN)** with **Long Short-Term Memory (LSTM)** architectures in TensorFlow/Keras and deployed as a low-latency web app via **Streamlit**.

---

## 💭 The "Why" & Architecture Planning

When starting this project, the primary goal wasn't just to build another text generator—it was to look under the hood of sequence modeling and genuinely understand how recurrent neural networks handle vanishing gradients and contextual memory over sequential time steps. 

### Why LSTM over N-Grams or Transformers?
* **Traditional N-Grams:** Statistical models like Markov chains look at fixed windows of $n-1$ words. They are fast, but they fail completely at capturing long-term dependencies or contextual nuance across long sentences.
* **Transformers (LLMs):** While attention mechanisms dominate modern NLP, training a custom transformer from scratch requires massive compute resources. 
* **The LSTM Sweet Spot:** LSTMs introduce dedicated internal memory cells governed by three gates (**Forget Gate**, **Input Gate**, and **Output Gate**). This architecture allows the model to selectively retain relevant context from earlier in a sentence without blowing up compute budgets or suffering from vanishing gradients.

### Pipeline Architecture Flow
```text
[ Raw Text Corpus ] 
        │
        ▼  (Tokenization & Text Cleaning)
[ Integer Sequences ] 
        │
        ▼  (N-Gram Sequence Extraction & Pre-padding)
[ Equal-Length Input Vectors (X) & Target Word (y) ]
        │
        ▼  (Embedding Layer -> LSTM Layer -> Dense Softmax)
[ Trained Model (.keras) + Serialized Tokenizer (.pkl) ]
        │
        ▼  (Streamlit @st.cache_resource)
[ Real-Time Web App Interface ]
