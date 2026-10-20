# 🧠 Natural Language Processing (NLP) Master Curriculum & Study Guide

> **A Complete, Practical, Math-Grounded Journey from Text Preprocessing to Modern Transformers, LLMs, and RAG Pipelines.**

Designed specifically for self-study, research foundations, and production-grade mastery. Every module features rigorous mathematical intuitions, from-scratch Python/PyTorch implementations, visual architecture diagrams, and self-assessment challenges with solutions.

---

## 🗺️ Master Curriculum Roadmap

```text
[ Phase 1: Classical Foundations ]
       │
       ├── 01. Text Processing (Regex, Tokenization, Stemming, Lemmatization, Stopwords, POS, NER)
       └── 02. Text Representation (One-Hot, Bag of Words, N-Grams, TF-IDF, Word2Vec, GloVe)
       │
       ▼
[ Phase 2: Sequential Deep Learning ]
       │
       └── 03. Sequence Models (Vanilla RNN, BPTT, Vanishing Gradients, LSTM, GRU, BiLSTM)
       │
       ▼
[ Phase 3: The Transformer Revolution ]
       │
       ├── 04. Attention & Transformers (Scaled Dot-Product, Multi-Head Attention, Positional Encoding)
       └── 05. Pre-Trained Transformers & Hugging Face (BERT, RoBERTa, Subword Tokenizers, PyTorch Fine-Tuning)
       │
       ▼
[ Phase 4: Generative AI & Frontier Systems ]
       │
       └── 06. LLMs, Prompt Engineering & RAG (Autoregressive Generation, Vector Stores, Cosine Search, LoRA)
```

---

## 📚 Curriculum Breakdown

| Module / Notebook | Core Concepts Covered | Key Implementations & Libraries |
| :--- | :--- | :--- |
| **[01. Text Processing](1.Text_Processing.ipynb)** | Tokenization, Stemming vs Lemmatization, Stopword removal, Part-of-Speech (POS) tagging, Named Entity Recognition (NER) | `NLTK`, regex, text cleaning pipelines |
| **[02. Vectorization](2_Vectorization.ipynb)** | One-Hot Encoding, Bag of Words (CountVectorizer), TF-IDF mathematical derivation, N-gram modeling, Word2Vec (Skip-gram, CBOW) | `scikit-learn`, `gensim`, NumPy |
| **[03. Deep Sequence Models](3_Deep_Learning_Sequence_Models_RNN_LSTM_GRU.ipynb)** | Why BoW fails on order; BPTT & gradient vanishing proof; LSTM 3-gate anatomy ($f_t, i_t, o_t$); GRU 2-gate design ($r_t, z_t$); Bidirectional RNNs | From-scratch NumPy cells, PyTorch `nn.LSTM`, custom dataset loader, text classifier |
| **[04. Attention & Transformers](4_Attention_and_Transformers_Explained.ipynb)** | Seq2Seq bottleneck; Bahdanau vs Luong attention; Scaled Dot-Product Attention $Q, K, V$; Softmax saturation proof $\frac{1}{\sqrt{d_k}}$; Multi-Head Attention; Sinusoidal Positional Encoding | From-scratch PyTorch `ScaledDotProductAttention`, `MultiHeadAttention`, `PositionalEncoding`, Attention Heatmap visualizer |
| **[05. Hugging Face & BERT Fine-Tuning](5_HuggingFace_BERT_FineTuning_Mastery.ipynb)** | Pre-training vs Fine-tuning; MLM & NSP; Encoder vs Decoder vs Seq2Seq; WordPiece tokenization (`input_ids`, `attention_mask`); `pipeline()` API; AdamW & linear warmup | `transformers`, `AutoTokenizer`, `AutoModelForSequenceClassification`, custom fine-tuning loop |
| **[06. Generative AI, LLMs & RAG](6_Generative_AI_LLMs_and_RAG_Pipeline.ipynb)** | Autoregressive decoding (Greedy, Top-$k$, Top-$p$, Temperature); In-context learning, CoT, ReAct; Dense vector embeddings; In-Memory Vector DB; RAG context injection; PEFT & LoRA | Pure Python Vector Store, Top-$k$ Cosine search, Grounded Prompt Synthesis, LoRA parameter analysis |

---

## 🛠️ End-to-End Projects

| Project Notebook | Task & Domain | Methodology |
| :--- | :--- | :--- |
| **[Project 01: Ham/Spam Classification](Project01_Ham_Spam_Classification_using_NLTK_BoW.ipynb)** | SMS Spam Detection | NLTK preprocessing + Bag of Words + Naive Bayes |
| **[Project 02: Ham/Spam TF-IDF](Project02_Ham_Spam_Classification_using_BoW,_TF_IDF.ipynb)** | SMS Spam Filtering | TF-IDF Vectorizer + Scikit-Learn Classifiers |
| **[Project 03: Text Classification with Word2Vec](Project03_Text_Classification_with_Word2Vec_and_AvgWord2Vec.ipynb)** | Semantic Document Classification | Gensim Word2Vec + Average Word2Vec aggregation |
| **[Project 04: Kindle Review Sentiment Analysis](Project04_Kindle_Review_Sentiment_Analysis.ipynb)** | E-commerce Customer Review Sentiment | End-to-end NLP pipeline with feature extraction & model evaluation |
| **[Project 05 (Template): BiLSTM Sentiment Classifier](Project05_Template_BiLSTM_Sentiment_Classifier.ipynb)** | Deep Sentiment Sequence Classification | PyTorch `nn.LSTM(bidirectional=True)`, embeddings, training loop, live inference |
| **[Project 06 (Template): End-to-End RAG Document Assistant](Project06_Template_End_to_End_RAG_Document_Assistant.ipynb)** | Enterprise Document QA Assistant | Sliding-window chunking, Dense vector indexing, Cosine similarity, Grounded prompts |
| **[Project 07 (SOTA Flagship): Hybrid RAG & Re-Ranking System](Project07_SOTA_Hybrid_RAG_and_Reranking_System.ipynb)** | Production-Grade AI Knowledge Engine | Dense Vectors + BM25 Lexical, Reciprocal Rank Fusion (RRF), Cross-Encoder Re-Ranking, Faithfulness scoring |

---

## 🧠 Key Theoretical Equations Quick-Reference

### 1. TF-IDF
$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}, \quad \text{IDF}(t, D) = \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right)$$
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### 2. LSTM Cell Equations
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(Forget Gate)}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(Input Gate)}$$
$$\tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(Candidate State)}$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(Cell State)}$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(Output Gate)}$$
$$h_t = o_t \odot \tanh(C_t) \quad \text{(Hidden State)}$$

### 3. Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 4. Sinusoidal Positional Encoding
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### 5. Low-Rank Adaptation (LoRA)
$$W = W_0 + \frac{\alpha}{r}(B \times A), \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}, \; r \ll \min(d, k)$$

---

## 🚀 How to Study This Repository

1. **Step-by-Step Progression**: Follow notebooks in order from `1` through `6`.
2. **Type the Code Out**: Do not just read — execute the cells, inspect tensor shapes (`.shape`), and modify hyperparameters.
3. **Solve the Practice Challenges**: At the end of each module, attempt the Self-Assessment exercises without looking at the hidden solution blocks first.
4. **Build Custom Variations**: Experiment with changing hidden dimensions, sequence lengths, and testing custom input sentences!

---

## 💻 Tech Stack
- **Languages**: Python 3.10+
- **Deep Learning Frameworks**: PyTorch, Hugging Face Transformers
- **NLP & Scientific Computing**: NLTK, Gensim, Scikit-Learn, NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
