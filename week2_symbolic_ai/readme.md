# Statistical Spam Classifier

**Course:** CSC6313 AI Foundations  
**Week:** 02 Statistical AI / Naive Bayes  
**Author:** Shaun Clarke

---

## Project Overview

This project analyzes, documents, and extends a **statistical spam classifier** built with scikit-learn. Rather than encoding rules by hand like in Week 1, this classifier *learns* spam patterns from labeled training data using the **Multinomial Naive Bayes** algorithm. The assignment required deep-diving into each library, function, and variable to explain what is happening under the hood then comparing the statistical approach to the symbolic approach from Week 1.

The bonus task was also completed: the training set was expanded to 30 labeled examples (15 spam / 15 not spam), including tricky edge cases, and all test messages now return predictions rather than just the first one.

---

## How It Works

The classifier runs in three stages:

### Stage 1 Vectorization (CountVectorizer)
Raw text cannot be fed directly into a math model. `CountVectorizer` converts each message into a **numeric vector** by:
1. Tokenizing all training messages into individual words
2. Building a vocabulary of every unique word across all documents, where each word gets an index position
3. Representing each message as a vector of word frequency counts aligned to that vocabulary

Example: if the vocabulary is `["buy", "now", "free"]`, then the message `"buy now"` becomes `[1, 1, 0]` the `0` is because `"free"` does not appear, but the vector must have a position for every word in the vocabulary.

### Stage 2 Training (MultinomialNB)
`MultinomialNB` implements the **Multinomial Naive Bayes** algorithm:
- **Naive** assumes all features (words) are statistically independent of each other
- **Bayes** uses Bayes' theorem to calculate probabilities
- **Multinomial** designed for discrete frequency counts (word counts), making it well-suited for text classification

During `model.fit()`, the model calculates the probability of each word appearing in spam vs. not-spam messages. Words like `"click"`, `"free"`, and `"urgent"` will develop high spam probabilities from the training data.

### Stage 3 — Prediction
The test messages are converted using `vectorizer.transform()` **not** `fit_transform()`, because the vocabulary must stay fixed from training. Using `fit_transform` on the test set would create a new vocabulary and break the model. The trained model then calculates the probability of each label for each test message and returns the most likely one.

---

## Week 1 vs. Week 2 Comparison

| Dimension | Symbolic (Week 1) | Statistical (Week 2) |
|---|---|---|
| **Approach** | Hand-written `if/else` rules | Learns patterns from labeled data |
| **Transparency** | Fully explainable every decision traces to a rule | Black box decisions based on probability |
| **Scalability** | Poor — rules must be written and updated manually | Excellent retrain on new data as patterns evolve |
| **Accuracy** | Limited to what the designer anticipated | Can recognize spam it has never seen before |
| **Maintenance** | High effort new spam tactics require new rules | Low effort feed new labeled examples and retrain |
| **False positives** | Higher rigid rules can misfire | Lower probabilistic scoring handles edge cases better |
| **Best for** | Deterministic, stable, rule-based environments | Dynamic, large-scale, evolving classification problems |

The symbolic model wins on explainability and requires no data. The statistical model wins on almost everything else especially as the volume and variety of messages grow.

---

## Bonus Task — Expanded Training Set

The original skeleton had no training data. The expanded version includes **30 labeled messages** (15 spam, 15 not spam) with several intentionally tricky examples:

- `"I need to buy a birthday present for my mom"` contains the word `"buy"` but is **not spam**. The model correctly classifies it because the surrounding context (birthday, mom, present) matches the not-spam pattern in training data.
- `"Your package could not be delivered. Confirm your address at the link below."` deceptive phrasing that mimics a legitimate delivery notification but is spam.

Adding diverse, edge-case examples directly improved classification accuracy for borderline messages. The more varied the training data, the better the model generalizes.

---

## Prerequisites

- Python 3.10+
- scikit-learn
- numpy
- scipy

```bash
pip install scikit-learn numpy scipy
```

---

## How to Run

```bash
python statistical_classifier.py
```

The program runs automatically — no user input required. It trains the model on the hardcoded training set, runs predictions on the test set, and prints each test message alongside its predicted label.

**Example output:**
```
IRS warning: Unpaid taxes detected. Immediate payment required to avoid arrest.: spam

Hey, are we still on for dinner tonight at 7?: not spam

Your Amazon order has shipped and will arrive tomorrow.: not spam
```

---

## Libraries Used

| Library | Purpose |
|---|---|
| `sklearn.feature_extraction.text.CountVectorizer` | Converts raw text into word-frequency numeric vectors for model input |
| `sklearn.naive_bayes.MultinomialNB` | Implements the Multinomial Naive Bayes algorithm for text classification |
| `numpy` | Numeric array operations; provides the `NDArray` return type for predictions |
| `scipy.sparse.csr_matrix` | Compressed sparse row matrix format used internally by CountVectorizer for memory-efficient storage of the word count matrix |

---

## File Structure

```
week02/
├── statistical_classifier.py    # Fully documented implementation
└── README.md                    # This file
```