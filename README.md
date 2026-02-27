# CSC6313 AI Foundations

**Student:** Shaun Clarke  
**Instructor:** Margaret Mulhall  
**Institution:** Merrimack College

---

## About This Repository

This repository contains all weekly projects completed for CSC6313 AI Foundations. Each project builds on the previous one, progressing from hand-written rule systems through statistical models, search algorithms, retrieval-augmented generation, supervised learning pipelines, non-linear classifiers, and finally a neural network built from scratch.

Every project folder includes the full implementation and a detailed README covering how the code works, the design decisions behind it, and instructions for running it locally.

---

## Projects

### Week 01 [The Symbolic Spam Classifier](https://github.com/shaunclarke2333/ai_foundations_class/tree/main/week1_ai_landscape)
**Topics:** Rule-based AI, symbolic reasoning  
A hand-crafted spam classifier built on 4 explicit rules no machine learning, no training data. Demonstrates how AI worked before statistical methods: hard-coded logic that humans can read, audit, and debug directly.

---

### Week 02 [The Statistical Spam Classifier](https://github.com/shaunclarke2333/ai_foundations_class/tree/main/week2_symbolic_ai)
**Topics:** Naive Bayes, probabilistic classification, scikit-learn  
A Naive Bayes classifier trained on 30 labeled email examples. The first project where the model *learns* from data rather than following hand-written rules. Demonstrates how statistical models differ fundamentally from symbolic ones.

---

### Week 03 [BFS vs A* Pathfinding](https://github.com/shaunclarke2333/ai_foundations_class/tree/main/week3_search_problem_solving)
**Topics:** Search algorithms, heuristics, graph traversal  
Side-by-side comparison of Breadth-First Search and A* across 4 maze configurations. BFS explores blindly in all directions; A* uses Manhattan distance as a heuristic to prioritize paths heading toward the goal. A counter tie-breaker ensures consistent, reproducible results.

---

### Week 04 [The RAG Chatbot](https://github.com/shaunclarke2333/ai_foundations_class/tree/main/week4_knowledge_language_generation)
**Topics:** Retrieval-Augmented Generation, vector databases, semantic search  
A question-answering chatbot backed by ChromaDB and sentence-transformers. User queries are converted to embeddings, matched against 6 stored documents by cosine similarity, and the most relevant passage is returned as the answer no LLM API required.

---

### Week 05 [The Diagnostic Prediction Engine](https://github.com/shaunclarke2333/ai_foundations_class/tree/main/week5_machine_learning_basics)
**Topics:** Supervised ML pipeline, preprocessing, bias-variance tradeoff, logistic regression  
A full end-to-end supervised learning pipeline predicting patient health outcomes from synthetic data. Covers synthetic data generation with intentional missing values, median imputation, Z-score standardization, and three regression models (underfit / overfit / optimal) to demonstrate the bias-variance tradeoff. Closes with an interactive terminal inference engine.

---

### Week 06 [The Classifier Showdown](https://github.com/shaunclarke2333/ai_foundations_class/tree/main/week6_machine_learning_methods)
**Topics:** Decision Trees, Random Forests, k-Nearest Neighbors, ensemble methods  
Three non-linear classifiers trained on the same patient dataset and compared head-to-head. Includes a Matplotlib popup showing the Decision Tree's learned if-then rules and feature importance scores. An interactive inference engine collects votes from all three models and uses majority vote for the final diagnosis.

---

### Week 07 — [The Neural Thermostat Agent](https://github.com/shaunclarke2333/ai_foundations_class/tree/main/week7_neural_networks)
**Topics:** Single-neuron neural network, sigmoid activation, gradient descent  
A Perceptron implemented entirely from scratch using only NumPy no PyTorch, no TensorFlow. Implements the sigmoid activation function, a manual forward pass, and weight updates via gradient descent by hand. The agent learns to control a smart thermostat based on temperature and room occupancy, and a loss curve confirms convergence.

---

## Progression at a Glance

```
Week 01  →  Hard-coded rules          (No learning)
Week 02  →  Probabilistic model       (Learns from labeled examples)
Week 03  →  Search algorithms         (Intelligent traversal with heuristics)
Week 04  →  Semantic retrieval        (Embedding-based knowledge lookup)
Week 05  →  Supervised ML pipeline    (Regression, preprocessing, model comparison)
Week 06  →  Non-linear classifiers    (Trees, forests, distance-based voting)
Week 07  →  Neural network from scratch (Weights, activation, gradient descent)
```

---

## Running Any Project

Each project is self-contained. Navigate into the folder and follow the instructions in its README.

```bash
cd week01
python shaun_clarke_csc6313_week01.py
```

### Shared dependencies across most projects
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Additional dependencies for specific weeks
```bash
# Week 04 — RAG Chatbot
pip install chromadb sentence-transformers
```