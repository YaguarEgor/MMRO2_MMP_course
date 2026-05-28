# MMRO / MMP Course II — Machine Learning Practice

This repository contains solved practical assignments from the second part of the **Mathematical Methods of Object Recognition** course, completed as part of the **MMP program at CMC MSU**.

The repository continues the practical machine learning workflow from the first part of the course and focuses on advanced topics, including optimization methods, probabilistic models, unsupervised learning, and reinforcement learning.

---

## Repository Overview

Each directory corresponds to a separate homework assignment from the course. The assignments are implemented mainly as Jupyter notebooks and include Python code, experiments, visualizations, and analysis.

```text
MMRO2_MMP_course/
│
├── dz1_metopts/
│   └── MMF_opt_hw_Podyablonskiy.ipynb
│
├── dz2_em/
│   ├── homework_practice_09_em_podyablonskiy.ipynb
│   ├── metrics.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── L.npy
│   └── y.npy
│
├── dz3_unsupervised_learning/
│   └── homework-practice-1.ipynb
│
├── dz4_ocrl/
│   └── HomeworkOCRL_2026_Podyablonskiy.ipynb
│
└── README.md
```

---

## Topics Covered

### 1. Optimization Methods

The first assignment is focused on numerical optimization methods used in machine learning.

This part covers:

* gradient-based optimization;
* iterative optimization algorithms;
* loss function minimization;
* convergence analysis;
* comparison of optimization techniques;
* experiments with different parameter settings.

Optimization is one of the central tools behind modern machine learning, since model training is usually formulated as the minimization of a loss function.

---

### 2. Expectation-Maximization Algorithm

The second assignment is dedicated to the **Expectation-Maximization algorithm**, a classical method for training probabilistic models with hidden variables.

This part covers:

* probabilistic modeling;
* latent variables;
* the EM algorithm;
* iterative parameter estimation;
* model quality metrics;
* data preprocessing;
* experiments with model behavior.

The folder also contains additional Python modules for metrics, models and preprocessing, which makes the implementation more modular and structured.

---

### 3. Unsupervised Learning

The third assignment focuses on unsupervised learning methods.

This part covers:

* clustering;
* dimensionality reduction;
* analysis of hidden data structure;
* representation of unlabeled data;
* visualization of high-dimensional data;
* comparison of unsupervised learning methods.

Unlike supervised learning, where the model is trained using known target labels, unsupervised learning discovers structure in data automatically.

---

### 4. OCRL: Optimal Control and Reinforcement Learning

The fourth assignment is related to **optimal control and reinforcement learning**.

This part covers:

* Markov decision processes;
* agents and environments;
* reward-based learning;
* value functions;
* policies;
* practical reinforcement learning experiments.

Reinforcement learning studies how an agent learns to make decisions through interaction with an environment, receiving rewards or penalties for its actions.

---

## Technologies Used

The repository is based on the Python scientific and machine learning ecosystem:

* **Python**
* **Jupyter Notebook**
* **NumPy**
* **pandas**
* **Matplotlib**
* **scikit-learn**

The assignments also use additional Python tools for optimization, probabilistic modeling, visualization and reinforcement learning experiments.

---

## Purpose of the Repository

This repository is intended as a personal collection of solved practical assignments from the second part of the MMRO/MMP machine learning course.

It can be useful for:

* reviewing advanced machine learning topics;
* studying practical implementations of ML algorithms;
* comparing different models and optimization approaches;
* refreshing course material before exams;
* keeping track of solved homework assignments;
* using notebooks as references for future ML projects.

