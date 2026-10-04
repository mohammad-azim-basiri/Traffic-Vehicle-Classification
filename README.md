# Traffic Vehicle Classification Using Deep Learning

## Project Overview

This project focuses on developing a deep learning-based system for **traffic vehicle image classification** using **Convolutional Neural Networks (CNNs)** and transfer learning.

The goal is to classify vehicle images into **8 different classes** and investigate the impact of different deep learning techniques, training strategies, and model architectures on classification performance.

The dataset contains the following vehicle classes:

* `ambulance`
* `autobus`
* `kamyun`
* `kamyunet`
* `minibus`
* `savari`
* `taxi`
* `vanet`

The project initially uses a custom **Small CNN** as a baseline model. Several techniques are then investigated, including **Data Augmentation, Dropout, Pooling, Weight Decay, Learning Rate Scheduling, and handling class imbalance**.

In addition, several pretrained architectures are explored and compared to evaluate their performance, parameter efficiency, and suitability for the vehicle classification task.

## How to Run

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Then, run the project using the scripts and modules provided in the `src` directory.

The general project structure is:

```text
project/
├── src/
│   ├── 01_eda.ipynb
│   ├── datasets.py
│   ├── evaluation.ipynb
│   ├── utils.py
│   ├── app.py
│   ├── models.py
│   ├── predict.py
│   ├── train.py
│   ├── transforms.py
├── report/
│   ├── traffic_vehicle_report.md
│   ├──epochs_report.md
├── data/
├── results/
├── requirements.txt
└── README.md
```

For detailed information about dataset preparation, training configurations, experiments, model evaluation, and results, please refer to the full project report.

## Results

Several models and training configurations were evaluated throughout the project.

The **Small CNN** was used as the baseline model, followed by experiments investigating the effect of different regularization and optimization techniques. Pretrained architectures such as **ResNet18** and additional architectures from the **MobileNet Small and MobileNet Large** families were also investigated.

The experiments include:

* Comparison of different model architectures
* Effect of Data Augmentation
* Effect of Dropout
* Comparison of Pooling strategies
* Effect of Weight Decay
* Effect of Learning Rate Scheduling
* Evaluation under class imbalance
* Comparison of model parameter counts
* Evaluation using Accuracy, Precision, Recall, and F1-Score
* Analysis of misclassified validation samples

Detailed experimental results, graphs, and comparisons are provided in the project report.

## Full Report

[View the Full Report](./reports/traffic_vehicle_report.md)

## Project Code

[View the Source Code](./src/)
