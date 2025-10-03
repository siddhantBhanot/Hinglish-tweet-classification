# Hinglish Text Classification

This project focuses on classifying Hinglish (Hindi-English) social media text into three categories based on aggression levels:

- **NAG**: Non-Aggressive
- **CAG**: Covertly Aggressive
- **OAG**: Overly Aggressive

## Overview

Hinglish text is often written in the Roman script, making it challenging to process for NLP tasks. To address this, the project involves transliterating the text to the Devanagari script, handling dataset imbalances, and implementing both traditional machine learning (ML) models and deep learning (DL) models for classification.

---

## Key Features

### 1. **Data Conversion to Devanagari Script**

Given the Hinglish data was in Roman script, it was transliterated and/or translated into Devanagari script using:

- **Google Translate API**
- **IndicTrans library**

This conversion enabled the use of pre-trained Hindi word vectors for the deep learning models.

### 2. **Handling Imbalanced Dataset**

The Hindi dataset was divided into two parts:
- **hindi_train**
- **hindi_dev**

There was significant class imbalance among the categories (NAG, CAG, OAG) in the training data. To address this:

- **Text Augmentation**: Initially, the **iNLTK library** was explored to generate similar sentences. However, due to high noise in the text data, this approach was not effective.
- **Back Translation**: A more robust back-translation technique was implemented using the **Google Translate API**, which produced better results. This process not only balanced the dataset but also increased its size.

### 3. **Baseline ML Models**

Three baseline machine learning models were implemented using **TF-IDF** features:
- Support Vector Machine (**SVM**)
- **XGBoost**
- Logistic Regression

### 4. **Final Approach**

Inspired by the work of Md Abul Bashar et al. (2020) in *"Misogynistic Tweet Detection: Modelling CNN with Small Datasets"*, we:

1. Converted data into Devanagari script to utilize pre-trained Hindi word vectors.
2. Trained deep learning models, including **CNN** and **BiLSTM**, using these task-specific word vectors.

---

## Results

The performance of various models is summarized below:

| Method                | Accuracy (%) |
|-----------------------|--------------|
| **SVM**               | 74.31        |
| **XGBoost**           | 68.96        |
| **Logistic Regression**| 71.13        |
| **Bag of Words (BOW)**| 66.73        |
| **BiLSTM**            | 59.18        |
| **CNN**               | 68.30        |

---

## Usage

1. **Streamlit App**:
   A Streamlit application is deployed and can be accessed here: [Hinglish Text Classification App](https://hinglish-tweet-classification.streamlit.app/)

---

## Project Structure

```
Hinglish-Classification/
├── Resources/
├── Project Report and PPT/
├── Dataset/
├── Google Translate and IndicTrans Notebook/
├── Backtranslate Notebook/
├── Models/
├── deployment/
│   ├── hinglish_classifier.py
├── .gitignore
└── README.md
```

---

## References

1. TRAC 2 dataset, {Bhattacharya, Shiladitya and Singh, Siddharth and Kumar, Ritesh and Bansal, Akanksha and Bhagat, Akash and Dawer, Yogesh and Lahiri, Bornini and Ojha, Atul Kr.}, [ACL Anthology](https://www.aclweb.org/anthology/2020.trac2-1.25)
2. Jason Wei and Kai Zou, 2019. *"EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks"*, [arXiv:1901.11196](https://arxiv.org/abs/1901.11196)
3. Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad Norouzi, and Quoc V. Le. 2018. *"Qanet: Combining local convolution with global self-attention for reading comprehension"*, [arXiv:1804.09541](https://arxiv.org/abs/1804.09541)
4. Md Abul Bashar, Richi Nayak, Nicolas Suzor and Bridget Weir, 2020. *"Misogynistic Tweet Detection: Modelling CNN with Small Datasets"*, [arXiv:2008.12452](https://arxiv.org/abs/2008.12452)
5. Ai4Bharat pre-trained Hindi Word Embeddings: [GitHub](https://github.com/AI4Bharat/indicnlp_corpus#word-embeddings)

---
