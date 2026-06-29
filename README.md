# 🔍 Fake Job Posting Detector

> A Machine Learning-powered system to detect fraudulent job listings using NLP and hybrid rule-based detection.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Flask](https://img.shields.io/badge/Flask-3.0.0-green) ![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

With the rise of online recruitment scams, this system automatically identifies fraudulent job postings by analyzing job descriptions using NLP techniques and machine learning models. It flags suspicious postings in real time and provides confidence scores and detailed explanations.

<img width="1686" height="857" alt="image" src="https://github.com/user-attachments/assets/f6df734f-14c3-40cd-a9a5-9196e8343cc6" />
<img width="1752" height="871" alt="image" src="https://github.com/user-attachments/assets/d3a4159f-c837-4d1d-96b2-1baaeb93bd80" />

---

## ✨ Features

- Classifies job postings as **Real** or **Fake** with confidence score
- Detects **60+ fraud patterns** across 6 categories (financial, urgency, payment, etc.)
- Hybrid detection: ML prediction + rule-based red flag analysis
- REST API with Flask for real-time predictions
- Batch prediction support
- URL scraping to auto-extract job postings from web pages
- Explainable AI: shows exactly which red flags were detected and why

---

## 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11 |
| ML Models | Naïve Bayes, Logistic Regression, Random Forest |
| NLP | NLTK, TF-IDF Vectorizer (bigrams) |
| Backend | Flask 3.0, Flask-CORS |
| Data | Pandas, NumPy, Scikit-learn |
| Scraping | BeautifulSoup |
| Dataset | EMSCAD (17,880 job postings) |

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Naïve Bayes** ⭐ | **97.54%** | **84.0%** | 60.7% | **0.705** |
| Logistic Regression | 96.28% | 57.8% | **85.5%** | 0.690 |
| Random Forest | 96.64% | 61.2% | 71.7% | 0.665 |

**Best Model: Naïve Bayes** — Highest accuracy and precision with fast inference.

---

## 🔴 Red Flags Detected

The system detects fraud across 6 categories:

| Category | Examples |
|----------|---------|
| Financial | Registration fee, internship fee, one-time payment |
| Payment Methods | Wire transfer, Western Union, cryptocurrency, gift cards |
| Personal Info | Bank account, SSN, credit card, routing number |
| False Promises | Guaranteed income, get rich quick, financial freedom |
| Urgency Tactics | Act now, limited time, confirm your seat, urgent |
| Suspicious Contact | Free email domains (gmail, yahoo for company contact) |

---

## 🔮 Future Work

- BERT/RoBERTa transformer models for better contextual understanding
- Chrome browser extension for real-time job portal scanning
- WHOIS domain verification
- LinkedIn/Glassdoor API integration
- Multilingual support
- Mobile app
- Docker containerization and cloud deployment (AWS/GCP/Azure)

---

## 📚 Dataset

**EMSCAD (Employment Scam Aegean Dataset)**
- Source: [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- 17,880 job postings (~95% legitimate, ~5% fraudulent)
- English-language job listings from real job portals

---

## ⚠️ Disclaimer

This tool is intended to assist users in identifying potentially fraudulent job postings. It is not 100% accurate. Always:
- Research companies independently
- Never pay any fee to apply for a job
- Verify contact information through official channels
- Check reviews on LinkedIn, Glassdoor, or Indeed

---

open by start index.html
