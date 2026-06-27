# 🔍 Fake Job Posting Detector

> A Machine Learning-powered system to detect fraudulent job listings using NLP and hybrid rule-based detection.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Flask](https://img.shields.io/badge/Flask-3.0.0-green) ![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

With the rise of online recruitment scams, this system automatically identifies fraudulent job postings by analyzing job descriptions using NLP techniques and machine learning models. It flags suspicious postings in real time and provides confidence scores and detailed explanations.

**Academic Project** — UML501, BE Third Year CSE  
Thapar Institute of Engineering and Technology, Patiala  
Submitted by: Tarandeep Kaur (102303394) & Jasman Kaur (102303395)

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

## 📁 Project Structure

```
fakeJobDetection/
├── backend/
│   ├── app.py                          # Main Flask API
│   ├── enhanced_app.py                 # Enhanced API with red flag detection
│   ├── fake_job_detector.py            # Model training script
│   ├── enhanced_fake_job_detector.py   # Enhanced training script
│   ├── best_fake_job_model.pkl         # Best trained model (Naïve Bayes)
│   ├── fake_job_model_balanced.pkl     # Fallback balanced model
│   ├── fake_job_postings.csv           # EMSCAD dataset
│   ├── requirements.txt                # Python dependencies
│   ├── learning_curve.png              # Model learning curve
│   └── roc_curves_comparison.png       # ROC comparison chart
├── frontend/
│   ├── index.html                      # Web UI
│   └── package-lock.json
├── models/                             # Additional model files
├── venv/                               # Virtual environment (not committed)
└── README.md
```

---

## ⚡ Quick Start (Local)

### Prerequisites
- Python 3.11+
- Git

### Step 1: Clone the repository
```bash
git clone https://github.com/jasman5/Fake-job-detector-.git
cd Fake-job-detector-
```

### Step 2: Create and activate virtual environment

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
cd backend
pip install numpy==1.26.2 scipy==1.11.4 scikit-learn==1.3.2
pip install -r requirements.txt
```

### Step 4: Run the backend
```bash
python app.py
```

Backend runs at: `http://localhost:5000`

### Step 5: Open the frontend

Open a new terminal (keep backend running), then:
```bash
# Windows
start ..\frontend\index.html

# Mac
open ../frontend/index.html
```

---

## 🌐 API Endpoints

### `GET /health`
Check if the API is running.
```json
{ "status": "healthy", "model_loaded": true, "model_name": "Naive Bayes" }
```

### `GET /model-info`
Get information about the loaded model.

### `POST /predict`
Analyze a single job posting.

**Request:**
```json
{
  "job_description": "We are hiring a software engineer..."
}
```

**Response:**
```json
{
  "prediction": "fake",
  "confidence": 0.92,
  "is_fake": true,
  "red_flags": ["Requires registration fee", "Charges internship fee - SCAM INDICATOR"],
  "red_flag_count": 2,
  "positive_indicators": ["Contains company website"],
  "model_used": "Naive Bayes",
  "warning": "Always verify job postings independently."
}
```

### `POST /batch_predict`
Analyze multiple job postings at once.

**Request:**
```json
{
  "job_descriptions": ["Job posting 1...", "Job posting 2..."]
}
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Naïve Bayes** ⭐ | **97.54%** | **84.0%** | 60.7% | **0.705** |
| Logistic Regression | 96.28% | 57.8% | **85.5%** | 0.690 |
| Random Forest | 96.64% | 61.2% | 71.7% | 0.665 |

**Best Model: Naïve Bayes** — Highest accuracy and precision with fast inference.

---

## ☁️ Deployment

### Option 1: Deploy on Render (Free)

1. Push your code to GitHub (ensure `venv/` is in `.gitignore`)
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set these settings:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
5. Click **Deploy** — your API will be live at `https://your-app.onrender.com`

### Option 2: Deploy on Railway

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Select your repo
3. Set **Root Directory** to `backend`
4. Add environment variable: `PORT=5000`
5. Railway auto-detects Flask and deploys

### Option 3: Deploy on PythonAnywhere (Free)

1. Sign up at [pythonanywhere.com](https://pythonanywhere.com)
2. Upload your `backend/` folder via Files tab
3. Create a new Web App → Flask → Python 3.11
4. Set source code path to `/home/yourusername/backend`
5. Set WSGI config to point to `app.py`

### After Deployment

Update your `frontend/index.html` — find the API URL variable and replace `localhost:5000` with your deployed URL:
```javascript
// Change this line in index.html
const API_URL = "https://your-deployed-app.onrender.com";
```

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

## 📄 License

MIT License — free to use for educational and personal projects.
