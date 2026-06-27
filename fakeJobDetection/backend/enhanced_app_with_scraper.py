# enhanced_app_with_scraper.py
# Flask API with URL Scraping Feature

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)

# Load the best trained model
try:
    with open('best_fake_job_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        model_name = model_data['model_name']
    print(f"✓ Best model loaded successfully: {model_name}")
except:
    try:
        with open('fake_job_model_balanced.pkl', 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            vectorizer = model_data['vectorizer']
            model_name = "Random Forest (Balanced)"
        print(f"✓ Fallback model loaded: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        vectorizer = None
        model_name = None

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def scrape_job_posting(url):
    """
    Scrape job posting from URL
    Supports: LinkedIn, Indeed, Glassdoor, and generic job sites
    """
    try:
        # Set headers to mimic browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make request with timeout
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check for 404/error pages
        page_title = soup.title.string.lower() if soup.title else ''
        page_text = soup.get_text().lower()
        
        error_indicators = [
            '404', 'not found', 'page not found', 'page doesn\'t exist',
            'page cannot be found', 'error 404', 'no longer available',
            'job posting has expired', 'position has been filled',
            'this job is no longer accepting applications'
        ]
        
        if any(indicator in page_title or indicator in page_text[:500] for indicator in error_indicators):
            return {
                'success': False, 
                'error': 'This job posting appears to be expired, removed, or does not exist. Please verify the URL.'
            }
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Check if we got meaningful content
        if len(text.strip()) < 100:
            return {
                'success': False,
                'error': 'Could not extract sufficient content from the page. It may be empty or require authentication.'
            }
        
        # Try to extract specific elements (works for most job sites)
        job_data = {
            'title': '',
            'company': '',
            'location': '',
            'description': text,
            'url': url
        }
        
        # Try to find job title
        title_selectors = ['h1', '.job-title', '.jobTitle', '[class*="title"]']
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem and title_elem.get_text().strip():
                job_data['title'] = title_elem.get_text().strip()
                break
        
        # Try to find company name
        company_selectors = ['.company', '[class*="company"]', '[class*="employer"]']
        for selector in company_selectors:
            company_elem = soup.select_one(selector)
            if company_elem and company_elem.get_text().strip():
                job_data['company'] = company_elem.get_text().strip()
                break
        
        # Try to find location
        location_selectors = ['.location', '[class*="location"]']
        for selector in location_selectors:
            location_elem = soup.select_one(selector)
            if location_elem and location_elem.get_text().strip():
                job_data['location'] = location_elem.get_text().strip()
                break
        
        return {
            'success': True,
            'job_data': job_data,
            'raw_text': text[:5000]  # Limit to first 5000 chars
        }
        
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timed out. Please try again.'}
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f'Failed to fetch URL: {str(e)}'}
    except Exception as e:
        return {'success': False, 'error': f'Error scraping page: {str(e)}'}


def clean_text(text):
    """Clean and preprocess text data"""
    if not text or text.strip() == "":
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
              if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)


def highlight_suspicious_patterns(text):
    """
    Identify and return suspicious patterns with their positions
    Returns list of {text, type, severity}
    """
    highlights = []
    
    # Critical red flags (high severity)
    critical_patterns = [
        (r'payment\s+required', 'upfront payment', 'critical'),
        (r'registration\s+fee', 'registration fee', 'critical'),
        (r'wire\s+transfer', 'wire transfer', 'critical'),
        (r'western\s+union', 'Western Union', 'critical'),
        (r'send\s+money', 'money request', 'critical'),
        (r'bank\s+account\s+details', 'bank details request', 'critical'),
        (r'\bssn\b', 'SSN request', 'critical'),
    ]
    
    # High severity red flags
    high_patterns = [
        (r'guaranteed\s+(income|\$|\d+)', 'guaranteed income', 'high'),
        (r'act\s+now', 'urgency tactic', 'high'),
        (r'limited\s+time', 'false urgency', 'high'),
        (r'no\s+experience\s+(needed|required)', 'no experience claim', 'high'),
        (r'get\s+rich', 'get rich scheme', 'high'),
        (r'purchase.{0,20}(laptop|equipment)', 'equipment purchase', 'high'),
    ]
    
    # Medium severity warnings
    medium_patterns = [
        (r'@(gmail|yahoo|hotmail|outlook)\.com', 'free email', 'medium'),
        (r'urgent', 'urgency language', 'medium'),
        (r'apply\s+immediately', 'pressure tactic', 'medium'),
        (r'work\s+from\s+home', 'work from home', 'medium'),
    ]
    
    all_patterns = critical_patterns + high_patterns + medium_patterns
    
    for pattern, label, severity in all_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            highlights.append({
                'start': match.start(),
                'end': match.end(),
                'text': text[match.start():match.end()],
                'label': label,
                'severity': severity
            })
    
    # Sort by position
    highlights.sort(key=lambda x: x['start'])
    
    return highlights


def extract_red_flags(text):
    """Enhanced red flag detection with regex patterns"""
    text_lower = text.lower()
    red_flags = []
    
    financial_patterns = [
        (r'payment\s+required', 'Requests upfront payment'),
        (r'registration\s+fee', 'Requires registration fee'),
        (r'starter\s+kit', 'Requests payment for starter kit'),
        (r'training\s+(materials\s+)?fee', 'Charges for training materials'),
        (r'purchase.{0,20}(laptop|computer|equipment)', 'Requires purchasing equipment'),
        (r'buy.{0,20}software', 'Requires buying software'),
        (r'investment\s+required', 'Requests financial investment'),
        (r'processing\s+fee', 'Charges processing fee'),
        (r'reimbursed\s+after', 'Promises later reimbursement'),
    ]
    
    payment_patterns = [
        (r'wire\s+transfer', 'Mentions wire transfers'),
        (r'western\s+union', 'Mentions Western Union'),
        (r'money\s+order', 'Requests money orders'),
        (r'bitcoin|cryptocurrency', 'Requests cryptocurrency'),
    ]
    
    info_patterns = [
        (r'\bssn\b', 'Asks for Social Security Number'),
        (r'social\s+security', 'Requests social security info'),
        (r'bank\s+account', 'Requests bank account details'),
        (r'credit\s+card', 'Asks for credit card info'),
    ]
    
    income_patterns = [
        (r'guaranteed\s+(income|\$|\d+)', 'Promises guaranteed income'),
        (r'get\s+rich', 'Get rich quick scheme'),
        (r'no\s+experience.{0,30}high\s+(pay|salary)', 'High pay with no experience'),
    ]
    
    urgency_patterns = [
        (r'act\s+now', 'Uses urgency tactics'),
        (r'apply\s+immediately', 'Creates false urgency'),
        (r'limited\s+time', 'Limited time pressure'),
        (r'only\s+\d+\s+positions?\s+(left|remaining)', 'False scarcity'),
        (r'urgent', 'Urgency language'),
    ]
    
    contact_patterns = [
        (r'@(gmail|yahoo|outlook|hotmail)\.com', 'Uses free email service'),
    ]
    
    all_patterns = (
        financial_patterns + payment_patterns + info_patterns + 
        income_patterns + urgency_patterns + contact_patterns
    )
    
    for pattern, flag in all_patterns:
        if re.search(pattern, text_lower):
            if flag not in red_flags:
                red_flags.append(flag)
    
    # Check for excessive caps
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.1:
        red_flags.append('Excessive use of capital letters')
    
    # Check for excessive exclamation marks
    exclamation_count = text.count('!')
    if exclamation_count > 5:
        red_flags.append(f'Excessive exclamation marks ({exclamation_count} found)')
    
    return red_flags


def extract_positive_indicators(text):
    """Identify positive indicators in job posting"""
    text_lower = text.lower()
    indicators = []
    
    positive_patterns = [
        (r'(https?://|www\.)\S+\.(com|org|net|edu)', 'Contains company website'),
        (r'benefits', 'Lists employee benefits'),
        (r'(salary|compensation).{0,50}\$\d+', 'Provides salary information'),
        (r'(responsibilities|duties):', 'Clear job responsibilities'),
        (r'(qualifications|requirements):', 'Specific qualifications listed'),
        (r'equal\s+opportunity', 'Equal opportunity employer'),
        (r'(bachelor|master|degree)', 'Education requirements specified'),
        (r'\d+\+?\s+years?\s+(of\s+)?experience', 'Experience requirements clear'),
        (r'(health|dental|vision)\s+insurance', 'Health benefits mentioned'),
        (r'401\s*k', 'Retirement benefits mentioned'),
    ]
    
    for pattern, indicator in positive_patterns:
        if re.search(pattern, text_lower):
            if indicator not in indicators:
                indicators.append(indicator)
    
    return indicators


def calculate_risk_score(ml_confidence, red_flags):
    """Calculate adjusted risk score based on red flags"""
    adjusted_confidence = ml_confidence
    
    critical_flags = [
        'Requests upfront payment',
        'Requires registration fee',
        'Requires purchasing equipment',
        'Mentions wire transfers',
        'Mentions Western Union',
        'Asks for Social Security Number',
        'Requests bank account details',
    ]
    
    critical_count = sum(1 for flag in red_flags if flag in critical_flags)
    
    if critical_count >= 3:
        adjusted_confidence = min(adjusted_confidence * 1.5, 0.95)
    elif critical_count >= 2:
        adjusted_confidence = min(adjusted_confidence * 1.3, 0.90)
    elif critical_count >= 1:
        adjusted_confidence = min(adjusted_confidence * 1.2, 0.85)
    
    total_flags = len(red_flags)
    if total_flags >= 8:
        adjusted_confidence = min(adjusted_confidence * 1.4, 0.95)
    elif total_flags >= 5:
        adjusted_confidence = min(adjusted_confidence * 1.2, 0.90)
    
    return adjusted_confidence


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Enhanced Fake Job Detector API with URL Scraping",
        "version": "3.0",
        "model": model_name if model_name else "Not loaded",
        "features": ["text_analysis", "url_scraping", "visual_highlights"],
        "endpoints": {
            "/predict": "POST - Analyze job posting text",
            "/analyze-url": "POST - Analyze job posting from URL",
            "/health": "GET - Check API health"
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_name if model_name else "None"
    })


@app.route('/analyze-url', methods=['POST'])
def analyze_url():
    """NEW: Analyze job posting from URL"""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({
                "error": "Missing 'url' in request body"
            }), 400
        
        url = data['url']
        
        # Validate URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return jsonify({
                    "error": "Invalid URL format. Please include http:// or https://"
                }), 400
        except:
            return jsonify({
                "error": "Invalid URL"
            }), 400
        
        # Scrape the URL
        print(f"Scraping URL: {url}")
        scrape_result = scrape_job_posting(url)
        
        if not scrape_result['success']:
            return jsonify({
                "error": scrape_result['error'],
                "suggestion": "Try copying the job description text directly instead"
            }), 400
        
        job_data = scrape_result['job_data']
        job_text = job_data['description']
        
        if not job_text or len(job_text.strip()) < 50:
            return jsonify({
                "error": "Could not extract enough text from URL. The page might be JavaScript-heavy or require login.",
                "suggestion": "Try copying the job description text directly"
            }), 400
        
        # Clean and analyze
        cleaned_text = clean_text(job_text)
        
        if model is None or vectorizer is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Vectorize and predict
        text_vectorized = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        ml_confidence = float(probability[1])
        red_flags_detected = extract_red_flags(job_text)
        positive_indicators_detected = extract_positive_indicators(job_text)
        highlights = highlight_suspicious_patterns(job_text)
        
        adjusted_confidence = calculate_risk_score(ml_confidence, red_flags_detected)
        is_fake = adjusted_confidence > 0.5
        
        response = {
            "prediction": "fake" if is_fake else "legitimate",
            "confidence": adjusted_confidence,
            "is_fake": bool(is_fake),
            "ml_confidence": ml_confidence,
            "probability_fake": float(probability[1]),
            "probability_legitimate": float(probability[0]),
            "red_flags": red_flags_detected,
            "red_flag_count": len(red_flags_detected),
            "positive_indicators": positive_indicators_detected,
            "positive_indicator_count": len(positive_indicators_detected),
            "highlights": highlights,
            "scraped_data": {
                "title": job_data.get('title', 'Not found'),
                "company": job_data.get('company', 'Not found'),
                "location": job_data.get('location', 'Not found'),
                "url": url
            },
            "job_text": job_text[:2000],  # First 2000 chars for display
            "model_used": model_name,
            "warning": "This is a prediction based on machine learning. Always verify job postings independently." if is_fake else None
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": f"Error processing URL: {str(e)}"
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Analyze job posting text with visual highlights"""
    try:
        data = request.get_json()
        
        if not data or 'job_description' not in data:
            return jsonify({
                "error": "Missing 'job_description' in request body"
            }), 400
        
        job_description = data['job_description']
        
        if not job_description or job_description.strip() == "":
            return jsonify({
                "error": "Job description cannot be empty"
            }), 400
        
        if model is None or vectorizer is None:
            return jsonify({
                "error": "Model not loaded"
            }), 500
        
        cleaned_text = clean_text(job_description)
        
        if not cleaned_text:
            return jsonify({
                "error": "Job description contains no valid text after preprocessing"
            }), 400
        
        text_vectorized = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        ml_confidence = float(probability[1])
        red_flags_detected = extract_red_flags(job_description)
        positive_indicators_detected = extract_positive_indicators(job_description)
        highlights = highlight_suspicious_patterns(job_description)
        
        adjusted_confidence = calculate_risk_score(ml_confidence, red_flags_detected)
        is_fake = adjusted_confidence > 0.5
        
        response = {
            "prediction": "fake" if is_fake else "legitimate",
            "confidence": adjusted_confidence,
            "is_fake": bool(is_fake),
            "ml_confidence": ml_confidence,
            "probability_fake": float(probability[1]),
            "probability_legitimate": float(probability[0]),
            "red_flags": red_flags_detected,
            "red_flag_count": len(red_flags_detected),
            "positive_indicators": positive_indicators_detected,
            "positive_indicator_count": len(positive_indicators_detected),
            "highlights": highlights,
            "model_used": model_name,
            "warning": "This is a prediction based on machine learning. Always verify job postings independently." if is_fake else None
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": f"Error processing request: {str(e)}"
        }), 500


if __name__ == '__main__':
    print("="*70)
    print("Starting Enhanced Fake Job Detector API with URL Scraping...")
    print("="*70)
    if model_name:
        print(f"✓ Model: {model_name}")
    print("✓ Features: URL Scraping, Visual Highlights")
    print("✓ API available at http://localhost:5000")
    print("="*70)
    app.run(debug=True, host='0.0.0.0', port=5000)