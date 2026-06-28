# enhanced_app.py
# Flask API for Enhanced Fake Job Detection

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
    # Fallback to balanced model
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

# Initialize text processing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


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


def extract_red_flags(text):
    """Enhanced red flag detection with regex patterns"""
    text_lower = text.lower()
    red_flags = []
    
    # Financial red flags - CRITICAL
    financial_patterns = [
        (r'payment\s+required', 'Requests upfront payment'),
        (r'registration\s+fee', 'Requires registration fee'),
        (r'starter\s+kit', 'Requests payment for starter kit'),
        (r'training\s+(materials\s+)?fee', 'Charges for training materials'),
        (r'purchase.{0,20}(laptop|computer|equipment)', 'Requires purchasing equipment'),
        (r'buy.{0,20}software', 'Requires buying software'),
        (r'investment\s+required', 'Requests financial investment'),
        (r'processing\s+fee', 'Charges processing fee'),
        (r'background\s+check\s+fee', 'Charges for background check'),
        (r'reimbursed\s+after', 'Promises later reimbursement'),
        (r'refundable\s+after', 'Claims refundable fees'),
    ]
    
    # Payment method red flags
    payment_patterns = [
        (r'wire\s+transfer', 'Mentions wire transfers'),
        (r'western\s+union', 'Mentions Western Union'),
        (r'money\s+order', 'Requests money orders'),
        (r'cashier.{0,5}check', 'Mentions cashier checks'),
        (r'bitcoin|cryptocurrency', 'Requests cryptocurrency'),
        (r'gift\s+card', 'Mentions gift cards'),
    ]
    
    # Personal information red flags
    info_patterns = [
        (r'\bssn\b', 'Asks for Social Security Number'),
        (r'social\s+security', 'Requests social security info'),
        (r'bank\s+account', 'Requests bank account details'),
        (r'credit\s+card', 'Asks for credit card info'),
        (r'routing\s+number', 'Requests routing number'),
    ]
    
    # Income/promise red flags
    income_patterns = [
        (r'guaranteed\s+(income|\$|\d+)', 'Promises guaranteed income'),
        (r'earn.{0,20}guaranteed', 'Guarantees earnings'),
        (r'get\s+rich', 'Get rich quick scheme'),
        (r'financial\s+freedom', 'Promises financial freedom'),
        (r'no\s+experience.{0,30}high\s+(pay|salary)', 'High pay with no experience'),
    ]
    
    # Urgency red flags
    urgency_patterns = [
        (r'act\s+now', 'Uses urgency tactics'),
        (r'apply\s+immediately', 'Creates false urgency'),
        (r'limited\s+time', 'Limited time pressure'),
        (r'only\s+\d+\s+positions?\s+(left|remaining)', 'False scarcity'),
        (r'hiring\s+immediately', 'Immediate hire pressure'),
        (r'urgent', 'Urgency language'),
    ]
    
    # Contact red flags
    contact_patterns = [
        (r'@(gmail|yahoo|outlook|hotmail)\.com', 'Uses free email service'),
    ]
    
    # Combine all patterns
    all_patterns = (
        financial_patterns + 
        payment_patterns + 
        info_patterns + 
        income_patterns + 
        urgency_patterns + 
        contact_patterns
    )
    
    # Check for patterns using regex
    for pattern, flag in all_patterns:
        if re.search(pattern, text_lower):
            if flag not in red_flags:  # Avoid duplicates
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
        (r'apply\s+(through|via|at)', 'Professional application process'),
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
    
    # Critical red flags that should significantly boost confidence
    critical_flags = [
        'Requests upfront payment',
        'Requires registration fee',
        'Requires purchasing equipment',
        'Requires buying software',
        'Mentions wire transfers',
        'Mentions Western Union',
        'Asks for Social Security Number',
        'Requests bank account details',
    ]
    
    critical_count = sum(1 for flag in red_flags if flag in critical_flags)
    
    # Boost confidence based on critical flags
    if critical_count >= 3:
        adjusted_confidence = min(adjusted_confidence * 1.5, 0.95)
    elif critical_count >= 2:
        adjusted_confidence = min(adjusted_confidence * 1.3, 0.90)
    elif critical_count >= 1:
        adjusted_confidence = min(adjusted_confidence * 1.2, 0.85)
    
    # Also boost based on total red flag count
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
        "message": "Enhanced Fake Job Detector API",
        "version": "2.0",
        "model": model_name if model_name else "Not loaded",
        "endpoints": {
            "/predict": "POST - Predict if a job posting is fake",
            "/model-info": "GET - Get model information",
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


@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    info = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "vectorizer_features": vectorizer.max_features if hasattr(vectorizer, 'max_features') else "Unknown",
        "ngram_range": str(vectorizer.ngram_range) if hasattr(vectorizer, 'ngram_range') else "Unknown"
    }
    
    return jsonify(info)


@app.route('/predict', methods=['POST'])
def predict():
    """Predict if a job posting is fake with enhanced analysis"""
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
                "error": "Model not loaded. Please train and load the model first."
            }), 500
        
        # Clean and preprocess text
        cleaned_text = clean_text(job_description)
        
        if not cleaned_text:
            return jsonify({
                "error": "Job description contains no valid text after preprocessing"
            }), 400
        
        # Vectorize text
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        
        # Get ML confidence score
        ml_confidence = float(probability[1])
        
        # Extract flags
        red_flags_detected = extract_red_flags(job_description)
        positive_indicators_detected = extract_positive_indicators(job_description)
        
        # Calculate adjusted risk score
        adjusted_confidence = calculate_risk_score(ml_confidence, red_flags_detected)
        
        # Determine final prediction based on adjusted confidence
        is_fake = adjusted_confidence > 0.5
        
        # Prepare response
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
            "model_used": model_name,
            "warning": "This is a prediction based on machine learning. Always verify job postings independently." if is_fake else None,
            "risk_adjustment": f"Risk score adjusted from {ml_confidence*100:.1f}% to {adjusted_confidence*100:.1f}% based on {len(red_flags_detected)} red flags" if abs(adjusted_confidence - ml_confidence) > 0.01 else None
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": f"Error processing request: {str(e)}"
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict multiple job postings at once"""
    try:
        data = request.get_json()
        
        if not data or 'job_descriptions' not in data:
            return jsonify({
                "error": "Missing 'job_descriptions' array in request body"
            }), 400
        
        job_descriptions = data['job_descriptions']
        
        if not isinstance(job_descriptions, list):
            return jsonify({
                "error": "'job_descriptions' must be an array"
            }), 400
        
        if model is None or vectorizer is None:
            return jsonify({
                "error": "Model not loaded"
            }), 500
        
        results = []
        
        for idx, job_desc in enumerate(job_descriptions):
            try:
                cleaned_text = clean_text(job_desc)
                if not cleaned_text:
                    results.append({
                        "index": idx,
                        "error": "No valid text after preprocessing"
                    })
                    continue
                
                text_vectorized = vectorizer.transform([cleaned_text])
                prediction = model.predict(text_vectorized)[0]
                probability = model.predict_proba(text_vectorized)[0]
                
                ml_confidence = float(probability[1])
                red_flags = extract_red_flags(job_desc)
                adjusted_confidence = calculate_risk_score(ml_confidence, red_flags)
                is_fake = adjusted_confidence > 0.5
                
                results.append({
                    "index": idx,
                    "prediction": "fake" if is_fake else "legitimate",
                    "confidence": adjusted_confidence,
                    "is_fake": bool(is_fake),
                    "red_flag_count": len(red_flags)
                })
            except Exception as e:
                results.append({
                    "index": idx,
                    "error": str(e)
                })
        
        return jsonify({
            "results": results,
            "total": len(job_descriptions),
            "processed": len([r for r in results if 'error' not in r]),
            "model_used": model_name
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Error processing batch request: {str(e)}"
        }), 500


if __name__ == '__main__':
    print("="*70)
    print("Starting Enhanced Fake Job Detector API...")
    print("="*70)
    if model_name:
        print(f"✓ Model: {model_name}")
    print("✓ API available at http://localhost:5000")
    print("="*70)
    app.run(debug=True, host='0.0.0.0', port=5000)