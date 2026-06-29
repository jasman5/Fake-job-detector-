# Flask API for Enhanced Fake Job Detection
# Version 3.0 - Added: Indian scam patterns, /analyze-url endpoint, highlight positions

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# ── URL scraping (used by /analyze-url) ──────────────────────────────────────
try:
    import requests as http_requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    print("⚠ requests/BeautifulSoup not installed – /analyze-url will be disabled.")
    print("  Run: pip install requests beautifulsoup4")

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BEST_MODEL     = os.path.join(BASE_DIR, "best_fake_job_model.pkl")
BALANCED_MODEL = os.path.join(BASE_DIR, "fake_job_model_balanced.pkl")

try:
    with open(BEST_MODEL, "rb") as f:
        model_data = pickle.load(f)
        model      = model_data["model"]
        vectorizer = model_data["vectorizer"]
        model_name = model_data["model_name"]
    print(f"✓ Best model loaded successfully: {model_name}")

except Exception:
    try:
        with open(BALANCED_MODEL, "rb") as f:
            model_data = pickle.load(f)
            model      = model_data["model"]
            vectorizer = model_data["vectorizer"]
            model_name = "Random Forest (Balanced)"
        print(f"✓ Fallback model loaded: {model_name}")

    except Exception as e:
        print(f"Error loading model: {e}")
        model = vectorizer = model_name = None

# ── NLTK setup ────────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text):
    """Clean and preprocess text data."""
    if not text or text.strip() == "":
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = " ".join(text.split())
    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
#  RED FLAG DETECTION  (original + new Indian/modern scam patterns)
# ─────────────────────────────────────────────────────────────────────────────

# Every entry: (regex_pattern, human_label, severity)
# severity: "critical" | "high" | "warning"
RED_FLAG_PATTERNS = [

    # ── Financial / upfront payment ──────────────────────────────────────────
    (r"payment\s+required",                      "Requests upfront payment",                         "critical"),
    (r"registration\s+fee",                      "Requires registration fee",                        "critical"),
    (r"internship\s+fee",                        "Charges internship fee – SCAM INDICATOR",          "critical"),
    (r"enrollment\s+fee",                        "Requires enrollment fee – SCAM INDICATOR",         "critical"),
    (r"one.time\s+(fee|charge|payment)",         "One-time fee required – SCAM INDICATOR",           "critical"),
    (r"confirm\s+your\s+seat",                   "Seat confirmation fee tactic",                     "critical"),
    (r"starter\s+kit",                           "Requests payment for starter kit",                 "critical"),
    (r"training\s+(materials?\s+)?fee",          "Charges for training materials",                   "critical"),
    (r"purchase.{0,20}(laptop|computer|equipment)", "Requires purchasing equipment",                 "critical"),
    (r"buy.{0,20}software",                      "Requires buying software",                         "critical"),
    (r"investment\s+required",                   "Requests financial investment",                    "critical"),
    (r"processing\s+fee",                        "Charges processing fee",                           "critical"),
    (r"background\s+check\s+fee",                "Charges for background check",                     "critical"),
    (r"security\s+deposit",                      "Demands security deposit – SCAM INDICATOR",        "critical"),
    (r"visa\s+processing\s+fee",                 "Charges visa processing fee – SCAM INDICATOR",     "critical"),
    (r"reimbursed\s+after",                      "Promises later reimbursement",                     "high"),
    (r"refundable\s+(deposit|fee|amount)",       "Claims fees are refundable",                       "high"),
    (r"activation\s+(fee|charge)",               "Charges account activation fee",                   "critical"),
    (r"employee\s+id\s+(fee|activation|charge)", "Charges for employee ID",                          "critical"),

    # ── Indian-context scam patterns (NEW) ───────────────────────────────────
    (r"₹\s*\d+.{0,30}(fee|deposit|pay|charge|transfer)",
                                                 "Charges fee in INR – common Indian scam",          "critical"),
    (r"whatsapp\s+(hr|us|now|immediately)",      "Recruits only via WhatsApp – SCAM INDICATOR",      "critical"),
    (r"whatsapp.{0,20}\+91",                     "WhatsApp Indian number recruitment – SCAM",        "critical"),
    (r"\+91.{0,20}(hr|recruit|job|apply|hire)",  "Suspicious Indian phone recruitment",              "high"),
    (r"shortlisted.{0,60}(fee|pay|deposit|transfer|register)",
                                                 "Shortlisting followed by payment demand",          "critical"),
    (r"login\s+credentials.{0,40}(payment|pay|fee)",
                                                 "Credentials held behind payment wall",             "critical"),
    (r"onboard(ing)?.{0,30}(fee|payment|pay)",   "Charges onboarding fee",                          "critical"),
    (r"(lakh|lakhs|lac)\s+per\s+month",          "Unrealistically high Indian salary claim",         "high"),
    (r"earn.{0,20}₹.{0,20}(lakh|lac|per month)", "Guaranteed high INR earnings claim",             "high"),
    (r"no\s+interview\s+required",               "No interview required – SCAM INDICATOR",          "critical"),
    (r"no\s+qualification\s+required",           "No qualification required – suspicious",           "high"),
    (r"immediate\s+(joining|hiring|start)",      "Immediate joining pressure",                       "high"),
    (r"seats?\s+are\s+limited",                  "Artificial seat scarcity",                         "high"),
    (r"profile\s+has\s+been\s+reviewed",         "Unsolicited shortlisting claim",                   "high"),
    (r"accommodation.{0,40}(provided|included|free)",
                                                 "Free accommodation promise – verify carefully",    "warning"),
    (r"flight\s+ticket.{0,30}(provided|included|free)",
                                                 "Free flight ticket promise – verify carefully",    "warning"),

    # ── Payment method red flags ─────────────────────────────────────────────
    (r"wire\s+transfer",                         "Mentions wire transfers",                          "critical"),
    (r"western\s+union",                         "Mentions Western Union",                           "critical"),
    (r"money\s+order",                           "Requests money orders",                            "critical"),
    (r"cashier.{0,5}check",                      "Mentions cashier checks",                          "high"),
    (r"bitcoin|cryptocurrency|crypto\s+pay",     "Requests cryptocurrency payment",                  "critical"),
    (r"gift\s+card",                             "Requests gift card payment",                       "critical"),
    (r"(zelle|cash\s*app|paytm|upi)\s+(pay|transfer|send)",
                                                 "Informal payment app for job fees – SCAM",        "critical"),
    (r"neft|rtgs|imps.{0,30}(fee|deposit|pay)",  "Requests bank transfer for fees",                 "critical"),

    # ── Personal information red flags ───────────────────────────────────────
    (r"\bssn\b",                                 "Asks for Social Security Number",                  "critical"),
    (r"social\s+security",                       "Requests social security info",                    "critical"),
    (r"bank\s+account\s+(number|details|info)",  "Requests bank account details",                    "critical"),
    (r"credit\s+card",                           "Asks for credit card info",                        "critical"),
    (r"routing\s+number",                        "Requests routing number",                          "critical"),
    (r"aadhar|aadhaar",                          "Requests Aadhaar number early – verify intent",   "warning"),
    (r"pan\s+card\s+(fee|required\s+upfront)",   "PAN card demanded with fee",                       "high"),

    # ── Income / promise red flags ───────────────────────────────────────────
    (r"guaranteed\s+(income|\$|₹|\d+)",          "Promises guaranteed income",                       "high"),
    (r"earn.{0,20}guaranteed",                   "Guarantees earnings",                              "high"),
    (r"get\s+rich",                              "Get-rich-quick scheme language",                   "high"),
    (r"financial\s+freedom",                     "Promises financial freedom",                       "warning"),
    (r"no\s+experience.{0,30}high\s+(pay|salary)", "High pay with no experience",                   "high"),
    (r"\$\s*\d{4,}.{0,20}(per\s+(hour|day)|guaranteed)",
                                                 "Unrealistically high hourly/daily pay",            "high"),
    (r"earn.{0,20}\$\s*\d{3,}.{0,20}(hour|day|week)",
                                                 "Suspiciously high pay rate",                       "high"),
    (r"work\s+(2|3|two|three)\s+hours?.{0,20}\$\s*\d{3,}",
                                                 "Extreme pay-for-minimal-hours claim",              "high"),

    # ── Reshipping / mule scams (NEW) ────────────────────────────────────────
    (r"receive.{0,30}package.{0,30}(reship|forward|send)",
                                                 "Reshipping/package mule scam",                    "critical"),
    (r"reship(ping)?\s+package",                 "Reshipping scam indicator",                        "critical"),
    (r"package\s+handler.{0,30}home\s+address",  "Home address package handler – scam",              "critical"),

    # ── Urgency red flags ────────────────────────────────────────────────────
    (r"act\s+now",                               "Uses urgency tactics",                             "high"),
    (r"apply\s+immediately",                     "Creates false urgency",                            "warning"),
    (r"limited\s+time",                          "Limited time pressure",                            "warning"),
    (r"only\s+\d+\s+positions?\s+(left|remaining|available)",
                                                 "False scarcity of positions",                      "high"),
    (r"hiring\s+immediately",                    "Immediate hire pressure",                          "warning"),
    (r"\burgent\b",                              "Urgency language",                                 "warning"),
    (r"confirm\s+your\s+(seat|enrollment)",      "Pressures to confirm seat",                        "high"),
    (r"click\s+here\s+to\s+register",           "Aggressive registration push",                     "warning"),
    (r"complete\s+(your\s+)?enrollment",         "Forced enrollment tactic",                         "high"),
    (r"failure\s+to\s+(pay|respond|submit).{0,40}cancel", "Threat of cancellation if no payment",  "critical"),
    (r"within\s+24\s+hours",                     "24-hour payment pressure",                         "high"),
    (r"joining\s+bonus.{0,30}(pay|fee|deposit)", "Joining bonus tied to upfront fee",               "critical"),

    # ── Contact red flags ────────────────────────────────────────────────────
    (r"@(gmail|yahoo|outlook|hotmail|rediffmail)\.com",
                                                 "Uses free personal email domain",                  "high"),
    (r"no\s+resume\s+required",                  "No resume required – unprofessional",              "high"),
    (r"hiring\s+is\s+100%\s+guaranteed",         "Guarantees hiring – SCAM INDICATOR",              "critical"),
]


def extract_red_flags(text):
    """
    Return list of human-readable red flag strings (original behaviour preserved).
    Deduplicates on label.
    """
    text_lower = text.lower()
    seen   = set()
    flags  = []

    for pattern, label, _ in RED_FLAG_PATTERNS:
        if label in seen:
            continue
        if re.search(pattern, text_lower):
            flags.append(label)
            seen.add(label)

    # Caps ratio
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.10:
        flags.append("Excessive use of capital letters")

    # Exclamation marks
    excl = text.count("!")
    if excl > 5:
        flags.append(f"Excessive exclamation marks ({excl} found)")

    return flags


def extract_highlights(text):
    """
    NEW – Return list of highlight objects so the frontend can colour-code
    suspicious spans.  Each item: {start, end, text, label, severity}
    """
    highlights = []
    seen_spans = []   # avoid overlapping highlights

    def overlaps(s, e):
        return any(not (e <= es or s >= ee) for es, ee in seen_spans)

    for pattern, label, severity in RED_FLAG_PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            s, e = m.start(), m.end()
            if not overlaps(s, e):
                highlights.append({
                    "start":    s,
                    "end":      e,
                    "text":     text[s:e],
                    "label":    label,
                    "severity": severity,
                })
                seen_spans.append((s, e))

    # Sort by position so the frontend can render left-to-right
    highlights.sort(key=lambda h: h["start"])
    return highlights


# ─────────────────────────────────────────────────────────────────────────────
#  POSITIVE INDICATORS  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def extract_positive_indicators(text):
    text_lower = text.lower()
    indicators = []

    positive_patterns = [
        (r"(https?://|www\.)\S+\.(com|org|net|edu|in)",   "Contains company website"),
        (r"benefits",                                      "Lists employee benefits"),
        (r"(salary|compensation|ctc|lpa).{0,50}(\$|₹|\d+)", "Provides salary information"),
        (r"(responsibilities|duties)\s*:",                 "Clear job responsibilities"),
        (r"(qualifications|requirements)\s*:",             "Specific qualifications listed"),
        (r"apply\s+(through|via|at)",                     "Professional application process"),
        (r"equal\s+opportunity",                           "Equal opportunity employer"),
        (r"(bachelor|master|degree|b\.?tech|m\.?tech)",   "Education requirements specified"),
        (r"\d+\+?\s+years?\s+(of\s+)?experience",         "Experience requirements clear"),
        (r"(health|dental|vision)\s+insurance",            "Health benefits mentioned"),
        (r"401\s*k|pf|provident\s+fund",                  "Retirement/PF benefits mentioned"),
        (r"job\s+(id|reference|ref)\s*[:#]?\s*\w+",       "Job ID/reference provided"),
        (r"careers?\.(com|in|net|org)|linkedin\.com",     "Listed on professional job platform"),
        (r"(esop|stock\s+option|equity)",                 "Equity/ESOP offered"),
    ]

    for pattern, indicator in positive_patterns:
        if re.search(pattern, text_lower):
            if indicator not in indicators:
                indicators.append(indicator)

    return indicators


# ─────────────────────────────────────────────────────────────────────────────
#  RISK SCORE  (original logic + extra critical flags)
# ─────────────────────────────────────────────────────────────────────────────

CRITICAL_FLAGS = {
    "Requests upfront payment",
    "Requires registration fee",
    "Charges internship fee – SCAM INDICATOR",
    "Requires enrollment fee – SCAM INDICATOR",
    "One-time fee required – SCAM INDICATOR",
    "Seat confirmation fee tactic",
    "Pressures to confirm seat",
    "Requires purchasing equipment",
    "Requires buying software",
    "Mentions wire transfers",
    "Mentions Western Union",
    "Asks for Social Security Number",
    "Requests bank account details",
    "Demands security deposit – SCAM INDICATOR",
    "Charges visa processing fee – SCAM INDICATOR",
    "Charges account activation fee",
    "Charges for employee ID",
    "Charges fee in INR – common Indian scam",
    "Recruits only via WhatsApp – SCAM INDICATOR",
    "WhatsApp Indian number recruitment – SCAM",
    "Shortlisting followed by payment demand",
    "Credentials held behind payment wall",
    "No interview required – SCAM INDICATOR",
    "Reshipping/package mule scam",
    "Reshipping scam indicator",
    "Home address package handler – scam",
    "Threat of cancellation if no payment",
    "Joining bonus tied to upfront fee",
    "Guarantees hiring – SCAM INDICATOR",
    "Requests cryptocurrency payment",
    "Requests gift card payment",
    "Informal payment app for job fees – SCAM",
}


def calculate_risk_score(ml_confidence, red_flags):
    adjusted = ml_confidence

    critical_count = sum(1 for f in red_flags if f in CRITICAL_FLAGS)
    total_flags    = len(red_flags)

    if critical_count >= 1:
        adjusted = max(adjusted, 0.75)
    if critical_count >= 2:
        adjusted = max(adjusted, 0.88)
    if critical_count >= 3:
        adjusted = max(adjusted, 0.95)

    if total_flags >= 8:
        adjusted = max(adjusted, 0.92)
    elif total_flags >= 5:
        adjusted = max(adjusted, 0.80)

    return adjusted


# ─────────────────────────────────────────────────────────────────────────────
#  URL SCRAPING HELPER  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def scrape_job_from_url(url):
    """
    Scrape a job posting page and return a dict:
    { title, company, location, job_text, url }
    Raises ValueError on failure.
    """
    if not SCRAPING_AVAILABLE:
        raise ValueError("Scraping libraries not installed (pip install requests beautifulsoup4)")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = http_requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        raise ValueError(f"Could not fetch URL: {e}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # ── Remove boilerplate tags ──────────────────────────────────────────────
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "iframe", "noscript"]):
        tag.decompose()

    # ── Try to extract structured fields ────────────────────────────────────
    def meta(name):
        t = (soup.find("meta", property=name) or
             soup.find("meta", attrs={"name": name}))
        return t["content"].strip() if t and t.get("content") else "Not found"

    title   = meta("og:title")
    company = meta("og:site_name")

    if title == "Not found":
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else "Not found"

    # Location: look for common selectors
    location = "Not found"
    for sel in [
        "[data-testid='job-location']", ".job-location",
        ".location", "[class*='location']", "[itemprop='jobLocation']"
    ]:
        el = soup.select_one(sel)
        if el:
            location = el.get_text(strip=True)
            break

    # ── Extract main body text ───────────────────────────────────────────────
    # Prefer known job-description containers
    body_text = ""
    for sel in [
        "[data-testid='job-description']",
        ".job-description", ".jobDescription",
        "#job-description", "[class*='description']",
        "article", "main",
    ]:
        el = soup.select_one(sel)
        if el:
            body_text = el.get_text(separator="\n", strip=True)
            break

    # Fallback to full body
    if not body_text:
        body = soup.find("body")
        body_text = body.get_text(separator="\n", strip=True) if body else ""

    # Clean up excessive whitespace
    lines     = [l.strip() for l in body_text.splitlines() if l.strip()]
    job_text  = "\n".join(lines)

    if len(job_text) < 50:
        raise ValueError("Could not extract meaningful job text from the page.")

    return {
        "title":    title,
        "company":  company,
        "location": location,
        "job_text": job_text[:8000],   # cap to avoid huge payloads
        "url":      url,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({
        "message": "Enhanced Fake Job Detector API",
        "version": "3.0",
        "model":   model_name if model_name else "Not loaded",
        "endpoints": {
            "/predict":      "POST – Predict if a job posting is fake",
            "/analyze-url":  "POST – Scrape a URL and predict",
            "/batch_predict":"POST – Predict multiple job postings",
            "/model-info":   "GET  – Model information",
            "/health":       "GET  – Health check",
        },
    })


@app.route("/health")
def health():
    return jsonify({
        "status":           "healthy",
        "model_loaded":     model is not None,
        "model_name":       model_name or "None",
        "scraping_enabled": SCRAPING_AVAILABLE,
    })


@app.route("/model-info")
def model_info():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify({
        "model_name":         model_name,
        "model_type":         type(model).__name__,
        "vectorizer_features": getattr(vectorizer, "max_features", "Unknown"),
        "ngram_range":        str(getattr(vectorizer, "ngram_range", "Unknown")),
    })


# ── Core prediction helper ────────────────────────────────────────────────────

def _run_prediction(job_description):
    """
    Shared logic used by /predict and /analyze-url.
    Returns a result dict or raises ValueError.
    """
    if not job_description or not job_description.strip():
        raise ValueError("Job description cannot be empty")

    if model is None or vectorizer is None:
        raise ValueError("Model not loaded. Please train and load the model first.")

    cleaned = clean_text(job_description)
    if not cleaned:
        raise ValueError("Job description contains no valid text after preprocessing")

    text_vec   = vectorizer.transform([cleaned])
    prediction = model.predict(text_vec)[0]
    proba      = model.predict_proba(text_vec)[0]

    ml_confidence = float(proba[1])

    red_flags_detected  = extract_red_flags(job_description)
    positive_indicators = extract_positive_indicators(job_description)
    highlights          = extract_highlights(job_description)   # NEW
    adjusted_confidence = calculate_risk_score(ml_confidence, red_flags_detected)
    is_fake             = adjusted_confidence > 0.5

    return {
        "prediction":              "fake" if is_fake else "legitimate",
        "confidence":              adjusted_confidence,
        "is_fake":                 bool(is_fake),
        "ml_confidence":           ml_confidence,
        "probability_fake":        float(proba[1]),
        "probability_legitimate":  float(proba[0]),
        "red_flags":               red_flags_detected,
        "red_flag_count":          len(red_flags_detected),
        "positive_indicators":     positive_indicators,
        "positive_indicator_count": len(positive_indicators),
        "highlights":              highlights,                  # NEW
        "model_used":              model_name,
        "warning": (
            "This is a prediction based on machine learning. Always verify job postings independently."
            if is_fake else None
        ),
        "risk_adjustment": (
            f"Risk score adjusted from {ml_confidence*100:.1f}% to "
            f"{adjusted_confidence*100:.1f}% based on {len(red_flags_detected)} red flags"
            if abs(adjusted_confidence - ml_confidence) > 0.01 else None
        ),
    }


@app.route("/predict", methods=["POST"])
def predict():
    """Predict if a job posting is fake with enhanced analysis."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        # Accept multiple field names so old frontends still work
        job_description = (
            data.get("job_description") or
            data.get("description") or
            data.get("text") or
            ""
        )

        if not job_description:
            return jsonify({"error": "Missing job description. Use key: 'job_description'"}), 400

        result = _run_prediction(job_description)
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500


@app.route("/analyze-url", methods=["POST"])
def analyze_url():
    """
    NEW – Scrape a job posting URL, then run prediction on extracted text.
    Request body: { "url": "https://..." }
    """
    try:
        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "Missing 'url' in request body"}), 400

        url = data["url"].strip()
        if not url.startswith(("http://", "https://")):
            return jsonify({"error": "URL must start with http:// or https://"}), 400

        # Scrape
        scraped = scrape_job_from_url(url)

        # Predict on extracted text
        result = _run_prediction(scraped["job_text"])

        # Merge scrape metadata into response
        result["scraped_data"] = {
            "title":    scraped["title"],
            "company":  scraped["company"],
            "location": scraped["location"],
            "url":      scraped["url"],
        }
        result["job_text"] = scraped["job_text"]

        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Error processing URL: {str(e)}"}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Predict multiple job postings at once."""
    try:
        data = request.get_json()
        if not data or "job_descriptions" not in data:
            return jsonify({"error": "Missing 'job_descriptions' array in request body"}), 400

        job_descriptions = data["job_descriptions"]
        if not isinstance(job_descriptions, list):
            return jsonify({"error": "'job_descriptions' must be an array"}), 400

        if model is None or vectorizer is None:
            return jsonify({"error": "Model not loaded"}), 500

        results = []
        for idx, job_desc in enumerate(job_descriptions):
            try:
                res = _run_prediction(job_desc)
                results.append({
                    "index":          idx,
                    "prediction":     res["prediction"],
                    "confidence":     res["confidence"],
                    "is_fake":        res["is_fake"],
                    "red_flag_count": res["red_flag_count"],
                    "red_flags":      res["red_flags"],
                })
            except Exception as e:
                results.append({"index": idx, "error": str(e)})

        return jsonify({
            "results":   results,
            "total":     len(job_descriptions),
            "processed": len([r for r in results if "error" not in r]),
            "model_used": model_name,
        })

    except Exception as e:
        return jsonify({"error": f"Error processing batch request: {str(e)}"}), 500


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("Starting Enhanced Fake Job Detector API v3.0 ...")
    print("=" * 70)
    if model_name:
        print(f"✓ Model : {model_name}")
    if SCRAPING_AVAILABLE:
        print("✓ URL scraping : ENABLED")
    else:
        print("⚠ URL scraping : DISABLED  (pip install requests beautifulsoup4)")
    print("✓ API available at http://localhost:5000")
    print("=" * 70)
    app.run(debug=True, host="0.0.0.0", port=5000)  