# fake_job_detector.py
# Complete ML Pipeline for Fake Job Posting Detection

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class FakeJobDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and lemmatize
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                  if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def prepare_features(self, df):
        """Prepare features from the dataset"""
        # Combine text features
        df['combined_text'] = (
            df['title'].fillna('') + ' ' +
            df['location'].fillna('') + ' ' +
            df['department'].fillna('') + ' ' +
            df['company_profile'].fillna('') + ' ' +
            df['description'].fillna('') + ' ' +
            df['requirements'].fillna('') + ' ' +
            df['benefits'].fillna('')
        )
        
        # Clean the combined text
        df['cleaned_text'] = df['combined_text'].apply(self.clean_text)
        
        # Create additional features
        df['text_length'] = df['combined_text'].str.len()
        df['has_company_logo'] = df['has_company_logo'].fillna(0)
        df['has_questions'] = df['has_questions'].fillna(0)
        df['telecommuting'] = df['telecommuting'].fillna(0)
        df['has_salary_range'] = (~df['salary_range'].isna()).astype(int)
        
        return df
    
    def train(self, X_train, y_train, model_type='random_forest', handle_imbalance='class_weight'):
        """Train the model with imbalanced data handling"""
        print("Vectorizing text data...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        print(f"Training {model_type} model with {handle_imbalance} for imbalanced data...")
        
        # Handle imbalanced dataset
        if handle_imbalance == 'smote':
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_tfidf, y_train = smote.fit_resample(X_train_tfidf, y_train)
            print(f"After SMOTE - Class distribution: {np.bincount(y_train)}")
        
        if model_type == 'random_forest':
            # Use class_weight='balanced' to give more importance to minority class
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced' if handle_imbalance == 'class_weight' else None,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced' if handle_imbalance == 'class_weight' else None,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=0.1)
        
        self.model.fit(X_train_tfidf, y_train)
        print("Model training completed!")
        
    def predict(self, X_test):
        """Make predictions"""
        X_test_tfidf = self.vectorizer.transform(X_test)
        predictions = self.model.predict(X_test_tfidf)
        probabilities = self.model.predict_proba(X_test_tfidf)
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model with focus on minority class performance"""
        predictions, probabilities = self.predict(X_test)
        
        print("\n=== Model Evaluation ===")
        print(f"Overall Accuracy: {accuracy_score(y_test, predictions):.4f}")
        
        # Calculate metrics for both classes
        from sklearn.metrics import precision_score, recall_score, f1_score
        print(f"\nFake Job Detection (Minority Class) Metrics:")
        print(f"Precision: {precision_score(y_test, predictions, pos_label=1):.4f}")
        print(f"Recall: {recall_score(y_test, predictions, pos_label=1):.4f}")
        print(f"F1-Score: {f1_score(y_test, predictions, pos_label=1):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, 
                                    target_names=['Legitimate', 'Fake']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        print(f"\nTrue Negatives: {cm[0][0]} | False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]} | True Positives: {cm[1][1]}")
        
        # Calculate specificity and sensitivity
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"\nSensitivity (Recall for Fake): {sensitivity:.4f}")
        print(f"Specificity (Recall for Legitimate): {specificity:.4f}")
        
        return predictions, probabilities
    
    def save_model(self, filepath='fake_job_model.pkl'):
        """Save the trained model and vectorizer"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='fake_job_model.pkl'):
        """Load a trained model and vectorizer"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
        print(f"Model loaded from {filepath}")


def main():
    """Main training pipeline with imbalanced data handling"""
    # Load dataset (you'll need to download this from Kaggle)
    # Dataset: Employment Scam Aegean Dataset (EMSCAD)
    # URL: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
    
    print("Loading dataset...")
    df = pd.read_csv('fake_job_postings.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\n=== Class Distribution ===")
    print(f"Fraudulent jobs: {df['fraudulent'].sum()} ({df['fraudulent'].mean()*100:.2f}%)")
    print(f"Legitimate jobs: {(~df['fraudulent'].astype(bool)).sum()} ({(~df['fraudulent'].astype(bool)).mean()*100:.2f}%)")
    print(f"Imbalance Ratio: {(~df['fraudulent'].astype(bool)).sum() / df['fraudulent'].sum():.2f}:1")
    
    # Initialize detector
    detector = FakeJobDetector()
    
    # Prepare features
    print("\nPreparing features...")
    df = detector.prepare_features(df)
    
    # Split data with stratification to maintain class distribution
    X = df['cleaned_text']
    y = df['fraudulent']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Training set - Fake: {y_train.sum()}, Legitimate: {(~y_train.astype(bool)).sum()}")
    print(f"Test set size: {len(X_test)}")
    print(f"Test set - Fake: {y_test.sum()}, Legitimate: {(~y_test.astype(bool)).sum()}")
    
    # Train model with different imbalance handling techniques
    print("\n" + "="*60)
    print("APPROACH 1: Class Weight Balancing (Recommended)")
    print("="*60)
    detector.train(X_train, y_train, model_type='random_forest', handle_imbalance='class_weight')
    detector.evaluate(X_test, y_test)
    detector.save_model('fake_job_model_balanced.pkl')
    
    # Optional: Try SMOTE for comparison
    print("\n" + "="*60)
    print("APPROACH 2: SMOTE (Synthetic Minority Over-sampling)")
    print("="*60)
    print("Note: Uncomment to use SMOTE. Requires: pip install imbalanced-learn")
    # detector_smote = FakeJobDetector()
    # detector_smote.train(X_train, y_train, model_type='random_forest', handle_imbalance='smote')
    # detector_smote.evaluate(X_test, y_test)
    # detector_smote.save_model('fake_job_model_smote.pkl')
    
    # Test with sample
    print("\n" + "="*60)
    print("Testing with Sample Job Postings")
    print("="*60)
    
    sample_fake = """
    URGENT! Work from home opportunity!
    Earn $5000-$10000 per month guaranteed! No experience needed!
    Payment required for training materials.
    Contact us at quickmoney@email.com
    """
    
    sample_legit = """
    Senior Software Engineer at ABC Tech Solutions
    We are seeking an experienced Full Stack Developer for our team in San Francisco.
    Requirements: 5+ years experience, Bachelor's degree in Computer Science
    Competitive salary $120,000-$160,000 with benefits
    Apply through our careers portal at www.abctech.com/careers
    """
    
    cleaned_fake = detector.clean_text(sample_fake)
    cleaned_legit = detector.clean_text(sample_legit)
    
    pred_fake, prob_fake = detector.predict([cleaned_fake])
    pred_legit, prob_legit = detector.predict([cleaned_legit])
    
    print(f"\nSample Fake Job:")
    print(f"Prediction: {'FAKE' if pred_fake[0] == 1 else 'LEGITIMATE'}")
    print(f"Confidence: {prob_fake[0][1]*100:.2f}% fake, {prob_fake[0][0]*100:.2f}% legitimate")
    
    print(f"\nSample Legitimate Job:")
    print(f"Prediction: {'FAKE' if pred_legit[0] == 1 else 'LEGITIMATE'}")
    print(f"Confidence: {prob_legit[0][1]*100:.2f}% fake, {prob_legit[0][0]*100:.2f}% legitimate")
    
    print("\n" + "="*60)
    print("Model training complete! Use 'fake_job_model_balanced.pkl' for best results.")
    print("="*60)


if __name__ == "__main__":
    main()