# enhanced_fake_job_detector.py
# Advanced ML Pipeline for Fake Job Posting Detection with Model Comparison

import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV,
    StratifiedKFold,
    learning_curve
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


class EnhancedFakeJobDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.results = {}
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = ' '.join(text.split())
        
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                  if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def prepare_features(self, df):
        """Prepare features from the dataset"""
        df['combined_text'] = (
            df['title'].fillna('') + ' ' +
            df['location'].fillna('') + ' ' +
            df['department'].fillna('') + ' ' +
            df['company_profile'].fillna('') + ' ' +
            df['description'].fillna('') + ' ' +
            df['requirements'].fillna('') + ' ' +
            df['benefits'].fillna('')
        )
        
        df['cleaned_text'] = df['combined_text'].apply(self.clean_text)
        
        # Additional engineered features
        df['text_length'] = df['combined_text'].str.len()
        df['word_count'] = df['combined_text'].str.split().str.len()
        df['avg_word_length'] = df['combined_text'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
        df['exclamation_count'] = df['combined_text'].str.count('!')
        df['question_count'] = df['combined_text'].str.count('\?')
        df['caps_ratio'] = df['combined_text'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )
        
        return df
    
    def initialize_models(self):
        """Initialize multiple models for comparison"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                C=1.0
            ),
            'Naive Bayes': MultinomialNB(alpha=0.1),
            
        }
    
    def cross_validate_models(self, X_train, y_train):
        """Perform cross-validation on all models"""
        print("\n" + "="*70)
        print("CROSS-VALIDATION RESULTS (5-Fold Stratified)")
        print("="*70)
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_results = {}
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            
            # Calculate multiple metrics
            accuracy_scores = cross_val_score(model, X_train_tfidf, y_train, 
                                             cv=cv, scoring='accuracy', n_jobs=-1)
            precision_scores = cross_val_score(model, X_train_tfidf, y_train, 
                                              cv=cv, scoring='precision', n_jobs=-1)
            recall_scores = cross_val_score(model, X_train_tfidf, y_train, 
                                           cv=cv, scoring='recall', n_jobs=-1)
            f1_scores = cross_val_score(model, X_train_tfidf, y_train, 
                                       cv=cv, scoring='f1', n_jobs=-1)
            
            cv_results[name] = {
                'accuracy': accuracy_scores,
                'precision': precision_scores,
                'recall': recall_scores,
                'f1': f1_scores
            }
            
            print(f"  Accuracy:  {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std():.4f})")
            print(f"  Precision: {precision_scores.mean():.4f} (+/- {precision_scores.std():.4f})")
            print(f"  Recall:    {recall_scores.mean():.4f} (+/- {recall_scores.std():.4f})")
            print(f"  F1-Score:  {f1_scores.mean():.4f} (+/- {f1_scores.std():.4f})")
        
        return cv_results
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for Random Forest"""
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING - Random Forest (GridSearchCV)")
        print("="*70)
        
        X_train_tfidf = self.vectorizer.transform(X_train)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, 
            param_grid, 
            cv=3,  # 3-fold to save time
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nSearching through parameter combinations...")
        grid_search.fit(X_train_tfidf, y_train)
        
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best F1-Score: {grid_search.best_score_:.4f}")
        
        # Update Random Forest with best parameters
        self.models['Random Forest (Tuned)'] = grid_search.best_estimator_
        
        return grid_search.best_params_, grid_search.best_score_
    
    def train_all_models(self, X_train, y_train):
        """Train all models and compare performance"""
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        
        X_train_tfidf = self.vectorizer.transform(X_train)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_tfidf, y_train)
            print(f"  ✓ {name} training completed")
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n" + "="*70)
        print("MODEL COMPARISON - Test Set Performance")
        print("="*70)
        
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        comparison_results = []
        best_f1 = 0
        
        for name, model in self.models.items():
            predictions = model.predict(X_test_tfidf)
            
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            
            comparison_results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
            # Track best model based on F1-score
            if f1 > best_f1:
                best_f1 = f1
                self.best_model = model
                self.best_model_name = name
            
            print(f"\n{name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        print("\n" + "="*70)
        print("RANKING BY F1-SCORE:")
        print("="*70)
        print(comparison_df.to_string(index=False))
        
        print(f"\n🏆 Best Model: {self.best_model_name} (F1-Score: {best_f1:.4f})")
        
        return comparison_df
    
    def detailed_evaluation(self, X_test, y_test):
        """Detailed evaluation of the best model"""
        print("\n" + "="*70)
        print(f"DETAILED EVALUATION - {self.best_model_name}")
        print("="*70)
        
        X_test_tfidf = self.vectorizer.transform(X_test)
        predictions = self.best_model.predict(X_test_tfidf)
        probabilities = self.best_model.predict_proba(X_test_tfidf)
        
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, 
                                   target_names=['Legitimate', 'Fake']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        
        tn, fp, fn, tp = cm.ravel()
        print(f"\nTrue Negatives (Correct Legitimate):  {tn}")
        print(f"False Positives (Legit flagged Fake): {fp}")
        print(f"False Negatives (Fake missed):        {fn}")
        print(f"True Positives (Correct Fake):        {tp}")
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nSensitivity (Recall for Fake):        {sensitivity:.4f}")
        print(f"Specificity (Recall for Legitimate):  {specificity:.4f}")
        
        # ROC-AUC Score
        if hasattr(self.best_model, 'predict_proba'):
            roc_auc = roc_auc_score(y_test, probabilities[:, 1])
            print(f"ROC-AUC Score:                        {roc_auc:.4f}")
        
        return predictions, probabilities
    
    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models"""
        print("\nGenerating ROC curves...")
        
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_tfidf)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curves saved as 'roc_curves_comparison.png'")
        plt.close()
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance for tree-based models"""
        if 'Random Forest' in self.best_model_name or 'Gradient Boosting' in self.best_model_name:
            print(f"\nGenerating feature importance plot for {self.best_model_name}...")
            
            feature_names = self.vectorizer.get_feature_names_out()
            importances = self.best_model.feature_importances_
            
            # Get top N features
            indices = importances.argsort()[-top_n:][::-1]
            top_features = [(feature_names[i], importances[i]) for i in indices]
            
            print(f"\nTop {top_n} Most Important Features:")
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"{i:2d}. {feature:20s} - {importance:.6f}")
            
            # Plot
            plt.figure(figsize=(10, 8))
            features = [f[0] for f in top_features]
            importance_values = [f[1] for f in top_features]
            
            plt.barh(range(len(features)), importance_values)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Most Important Features - {self.best_model_name}')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("✓ Feature importance plot saved as 'feature_importance.png'")
            plt.close()
    
    def plot_learning_curve(self, X_train, y_train):
        """Plot learning curve for the best model"""
        print(f"\nGenerating learning curve for {self.best_model_name}...")
        
        X_train_tfidf = self.vectorizer.transform(X_train)
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.best_model, X_train_tfidf, y_train,
            cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='f1'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training Score', color='blue', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                         alpha=0.15, color='blue')
        plt.plot(train_sizes, val_mean, label='Cross-Validation Score', color='red', marker='s')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                         alpha=0.15, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('F1-Score')
        plt.title(f'Learning Curve - {self.best_model_name}')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
        print("✓ Learning curve saved as 'learning_curve.png'")
        plt.close()
    
    def save_best_model(self, filepath='best_fake_job_model.pkl'):
        """Save the best trained model and vectorizer"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'model_name': self.best_model_name,
                'vectorizer': self.vectorizer
            }, f)
        print(f"\n✓ Best model ({self.best_model_name}) saved to '{filepath}'")


def main():
    """Main training pipeline with comprehensive ML analysis"""
    print("="*70)
    print("ENHANCED FAKE JOB DETECTOR - ML ANALYSIS PIPELINE")
    print("="*70)
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv('fake_job_postings.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"\n=== Class Distribution ===")
    print(f"Fraudulent jobs: {df['fraudulent'].sum()} ({df['fraudulent'].mean()*100:.2f}%)")
    print(f"Legitimate jobs: {(~df['fraudulent'].astype(bool)).sum()} ({(~df['fraudulent'].astype(bool)).mean()*100:.2f}%)")
    print(f"Imbalance Ratio: {(~df['fraudulent'].astype(bool)).sum() / df['fraudulent'].sum():.2f}:1")
    
    # Initialize detector
    detector = EnhancedFakeJobDetector()
    
    # Prepare features
    print("\nPreparing features...")
    df = detector.prepare_features(df)
    
    # Split data
    X = df['cleaned_text']
    y = df['fraudulent']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Vectorize
    print("\nVectorizing text data...")
    detector.vectorizer.fit(X_train)
    
    # Initialize models
    detector.initialize_models()
    
    # Cross-validation
    cv_results = detector.cross_validate_models(X_train, y_train)
    
    # Hyperparameter tuning
    best_params, best_score = detector.hyperparameter_tuning(X_train, y_train)
    
    # Train all models
    detector.train_all_models(X_train, y_train)
    
    # Compare models
    comparison_df = detector.evaluate_all_models(X_test, y_test)
    
    # Detailed evaluation of best model
    predictions, probabilities = detector.detailed_evaluation(X_test, y_test)
    
    # Generate visualizations
    detector.plot_roc_curves(X_test, y_test)
    detector.plot_feature_importance()
    detector.plot_learning_curve(X_train, y_train)
    
    # Save best model
    detector.save_best_model('best_fake_job_model.pkl')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - best_fake_job_model.pkl")
    print("  - roc_curves_comparison.png")
    print("  - feature_importance.png")
    print("  - learning_curve.png")
    print("\nYou can now use the best model for predictions!")


if __name__ == "__main__":
    main()