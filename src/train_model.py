"""
train_model.py - Improved Sign Language Model Training
Trains a Random Forest classifier on hand landmark data
Supports two-hand detection (126 features)
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class SignLanguageModelTrainer:
    def __init__(self, data_path="data/final_landmarks.csv", model_path="models/rf_model.joblib"):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and validate dataset"""
        print("=" * 70)
        print("📂 Loading Dataset...")
        print("=" * 70)
        
        if not os.path.exists(self.data_path):
            print(f"❌ ERROR: Dataset not found at {self.data_path}")
            return False
        
        try:
            df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded successfully")
            print(f"   Path: {self.data_path}")
            print(f"   Total samples: {len(df)}")
            
            # Separate features and labels
            X = df.iloc[:, :-1].values.astype(np.float32)
            y = df.iloc[:, -1].values.astype(str)
            
            print(f"   Features per sample: {X.shape[1]}")
            print(f"   Unique classes: {len(np.unique(y))}")
            
            # Validate feature count
            if X.shape[1] != 126:
                print(f"⚠️  WARNING: Expected 126 features, got {X.shape[1]}")
                print(f"   Make sure you're using two-hand landmark data")
                user_input = input("   Continue anyway? (y/n): ")
                if user_input.lower() != 'y':
                    return False
            
            # Class distribution
            print(f"\n📊 Class Distribution:")
            unique, counts = np.unique(y, return_counts=True)
            for label, count in sorted(zip(unique, counts), key=lambda x: x[0]):
                print(f"   {label}: {count} samples")
            
            self.X = X
            self.y = y
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR loading data: {str(e)}")
            return False
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess and split data"""
        print(f"\n" + "=" * 70)
        print("🔧 Preprocessing Data...")
        print("=" * 70)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(self.y)
        
        print(f"✅ Labels encoded")
        print(f"   Classes: {list(self.label_encoder.classes_)}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Testing samples: {len(self.X_test)}")
        print(f"   Split ratio: {int((1-test_size)*100)}% train / {int(test_size*100)}% test")
        
        return True
    
    def train_model(self, n_estimators=300, max_depth=None, min_samples_split=2):
        """Train Random Forest model with cross-validation"""
        print(f"\n" + "=" * 70)
        print("🤖 Training Model...")
        print("=" * 70)
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        print(f"Model Configuration:")
        print(f"   Algorithm: Random Forest")
        print(f"   Number of trees: {n_estimators}")
        print(f"   Max depth: {max_depth if max_depth else 'None (unlimited)'}")
        print(f"   Min samples split: {min_samples_split}")
        
        # Cross-validation
        print(f"\n🔄 Performing 5-Fold Cross-Validation...")
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        print(f"   CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"   Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train on full training set
        print(f"\n🏋️  Training on full training set...")
        self.model.fit(self.X_train, self.y_train)
        print(f"✅ Training completed")
        
        return True
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print(f"\n" + "=" * 70)
        print("📊 Model Evaluation")
        print("=" * 70)
        
        # Training accuracy
        train_pred = self.model.predict(self.X_train)
        train_acc = accuracy_score(self.y_train, train_pred)
        
        # Testing accuracy
        test_pred = self.model.predict(self.X_test)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        print(f"\n🎯 Accuracy Scores:")
        print(f"   Training Accuracy:   {train_acc*100:.2f}%")
        print(f"   Testing Accuracy:    {test_acc*100:.2f}%")
        
        # Check for overfitting
        if train_acc - test_acc > 0.1:
            print(f"   ⚠️  Warning: Possible overfitting detected")
            print(f"   (Training accuracy significantly higher than test)")
        else:
            print(f"   ✅ Good generalization")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report:")
        print("=" * 70)
        report = classification_report(
            self.y_test, test_pred,
            target_names=self.label_encoder.classes_,
            digits=3
        )
        print(report)
        
        # Feature importance (top 10)
        feature_importance = self.model.feature_importances_
        top_10_idx = np.argsort(feature_importance)[-10:][::-1]
        
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, idx in enumerate(top_10_idx, 1):
            hand = "Left" if idx < 63 else "Right"
            landmark_idx = idx % 63
            coord = ['X', 'Y', 'Z'][landmark_idx % 3]
            landmark_num = landmark_idx // 3
            print(f"   {i:2d}. Feature {idx:3d} ({hand} Hand, Landmark {landmark_num:2d}, {coord}): {feature_importance[idx]:.4f}")
        
        return test_acc
    
    def save_model(self):
        """Save trained model"""
        print(f"\n" + "=" * 70)
        print("💾 Saving Model...")
        print("=" * 70)
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Prepare model data
        model_data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "feature_size": self.X.shape[1],
            "classes": list(self.label_encoder.classes_),
            "n_samples": len(self.X),
            "accuracy": accuracy_score(self.y_test, self.model.predict(self.X_test))
        }
        
        # Save
        joblib.dump(model_data, self.model_path)
        
        file_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
        
        print(f"✅ Model saved successfully")
        print(f"   Path: {self.model_path}")
        print(f"   File size: {file_size:.2f} MB")
        print(f"   Feature size: {model_data['feature_size']}")
        print(f"   Classes: {len(model_data['classes'])}")
        print(f"   Test accuracy: {model_data['accuracy']*100:.2f}%")
        
        return True
    
    def run(self):
        """Complete training pipeline"""
        print("\n" + "🚀" * 35)
        print("Sign Language Model Training Pipeline")
        print("🚀" * 35 + "\n")
        
        # Load data
        if not self.load_data():
            return False
        
        # Preprocess
        if not self.preprocess_data():
            return False
        
        # Train
        if not self.train_model():
            return False
        
        # Evaluate
        test_acc = self.evaluate_model()
        
        # Save
        if not self.save_model():
            return False
        
        # Summary
        print(f"\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"📊 Final Model Statistics:")
        print(f"   Total samples trained: {len(self.X)}")
        print(f"   Number of classes: {len(self.label_encoder.classes_)}")
        print(f"   Test Accuracy: {test_acc*100:.2f}%")
        print(f"   Model saved to: {self.model_path}")
        print("=" * 70 + "\n")
        
        return True

def main():
    """Main entry point"""
    trainer = SignLanguageModelTrainer(
        data_path="data/final_landmarks.csv",
        model_path="models/rf_model.joblib"
    )
    
    success = trainer.run()
    
    if success:
        print("🎉 You can now use this model for sign language detection!")
        print("   Run: python detect_live.py")
        print("   Or:  python web_app.py")
    else:
        print("❌ Training failed. Please check the errors above.")

if __name__ == "__main__":
    main()