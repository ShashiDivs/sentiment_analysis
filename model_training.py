# Classification Probelm in ML we have regression Logistic regression, SVM, NaiveBayes, RandomForest XG

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from matplotlib.pyplot import plt
import seaborn as sns

class SentimentModel:

    def __init__(self, model_type='logistic_regression'):

        self.model_type = model_type
        self.model = None
        self.history = {}

    def create_model(self):

        models = {
            'naive_bayes': MultinomialNB(alpha=1.0),
            'logistic_regression': LogisticRegression(
                max_iter = 1000,
                random_state=42,
                C=1.0
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=20
            ),
            'svm': LinearSVC(
                max_iter=1000,
                random_state=42,
                C=1.0
            )
        }
        self.model = models.get(self.model_type)
        if self.model is None:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):

        if self.model is None:
            self.create_model()

        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train,train_pred)