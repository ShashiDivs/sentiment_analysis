

from data_ingestion import load_imdb_data
from preprocessing import TextPreprocessor
from model_training import SentimentModel
from feature_extraction import FeatureExractior
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import pickle


class SentimentAnalysisPipeline:


    def __init__(self,
                use_lemmatization=True,
                feature_method='tfidf',
                max_features=5000,
                model_type="logistic_regression"):
        
        self.preprocessor = TextPreprocessor(use_lemmatization=use_lemmatization)
        self.feature_extractor = FeatureExractior(
            method=feature_method,
            max_features=max_features
        )
        self.model = SentimentModel(model_type=model_type)


    def run_complete_pipeline(self,sample_size=5000,test_size=0.2):

        #step 1
        train_df, test_df = load_imdb_data(sample_size=sample_size)

        # combine data
        df = pd.concat([train_df,test_df],ignore_index=True)


        # preprocessing
        df = self.preprocessor.preprocess_text(df)

        X = self.feature_extractor.fit_transform(
            df['cleaned_text'].tolist()
        )

        y = df['label'].values

        #slpit the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        # Model training

        self.model.train(X_train, y_train)

        metrics = self.model.evaluate(X_test, y_test)

        return metrics


if __name__ == "__main__":

    pipeline = SentimentAnalysisPipeline(
            use_lemmatization=True,
            feature_method='tfidf',
            max_features=5000,
            model_type="logistic_regression")
    
    metrics = pipeline.run_complete_pipeline(sample_size=1000)

    




