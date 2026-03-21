"""
main.py — główny punkt wejścia do eksperymentów.

Porównuje różne modele ML (SVM, Logistic Regression, Random Forest, Naive Bayes)
na zadaniu klasyfikacji nastroju piosenek do 4 klas: angry, happy, sad, relaxed.

Reprezentacje cech:
  - 'tfidf'    : TF-IDF na przetworzonym tekście
  - 'bert'     : embeddingi BERT (all-MiniLM-L6-v2) + cechy VADER
  - 'combined' : TF-IDF + BERT + VADER łącznie

Uruchomienie:
  python main.py
  python main.py --data test/cleaned_training_data_watki.csv --results wyniki/
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.preprocessing import LyricsPreprocessor, VADERFeatureExtractor
from utils.features import BERTEmbeddingExtractor, TFIDFExtractor
from utils.evaluate import evaluate_model, print_best_report, save_visualizations
from models import SVMModel, LogisticRegressionModel, RandomForestModel, NaiveBayesModel


# ==============================================================================
# KONFIGURACJA MODELI DO URUCHOMIENIA
#
# Każdy wpis: (klasa_modelu, tryb_cech, parametry)
# Tryby: 'tfidf', 'bert', 'combined'
#
# Uwaga: Random Forest jest powolny w trybie 'tfidf' (rzadkie macierze),
# dlatego używamy go tylko z 'bert' i 'combined'.
# ==============================================================================
MODEL_CONFIGS = [
    (SVMModel,                'tfidf',    {'C': 1.0}),
    (SVMModel,                'bert',     {'C': 1.0}),
    (SVMModel,                'combined', {'C': 1.0}),
    (LogisticRegressionModel, 'tfidf',    {'C': 1.0}),
    (LogisticRegressionModel, 'bert',     {'C': 1.0}),
    (LogisticRegressionModel, 'combined', {'C': 1.0}),
    (RandomForestModel,       'bert',     {'n_estimators': 200}),
    (RandomForestModel,       'combined', {'n_estimators': 200}),
    (NaiveBayesModel,         'tfidf',    {}),
    (NaiveBayesModel,         'bert',     {}),
    (NaiveBayesModel,         'combined', {}),
]


def load_data(csv_path, text_column, label_column):
    print("=" * 70)
    print("KROK 1: Wczytywanie danych")
    print("=" * 70)
    df = pd.read_csv(csv_path)
    print(f"Kolumny: {df.columns.tolist()}")
    print(f"Piosenek: {len(df)}")
    print(f"\n📊 Rozkład etykiet:\n{df[label_column].value_counts()}")
    df = df.dropna(subset=[text_column])
    return df


def run_pipeline(df, X_bert, label_column, idx_train, idx_test, tfidf_extractor):
    """
    Trenuje i ewaluuje wszystkie skonfigurowane modele.
    Zwraca dict: {klucz_modelu: wynik}
    """
    y = df[label_column]
    vader_cols = [c for c in df.columns if c.startswith('vader_')]
    X_vader = df[vader_cols].values

    X_text_train  = df['PROCESSED_LYRICS'].iloc[idx_train]
    X_text_test   = df['PROCESSED_LYRICS'].iloc[idx_test]
    X_vader_train = X_vader[idx_train]
    X_vader_test  = X_vader[idx_test]
    X_bert_train  = X_bert[idx_train]
    X_bert_test   = X_bert[idx_test]
    y_train = y.iloc[idx_train]
    y_test  = y.iloc[idx_test]

    X_tfidf_train = tfidf_extractor.fit_transform(X_text_train)
    X_tfidf_test  = tfidf_extractor.transform(X_text_test)

    print(f"\nTrain: {len(y_train)} | Test: {len(y_test)}")
    print(f"{'Model':<35} {'Accuracy':>10}")
    print("-" * 47)

    results = {}
    for ModelClass, mode, params in MODEL_CONFIGS:
        model = ModelClass(mode=mode, **params)
        key = f"{model.name}_{mode}"
        try:
            model.train(X_tfidf_train, X_vader_train, X_bert_train, y_train)
            result = evaluate_model(model, X_tfidf_test, X_vader_test, X_bert_test, y_test)
            result['y_test'] = y_test
            results[key] = result
            print(f"  {key:<33} {result['acc']:>10.3f}")
        except Exception as e:
            print(f"  {key:<33} ❌ ERROR: {e}")

    print_best_report(results, y_test)
    return results


def print_summary(results):
    print("\n" + "=" * 70)
    print("PODSUMOWANIE — ranking modeli")
    print("=" * 70)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['acc'], reverse=True)
    for rank, (key, r) in enumerate(sorted_results, 1):
        print(f"  {rank:>2}. {key:<35} acc={r['acc']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Mood classification — porównanie modeli ML')
    parser.add_argument('--data',      default=r'd:\Git\licencjat\data\cleaned_training_data.csv')
    parser.add_argument('--results',   default=r'd:\Git\licencjat\results')
    parser.add_argument('--text-col',  default='LYRICS_EN')
    parser.add_argument('--label-col', default='MOOD_TAG')
    args = parser.parse_args()

    # ---------- WCZYTANIE ----------
    df = load_data(args.data, args.text_col, args.label_col)

    # ---------- PREPROCESSING ----------
    print("\n" + "=" * 70)
    print("KROK 2: Preprocessing tekstu")
    print("=" * 70)
    preprocessor = LyricsPreprocessor()
    df = preprocessor.preprocess_dataframe(df, text_column=args.text_col)

    # ---------- VADER (jako cechy, nie relabelowanie) ----------
    print("\n" + "=" * 70)
    print("KROK 3: Ekstrakcja cech VADER")
    print("=" * 70)
    vader_extractor = VADERFeatureExtractor()
    vader_df = vader_extractor.extract_dataframe(df, text_column=args.text_col)
    for col in vader_df.columns:
        df[col] = vader_df[col].values

    # ---------- BERT ----------
    print("\n" + "=" * 70)
    print("KROK 4: Embeddingi BERT")
    print("=" * 70)
    bert_extractor = BERTEmbeddingExtractor(model_name='all-MiniLM-L6-v2')
    X_bert = bert_extractor.extract_dataframe(df, text_column=args.text_col)

    # ---------- PODZIAŁ DANYCH ----------
    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=df[args.label_col]
    )

    # ---------- TRENOWANIE I EWALUACJA ----------
    print("\n" + "=" * 70)
    print("KROK 5: Trenowanie i ewaluacja modeli")
    print("=" * 70)
    tfidf = TFIDFExtractor(max_features=5000)
    results = run_pipeline(df, X_bert, args.label_col, idx_train, idx_test, tfidf)

    # ---------- PODSUMOWANIE ----------
    print_summary(results)

    # ---------- WIZUALIZACJE ----------
    print("\n" + "=" * 70)
    print("KROK 6: Zapis wizualizacji")
    print("=" * 70)
    save_visualizations(results, args.results)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
