import numpy as np
from scipy.sparse import issparse, hstack, csr_matrix
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler
from .base_model import BaseModel


class NaiveBayesModel(BaseModel):
    """
    Naive Bayes — algorytm probabilistyczny oparty na założeniu niezależności cech.
    Używa dwóch wartiantów NB w zależności od typu danych:
      - MultinomialNB dla cech TF-IDF (wymaga wartości nieujemnych)
      - GaussianNB dla cech gęstych (BERT + VADER)

    W trybie 'combined' oba są łączone przez uśrednienie log-prawdopodobieństw.
    W trybie 'tfidf' używany jest tylko MultinomialNB.
    W trybie 'bert' używany jest tylko GaussianNB.
    """

    name = "NaiveBayes"

    def __init__(self, mode='combined'):
        super().__init__(mode=mode)
        self.mnb = MultinomialNB()
        self.gnb = GaussianNB()
        self.minmax = MinMaxScaler()
        # clf jest placeholderem — NaiveBayes używa własnej logiki predict
        self.clf = None
        self._classes = None

    def train(self, X_tfidf_train, X_vader_train, X_bert_train, y_train):
        print(f"  Training [{self.name} | {self.mode}]...")
        self._classes = np.unique(y_train)

        if self.mode in ('tfidf', 'combined'):
            # MinMaxScaler żeby zapewnić nieujemność dla MultinomialNB
            tfidf_dense = X_tfidf_train.toarray() if issparse(X_tfidf_train) else X_tfidf_train
            tfidf_scaled = self.minmax.fit_transform(tfidf_dense)
            self.mnb.fit(tfidf_scaled, y_train)

        if self.mode in ('bert', 'combined'):
            dense = X_bert_train.copy()
            if X_vader_train is not None and self.mode == 'combined':
                dense = np.hstack([dense, X_vader_train])
            dense_scaled = self.scaler.fit_transform(dense)
            self.gnb.fit(dense_scaled, y_train)

        print(f"  Mode: {self.mode} | Classes: {self._classes}")
        return self

    def predict(self, X_tfidf_test, X_vader_test, X_bert_test):
        if self.mode == 'tfidf':
            tfidf_dense = X_tfidf_test.toarray() if issparse(X_tfidf_test) else X_tfidf_test
            tfidf_scaled = self.minmax.transform(tfidf_dense)
            return self.mnb.predict(tfidf_scaled)

        if self.mode == 'bert':
            dense = X_bert_test.copy()
            dense_scaled = self.scaler.transform(dense)
            return self.gnb.predict(dense_scaled)

        # combined: uśrednij log-prawdopodobieństwa obu modeli
        tfidf_dense = X_tfidf_test.toarray() if issparse(X_tfidf_test) else X_tfidf_test
        tfidf_scaled = self.minmax.transform(tfidf_dense)
        log_proba_mnb = self.mnb.predict_log_proba(tfidf_scaled)

        dense = np.hstack([X_bert_test.copy(), X_vader_test]) if X_vader_test is not None else X_bert_test.copy()
        dense_scaled = self.scaler.transform(dense)
        log_proba_gnb = self.gnb.predict_log_proba(dense_scaled)

        combined_log = log_proba_mnb + log_proba_gnb
        indices = np.argmax(combined_log, axis=1)
        return self._classes[indices]

    @property
    def classes_(self):
        return self._classes
