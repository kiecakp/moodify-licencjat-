import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler

class BaseModel:
    """
    BaseModel: klasa bazowa/szablon definiująca wspólne zachowanie dla
    wszystkich modeli (SVM, LR, RF, NB).

    Każdy model można trenować w trzech trybach w zależności od tego, 
    które cechy chcemy wykorzystać:
      - 'tfidf'    : tylko cechy TF-IDF (tylko słowa z tekstu)
      - 'bert'     : embeddingi BERT + cechy VADER
      - 'combined' : TF-IDF + BERT + VADER łącznie (wszystko razem).

    """

    name = "BaseModel"

    def __init__(self, mode='combined'):
        self.mode = mode
        self.scaler = StandardScaler()
        self.tfidf_extractor = None   # ustawiany z zewnątrz przez train()

    # budowanie mocierzy cech
    def _build_features(self, X_tfidf, X_vader, X_bert, fit=False):
        parts = []
        if self.mode in ('tfidf', 'combined'):
            parts.append(X_tfidf)
        if self.mode in ('bert', 'combined'):
            dense = X_bert.copy()   # kopiowanie embedding BERT zeby nie modyfikowac oryginalu
            
            if X_vader is not None and self.mode == 'combined':
                dense = np.hstack([dense, X_vader])
                
            dense_scaled = self.scaler.fit_transform(dense) if fit else self.scaler.transform(dense)
            parts.append(csr_matrix(dense_scaled))
        return hstack(parts) if len(parts) > 1 else parts[0]

    def train(self, X_tfidf_train, X_vader_train, X_bert_train, y_train):
        print(f"  Training [{self.name} | {self.mode}]...")
        features = self._build_features(X_tfidf_train, X_vader_train, X_bert_train, fit=True)
        print(f"  Feature matrix: {features.shape}")
        self.clf.fit(features, y_train)
        return self

    def predict(self, X_tfidf_test, X_vader_test, X_bert_test):
        features = self._build_features(X_tfidf_test, X_vader_test, X_bert_test, fit=False)
        return self.clf.predict(features)

    @property
    def classes_(self):
        return self.clf.classes_
