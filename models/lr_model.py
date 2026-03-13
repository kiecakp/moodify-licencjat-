from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    Regresja logistyczna z regularyzacją L2.
    (Oblicza prawdopodobieństwo przynależności do klasy, a następnie 
    dokonuje klasyfikacji wybierając najbardziej prawdopodobną klasę).

    Dobry punkt odniesienia — interpretowalny i szybki.
    Prosty model, pełni rolę "bazowego" do porównania z bardziej złożonymi 
    modelami (SVM, RF, NB).
    """

    name = "LogisticRegression"

    def __init__(self, mode='combined', C=1.0):
        super().__init__(mode=mode)
        self.C = C
        self.clf = LogisticRegression(
            C=C,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
        )
