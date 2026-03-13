from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    Regresja logistyczna z regularyzacją L2.
    Dobry punkt odniesienia — interpretowalny i szybki.
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
