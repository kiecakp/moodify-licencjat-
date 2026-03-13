from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest — model zespołowy oparty na drzewach decyzyjnych.
    Nie działa dobrze z bardzo rzadkimi macierzami TF-IDF (powolny),
    dlatego zalecany tryb to 'bert' lub 'combined'.
    """

    name = "RandomForest"

    def __init__(self, mode='combined', n_estimators=200):
        super().__init__(mode=mode)
        self.n_estimators = n_estimators
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
