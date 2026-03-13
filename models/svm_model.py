from sklearn.svm import LinearSVC
from .base_model import BaseModel


class SVMModel(BaseModel):
    """
    Klasyfikator SVM - Support Vector Machine (LinearSVC - liniowy).
    Model, który szuka linii podziału (hiperpłaszczyzny) która najlepiej 
    oddziela klasy w przestrzeni cech.
    
    Parametr C kontroluje regularyzację — im wyższe C, tym mniej regularyzacji.
    """

    name = "SVM"

    def __init__(self, mode='combined', C=1.0):
        super().__init__(mode=mode)
        self.C = C
        self.clf = LinearSVC(
            C=C,
            class_weight='balanced',
            max_iter=50000,
            random_state=42,
        )
