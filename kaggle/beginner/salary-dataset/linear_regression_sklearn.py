import numpy as np
from sklearn.linear_model import LinearRegression


class SklearnLinearRegression:
    """
    Wrapper simples em torno do sklearn LinearRegression
    para manter a mesma interface mental do modelo manual.
    """

    def __init__(self):
        # Cria o modelo do sklearn
        self.model = LinearRegression()
        self.b0 = None  # intercepto
        self.b1 = None  # inclinação

    def fit(self, x, y):
        """
        Treina o modelo de regressão linear.

        sklearn espera:
        X -> matriz 2D (n_samples, n_features)
        y -> vetor 1D
        """
        X = np.array(x).reshape(-1, 1)
        y = np.array(y)

        self.model.fit(X, y)

        # Guarda coeficientes no mesmo formato do manual
        self.b0 = self.model.intercept_
        self.b1 = self.model.coef_[0]

        return self.b0, self.b1

    def predict(self, x):
        """
        Gera previsões usando o modelo treinado.
        """
        X = np.array(x).reshape(-1, 1)
        return self.model.predict(X)
