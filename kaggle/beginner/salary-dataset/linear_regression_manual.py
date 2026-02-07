import numpy as np


def manual_linear_regression(x, y):
    """
    Calcula os coeficientes da regressão linear simples
    usando a solução analítica (fórmula fechada).

    Objetivo:
    Encontrar a melhor reta:
        y = b0 + b1 * x

    onde:
    - b1 é a inclinação (slope)
    - b0 é o intercepto
    """

    # Converte para numpy array para permitir operações vetoriais
    x = np.array(x)
    y = np.array(y)

    # Médias de X e Y
    x_mean = x.mean()
    y_mean = y.mean()

    # Numerador da fórmula do coeficiente angular (covariância)
    numerator = ((x - x_mean) * (y - y_mean)).sum()

    # Denominador da fórmula (variância de X)
    denominator = ((x - x_mean) ** 2).sum()

    # Inclinação da reta (quanto Y varia quando X aumenta 1 unidade)
    b1 = numerator / denominator

    # Intercepto da reta (valor de Y quando X = 0)
    b0 = y_mean - b1 * x_mean

    return b0, b1


def predict(x, b0, b1):
    """
    Aplica a equação da reta para gerar previsões.

    y_pred = b0 + b1 * x
    """
    return b0 + b1 * x


def residuals(x, y, b0, b1):
    """
    Calcula os resíduos do modelo.

    Resíduo = valor real - valor previsto

    Serve para avaliar:
    - qualidade do ajuste
    - se as assumptions da regressão são respeitadas
    """
    x = np.array(x)
    y = np.array(y)

    y_pred = predict(x, b0, b1)
    return y - y_pred


def mse(y_true, y_pred):
    """
    Mean Squared Error (Erro Quadrático Médio)

    Penaliza erros grandes mais fortemente.
    """
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error

    Mesmo erro do MSE, mas na unidade original do alvo (Salary).
    """
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """
    Mean Absolute Error

    Erro médio absoluto.
    Mais interpretável no dia a dia.
    """
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Coeficiente de determinação (R²)

    Mede o quanto da variância de Y é explicada pelo modelo.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
