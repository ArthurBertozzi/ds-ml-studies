import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from utils.kaggle_utils import load_first_csv
from utils.print_utils import print_breaker
from IPython.display import display

import matplotlib.pyplot as plt
import seaborn as sns
from linear_regression_manual import (
    manual_linear_regression,
    predict,
    residuals,
    mse,
    rmse,
    mae,
    r2_score,
)

from linear_regression_sklearn import SklearnLinearRegression


DATA_SET_NAME = "ishank2005/salary-csv"


def create_scatter_plot(axes, df):
    axes[1, 0].scatter(df["YearsExperience"], df["Salary"])
    axes[1, 0].set_xlabel("Years of Experience")
    axes[1, 0].set_ylabel("Salary")
    axes[1, 0].set_title("Salary vs Years of Experience")


def create_box_plot(axes, df):
    # Boxplot YearsExperience
    axes[0, 0].boxplot(df["YearsExperience"])
    axes[0, 0].set_title("Years of Experience")

    # Boxplot Salary
    axes[0, 1].boxplot(df["Salary"])
    axes[0, 1].set_title("Salary")


def create_heatmap(axes, df):
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=axes[1, 1])
    axes[1, 1].set_title("Correlation Heatmap")


def create_histogram(axes, df):
    df.hist(ax=axes[2, :2], bins=20)


def create_scatter_with_tendency(axes, df):
    sns.regplot(
        x="YearsExperience",
        y="Salary",
        data=df,
        ax=axes[0, 2],
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "red"},
    )
    axes[0, 2].set_title("Salary vs Years of Experience with Trend Line")


def create_residuals_scatter(axes, x, y, b0, b1):
    res = residuals(x, y, b0, b1)

    axes[1, 2].scatter(x, res)
    axes[1, 2].axhline(0, color="red", linestyle="--")
    axes[1, 2].set_title("Residuals vs YearsExperience")
    axes[1, 2].set_xlabel("YearsExperience")
    axes[1, 2].set_ylabel("Residual")


def create_residuals_histogram(axes, x, y, b0, b1):
    res = residuals(x, y, b0, b1)

    axes[2, 2].hist(res, bins=15)
    axes[2, 2].set_title("Residuals Distribution")
    axes[2, 2].set_xlabel("Residual")
    axes[2, 2].set_ylabel("Frequency")


def summarize_plt(rows, columns, x_total, y_total, x_ratio, y_ratio, df, b0, b1):
    ratios = [x_ratio] + [y_ratio] * (rows - 1)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(x_total, y_total),
        gridspec_kw={"height_ratios": ratios},
    )
    create_scatter_plot(axes, df)
    create_box_plot(axes, df)
    create_heatmap(axes, df)
    create_scatter_with_tendency(axes, df)
    create_histogram(axes, df)

    x = df["YearsExperience"]
    y = df["Salary"]

    create_residuals_scatter(axes, x, y, b0, b1)
    create_residuals_histogram(axes, x, y, b0, b1)

    plt.tight_layout()
    plt.show()


def df_analysis(df):
    try:

        # %%
        print_breaker("DF Head")
        display(df.head())

        # %%
        print_breaker("Informações do Dataset")
        display(df.describe())

        # %%
        print_breaker("Analises do DF")
        print(f"Existe algum dado faltando?\n {df.isnull().sum()}")
        print(f"Df data types:\n{df.dtypes}")

        # %%
        print_breaker("Analise de Correlação")
        print(df.corr())

        # %%
        """
            # Skew (assimetria) descreve para qual lado a distribuição dos dados “puxa”.
            # Valores positivos indicam que a cauda direita é mais longa ou mais gorda do que a cauda esquerda, enquanto valores negativos indicam o contrário.
            |skew| < 0.5 → ok
            0.5–1 → moderado
            1 → atenção
        """
        print_breaker("Checking skewness")
        print(df["YearsExperience"].skew())
        print(df["Salary"].skew())

    except Exception as e:
        print(f"Erro ao processar dataset: {e}")


if __name__ == "__main__":
    df = load_first_csv(DATA_SET_NAME)
    # DF general analysis
    df_analysis(df)

    # Manual linear regression
    x = df["YearsExperience"]
    y = df["Salary"]

    b0, b1 = manual_linear_regression(x, y)

    print_breaker("Manual Linear Regression Coefficients")
    print(f"Intercept - Intercepto (b0): {b0}")
    print(f"Slope  - Inclinação (b1): {b1}")
    print_breaker("Linear Equation")
    print(f"Salary = {b0:.2f} + {b1:.2f} * YearsExperience")

    print_breaker("Predictions")
    print(f"Predicted Salary for 5 years of experience: {predict(5, b0, b1):.2f}")

    # =========================
    # Model Error Metrics
    # =========================
    y_pred = predict(x, b0, b1)

    model_mse = mse(y, y_pred)
    model_rmse = rmse(y, y_pred)
    model_mae = mae(y, y_pred)
    model_r2 = r2_score(y, y_pred)

    print_breaker("Model Error Metrics")
    print(f"MSE  (Mean Squared Error): {model_mse:,.2f}")
    print(f"RMSE (Root Mean Squared Error): {model_rmse:,.2f}")
    print(f"MAE  (Mean Absolute Error): {model_mae:,.2f}")
    print(f"R²   (Coefficient of Determination): {model_r2:.4f}")

    # =========================
    # Sklearn Linear Regression
    # =========================
    sk_model = SklearnLinearRegression()
    sk_b0, sk_b1 = sk_model.fit(x, y)

    print_breaker("Sklearn Linear Regression Coefficients")
    print(f"Intercept (b0): {sk_b0}")
    print(f"Slope     (b1): {sk_b1}")

    print_breaker("Sklearn Prediction")
    print(f"Predicted Salary (5 years): {sk_model.predict([5])[0]:.2f}")

    # =========================
    # Compare Metrics
    # =========================
    y_pred_manual = predict(x, b0, b1)
    y_pred_sklearn = sk_model.predict(x)

    print_breaker("Model Comparison")

    print("Manual Model:")
    print(f"  RMSE: {rmse(y, y_pred_manual):,.2f}")
    print(f"  MAE : {mae(y, y_pred_manual):,.2f}")
    print(f"  R²  : {r2_score(y, y_pred_manual):.4f}")

    print("\nSklearn Model:")
    print(f"  RMSE: {rmse(y, y_pred_sklearn):,.2f}")
    print(f"  MAE : {mae(y, y_pred_sklearn):,.2f}")
    print(f"  R²  : {r2_score(y, y_pred_sklearn):.4f}")

    # Data visualization
    summarize_plt(
        rows=3,
        columns=3,
        x_total=14,
        y_total=12,
        x_ratio=1,
        y_ratio=1,
        df=df,
        b0=b0,
        b1=b1,
    )

    """
    Proximos passos
    Checar com modelo sklearn e comparar
    """
