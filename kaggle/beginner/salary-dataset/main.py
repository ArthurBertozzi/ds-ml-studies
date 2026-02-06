import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from utils.kaggle_utils import load_first_csv
from utils.print_utils import print_breaker
from IPython.display import display

import matplotlib.pyplot as plt
import seaborn as sns
from linear_regression import manual_linear_regression, predict

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


def summarize_plt(rows, columns, x_total, y_total, x_ratio, y_ratio, df):
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
    fig.delaxes(axes[2, 2])
    fig.delaxes(axes[1, 2])
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

    # Data visualization
    summarize_plt(
        rows=3, columns=3, x_total=14, y_total=12, x_ratio=1, y_ratio=1, df=df
    )
