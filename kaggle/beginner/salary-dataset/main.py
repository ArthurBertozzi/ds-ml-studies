import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from utils.kaggle_utils import load_first_csv
from utils.print_utils import print_breaker
from IPython.display import display

DATA_SET_NAME = "ishank2005/salary-csv"


def main():
    try:
        df = load_first_csv(DATA_SET_NAME)

        # %%
        print_breaker("DF Head")
        display(df.head())

        # %%
        print_breaker("Informações do Dataset")
        display(df.describe())

    except Exception as e:
        print(f"Erro ao processar dataset: {e}")


if __name__ == "__main__":
    main()
