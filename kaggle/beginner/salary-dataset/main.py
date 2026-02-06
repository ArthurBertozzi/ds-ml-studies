import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from utils.kaggle_utils import load_first_csv

DATA_SET_NAME = "ishank2005/salary-csv"


def main():
    try:
        df = load_first_csv(DATA_SET_NAME)

        print("Dataset carregado com sucesso!")
        print(df.head())

    except Exception as e:
        print(f"Erro ao processar dataset: {e}")


if __name__ == "__main__":
    main()
