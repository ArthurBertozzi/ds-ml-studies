import kagglehub
import os
import pandas as pd


def get_dataset_local_path(dataset_name):
    """
    Baixa e retorna o caminho de um dataset do Kaggle.
    Esse script é reutilizável para qualquer dataset.

    Args:
        dataset_name (str): Identificador do dataset (ex: "ishank2005/salary-csv")

    Returns:
        str: Caminho local onde os arquivos foram baixados.
    """
    print(f"Iniciando download da versão mais recente de: {dataset_name}")

    # Download latest version
    path = kagglehub.dataset_download(dataset_name)

    print("Path to dataset files:", path)
    return path


def load_first_csv(dataset_name):
    """
    Baixa o dataset e carrega o primeiro arquivo CSV encontrado em um DataFrame.
    """
    dataset_path = get_dataset_local_path(dataset_name)

    # Encontrar o arquivo CSV dentro da pasta baixada
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {dataset_path}")

    # Usar o primeiro CSV encontrado
    csv_path = os.path.join(dataset_path, csv_files[0])
    print(f"Lendo arquivo: {csv_path}")

    return pd.read_csv(csv_path)
