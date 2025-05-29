# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Nome: Joao Vitor Averaldo Antunes
# RA: 813979
# ################################################################

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def carregar_dados(rhp_data_path, train_path, test_path):
    """
    Carrega e converte para DataFrames cada um dos arquivos .csv relevantes.

    Parâmetros:
    - rhp_data_path: Caminho para o conjunto de dados
    - train_path: Caminho para o conjunto de treino
    - test_path: Caminho para o conjunto de teste

    Retorna:
    - rhp_data: DataFrame com os dados do conjunto principal.
    - train_labels: DataFrame com os rótulos de treino.
    - test_ids: DataFrame com os IDs do conjunto de teste.
    """
    rhp_data = pd.read_csv(rhp_data_path)
    train_labels = pd.read_csv(train_path)
    test_ids = pd.read_csv(test_path)
    return rhp_data, train_labels, test_ids

def construir_df_treino(rhp_data, train_labels):
    """
    Constrói o DataFrame de treino unindo o conjunto principal de dados com os rótulos de treino.

    Parâmetros:
    - rhp_data: DataFrame com os dados do conjunto principal.
    - train_labels: DataFrame com os rótulos de treino, que deve conter a coluna 'Id' para a junção.

    Retorna:
    - df_treino: DataFrame resultante da junção dos dados do conjunto principal com os rótulos de treino.
    """
    df_treino = train_labels.merge(rhp_data, on='Id', how='left')
    return df_treino

def construir_df_teste(rhp_data, test_ids):
    """
    Constrói o DataFrame de teste unindo o conjunto principal de dados com os IDs do teste.

    Parâmetros:
    - rhp_data: DataFrame com os dados do conjunto principal.
    - test_ids: DataFrame com os IDs do conjunto de teste, que deve conter a coluna 'Id' para a junção.

    Retorna:
    - df_teste: DataFrame resultante da junção dos dados do conjunto principal com os IDs de teste.
    """
    df_teste = test_ids.merge(rhp_data, on='Id', how='left')
    return df_teste

def plotar_histogramas(df, colunas=None):
    """
    Plota histogramas para as colunas numéricas do DataFrame.

    Parâmetros:
    - df: DataFrame contendo os dados.
    - colunas: Lista de nomes de colunas a serem plotadas. Se for None, serão selecionadas todas as colunas numéricas.
    """
    if colunas is None:
        colunas = df.select_dtypes(include=['float64', 'int64']).columns

    df[colunas].hist(bins=30, figsize=(15, 10))
    plt.suptitle("Histogramas das colunas numericas", fontsize=16)
    plt.show()

def plotar_correlacao(df, colunas_numericas=None):
    """
    Plota a matriz de correlação das colunas numéricas do DataFrame utilizando um heatmap.

    Parâmetros:
    - df: DataFrame contendo os dados.
    - colunas_numericas: Lista de nomes de colunas numéricas. Se for None, serão selecionadas todas as colunas numéricas.
    """
    if colunas_numericas is None:
        colunas_numericas = df.select_dtypes(include=[np.number]).columns

    corr = df[colunas_numericas].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='RdBu', fmt='.2f', vmin=-1, vmax=1)
    plt.title("Matriz de Correlação")
    plt.show()

def plotar_boxplots(df, colunas=None):
    """
    Plota boxplots para as colunas numéricas do DataFrame.

    Parâmetros:
      - df: DataFrame contendo os dados.
      - colunas: Lista de nomes de colunas a serem plotadas. Se for None, serão selecionadas todas as colunas numéricas.
    """
    if colunas is None:
        colunas = df.select_dtypes(include=[np.number]).columns

    n_cols = 3
    n_rows = int(np.ceil(len(colunas) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(colunas):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f"Boxplot de {col}")

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def analisar_variaveis_categoricas(df, colunas=None):
    """
    Analisa e plota a distribuição das variáveis categóricas do DataFrame.

    Para cada coluna categórica, exibe a contagem de ocorrências e plota um gráfico de barras com a frequência de cada categoria.

    Parâmetros:
      - df: DataFrame contendo os dados.
      - colunas: Lista de nomes de colunas categóricas. Se for None, serão selecionadas todas as colunas do tipo 'object' ou 'category'.
    """
    if colunas is None:
        colunas = df.select_dtypes(include=['object', 'category']).columns

    for col in colunas:
        print(f"\nAnálise da variável categórica '{col}':")
        freq = df[col].value_counts(dropna=False)
        print(freq)

        plt.figure(figsize=(6, 4))
        sns.countplot(x=df[col], order=freq.index)
        plt.title(f"Plot de {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
