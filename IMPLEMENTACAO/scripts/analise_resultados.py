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

# Arquivo com todas as funcoes e codigos referentes a analise dos resultados

# scripts/analise_resultados.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA

def analisar_resultados(resultados, resultados_df):
    """
    Analisa e exibe os resultados dos experimentos de classificação.

    Parâmetros:
    - resultados: Dicionário onde cada chave é o nome de um modelo e o valor é outro dicionário contendo:
        - 'Resultados': Dicionário com as métricas de avaliação.
        - 'ConfusionMatrix': Matriz de confusão.
        - 'ClassificationReport': Relatório de classificação.
    - resultados_df: DataFrame contendo as métricas de avaliação para facilitar a visualização.
    """   
    # Tabela das metricas
    print("Tabela das métricas por modelo:")
    display(resultados_df)
    
    # Graficos das metricas
    metricas = ['Acuracia', 'Precisao', 'Recall', 'F1-Score', 'ROC-AUC']
    plt.figure(figsize=(20, 10))
    for i, metrica in enumerate(metricas, 1):
        plt.subplot(2, 3, i)
        sns.barplot(x=resultados_df.index, y=resultados_df[metrica], palette="viridis", hue=resultados_df.index, legend=False)
        plt.title(metrica)
        plt.ylim(0, 1)
        plt.ylabel(metrica)
        plt.xlabel('Modelo')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Matrizes de Confusao e Relatorios de Classificacao
    for nome, info in resultados.items():
        print(f"\n-> {nome} <-")
        print("Métricas:")
        for metrica, valor in info['Resultados'].items():
            print(f"  {metrica}: {valor:.4f}")
        
        cm = info['ConfusionMatrix']
        print("\nMatriz de Confusão:")
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Matriz de Confusão - {nome}")
        plt.xlabel("Previsto")
        plt.ylabel("Verdadeiro")
        plt.show()
    
    # Consideracoes Finais
    print("\n### Considerações Finais ###")
    melhor_modelo_f1 = resultados_df['F1-Score'].idxmax()
    print(f"O melhor modelo em termos de F1-Score foi: {melhor_modelo_f1} ({resultados_df.loc[melhor_modelo_f1, 'F1-Score']:.4f})")
    melhor_modelo_auc = resultados_df['ROC-AUC'].idxmax()
    print(f"O melhor modelo em termos de ROC-AUC foi: {melhor_modelo_auc} ({resultados_df.loc[melhor_modelo_auc, 'ROC-AUC']:.4f})")

def plot_learning_curve(model, X, y, title='Curva de aprendizado', cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Plota a curva de aprendizado para um modelo específico, comparando a pontuação de treinamento e validação.

    Parâmetros:
    - model: Instância do modelo de machine learning que será avaliado.
    - X: Dados de entrada para treinamento.
    - y: Labels correspondentes aos dados.
    - title: Título do gráfico. Padrão é "Curva de aprendizado".
    - cv: Número de folds para validação cruzada. Padrão é 5.
    - train_sizes: Array de proporções do conjunto de treinamento a serem utilizados. Se for None, utiliza np.linspace(0.1, 1.0, 5).
    """
    plt.figure()
    plt.title(title)
    plt.xlabel("Exemplos de Treinamento")
    plt.ylabel("ROC-AUC")
    train_sizes, train_scores, validation_scores = learning_curve(model, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring='roc_auc')
    train_scores_mean = np.mean(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Pontuação no Treinamento")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Pontuação Cross-Validation")
    plt.legend(loc="best")
    plt.show()

def plot_pca(X, y, n_components=2):
    """
    Realiza a redução de dimensionalidade dos dados utilizando PCA e plota um scatter plot dos dados reduzidos.

    Parâmetros:
      - X: Dados de entrada que serão transformados.
      - y: Labels ou categorias que serão usadas para colorir os pontos no gráfico.
      - n_components: Número de componentes principais a serem extraídos (default=2).
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette="viridis", legend="full", alpha=0.7)
    plt.title("PCA - Redução para 2 Componentes")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.show()