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

# Arquivo com todas as funcoes e codigos referentes aos experimentos

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import os

def dividir_dados(df, target_col, test_size=0.2, random_state=42):
    """
    Divide o DataFrame em conjuntos de treino e teste de forma estratificada.

    Parâmetros:
    - df: DataFrame contendo os dados completos.
    - target_col: O nome da coluna alvo para a predição.
    - test_size: Proporção dos dados que serão reservados para o teste (default=0.2).
    - random_state: Semente para reprodução dos resultados (default=42).

    Retorna:
    - tuple: Quatro objetos (X_train, X_test, y_train, y_test) resultantes da divisão dos dados, onde X são as features e y o target.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def definir_modelos():
    """
    Define e retorna um dicionário com instâncias de diversos modelos de classificação.

    Retorna:
    - Dicionário onde as chaves são os nomes dos modelos e os valores são as instâncias dos classificadores:
        - 'KNN'               : Instância de KNeighborsClassifier.
        - 'NaiveBayes'        : Instância de GaussianNB.
        - 'LogisticRegression': Instância de LogisticRegression com max_iter=1000 e random_state=42.
        - 'MLP'               : Instância de MLPClassifier com max_iter=1000 e random_state=42.
        - 'SVM'               : Instância de SVC com probability=True e random_state=42.
        - 'RandomForest'      : Instância de RandomForestClassifier com random_state=42.
    """
    modelos = {
        'KNN': KNeighborsClassifier(),
        'NaiveBayes': GaussianNB(),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'MLP': MLPClassifier(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42)
    }
    return modelos

def ajustar_modelo(modelo, X_train, y_train, parametros=None, cv=5):
    """
    Ajusta um modelo utilizando GridSearchCV para otimização dos hiperparâmetros.

    Parâmetros:
    - modelo: Instância do modelo a ser treinado.
    - X_train: Dados de treinamento.
    - y_train: Labels do treinamento.
    - parametros: Dicionário contendo os hiperparâmetros a serem testados com GridSearchCV. Se None, o modelo é treinado diretamente sem busca de hiperparâmetros.
    - cv: Número de folds para validação cruzada no GridSearchCV (default=5).

    Retorna:
    - Uma tupla contendo:
        - O melhor modelo encontrado no GridSearchCV.
        - Os melhores parâmetros encontrados (ou None se parametros for None).
        - A melhor pontuação de validação (ou None se parametros for None).
    """
    if parametros:
        grid = GridSearchCV(estimator=modelo, param_grid=parametros, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_, grid.best_score_
    else:
        modelo.fit(X_train, y_train)
        return modelo, None, None

def avaliar_modelo(modelo, X_test, y_test):
    """
    Avalia o desempenho de um modelo treinado utilizando um conjunto de teste.

    Parâmetros:
    - modelo: Modelo treinado que será avaliado.
    - X_test: Dados de teste (features).
    - y_test: Labels reais do conjunto de teste.

    Retorna:
    - Uma tupla contendo:
        - Dicionário com as principais métricas de avaliação (Acuracia, Precisao, Recall, F1-Score e ROC-AUC).
        - Matriz comparando as previsões com os valores reais.
        - String contendo as métricas detalhadas por classe.
        - Vetor de probabilidades preditas para a classe positiva (se disponível; caso contrário, None).
    """
    y_pred = modelo.predict(X_test)

    # Verifica se o modelo pode prever probabilidades
    if hasattr(modelo, "predict_proba"):
        y_proba = modelo.predict_proba(X_test)[:,1]
    elif hasattr(modelo, "decision_function"):
        decision_scores = modelo.decision_function(X_test)
        scaler = MinMaxScaler(feature_range=(0, 1))
        y_proba = scaler.fit_transform(decision_scores.reshape(-1, 1)).flatten()
    else:
        y_proba = None

    resultados = {
        'Acuracia': accuracy_score(y_test, y_pred),
        'Precisao': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }
    return resultados, confusion_matrix(y_test, y_pred), classification_report(y_test, y_pred, zero_division=0), y_proba

def avaliar_modelo_cv(modelo, X, y, cv=5):
    """
    Avalia um modelo utilizando validação cruzada e calcula a média e o desvio padrão da métrica ROC_AUC.

    Parâmetros:
    - modelo: Modelo que será avaliado.
    - X: Dados de entrada.
    - y: Labels correspondentes.
    - cv: Número de folds para a validação cruzada (default=5).

    Retorna:
    - Uma tupla contendo:
        - Média da pontuação ROC_AUC obtida na validação cruzada.
        - Desvio padrão das pontuações ROC_AUC.
    """
    cv_scores = cross_val_score(modelo, X, y, cv=cv, scoring='roc_auc')
    return cv_scores.mean(), cv_scores.std()

def executar_experimentos(df, target_col):
    """
    Executa uma série de experimentos para treinar e avaliar diferentes modelos de classificação.

    Parâmetros:
    - df: DataFrame com os dados de entrada pré-processados que contém a coluna alvo.
    - target_col: Nome da coluna alvo para predição.

    Retorna:
    - Um dicionário com os resultados de cada modelo, onde cada chave é o nome do modelo e o valor é outro dicionário contendo:
        - 'Modelo': O modelo treinado.
        - 'Resultados': Métricas de avaliação (Acuracia, Precisao, Recall, F1-Score, ROC-AUC).
        - 'ConfusionMatrix': A matriz de confusão.
        - 'ClassificationReport': O relatório de classificação.
        - 'Probabilidades': Vetor de probabilidades preditas.
        - 'CV_ROC_AUC_Mean': Média da pontuação ROC_AUC na validação cruzada.
        - 'CV_ROC_AUC_Std': Desvio padrão da pontuação ROC_AUC na validação cruzada.
    """
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = dividir_dados(df, target_col)
    
    # Definicao dos modelos
    modelos = definir_modelos()
    
    # Dicionario para armazenar os resultados
    resultados_modelos = {}
    
    for nome, modelo in modelos.items():
        print(f"\nTreinando o modelo: {nome}")
        
        # Definicao dos parametros para GridSearch
        parametros = None
        if nome == 'KNN':
            parametros = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
        elif nome == 'LogisticRegression':
            parametros = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
        elif nome == 'MLP':
            parametros = {'hidden_layer_sizes': [(100,), (50,50), (100,50)], 'activation': ['relu', 'tanh']}
        elif nome == 'SVM':
            parametros = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        elif nome == 'RandomForest':
            parametros = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        
        # Ajusta o modelo
        if parametros:
            modelo_ajustado, best_params, best_score = ajustar_modelo(modelo, X_train, y_train, parametros)
            print(f"Melhores parâmetros: {best_params}")
            print(f"Melhor ROC_AUC-score: {best_score}")
        else:
            modelo_ajustado, best_params, best_score = ajustar_modelo(modelo, X_train, y_train)
        
        # Avaliar o modelo no conjunto de teste
        resultados, cm, cr, y_proba = avaliar_modelo(modelo_ajustado, X_test, y_test)
        
        # Cross-validation no conjunto de treino
        cv_mean, cv_std = avaliar_modelo_cv(modelo_ajustado, X_train, y_train)
        print(f"Cross-validation ROC_AUC: {cv_mean:.4f} ± {cv_std:.4f}")
        
        resultados_modelos[nome] = {
            'Modelo': modelo_ajustado,
            'Resultados': resultados,
            'ConfusionMatrix': cm,
            'ClassificationReport': cr,
            'Probabilidades': y_proba,
            'CV_ROC_AUC_Mean': cv_mean,
            'CV_ROC_AUC_Std': cv_std
        }
        
        print(f"Resultados do modelo {nome}:")
        for metric, value in resultados.items():
            print(f"  {metric}: {value}")
        print(f"Relatório de Classificação:\n{cr}")
    
    return resultados_modelos