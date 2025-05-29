# ############################################################
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
# ############################################################

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento

# imports
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

def tratar_outliers_personalizado(df):
    """
    Trata outliers de forma personalizada para colunas específicas.
    
    Regras:
    - Peso: valores iguais a 0 → substituição pela mediana
    - Altura: valores < 25 → substituição pela mediana
    - PA_Sistolica: valores < 50 ou > 200 → substituição pela mediana
    - PA_Diastolica: valores < 40 ou > 100 → substituição pela mediana
    - FC: valores < 40 ou > 180 → substituição pela mediana
    - IDADE: valores < 0 ou > 18 → substituição pela média
    
    Parâmetros:
    - df: DataFrame a ser processado.
    
    Retorna:
    - DataFrame com os outliers tratados.
    """
    
    # Peso: valores iguais a 0 → mediana
    if 'Peso' in df.columns:
        peso_zero_mask = df['Peso'] == 0
        if peso_zero_mask.any():
            mediana_peso = df.loc[~peso_zero_mask, 'Peso'].median()
            df.loc[peso_zero_mask, 'Peso'] = mediana_peso
            print(f"Substituído {peso_zero_mask.sum()} valores de 'Peso' iguais a 0 pela mediana {mediana_peso}.")
    
    # Altura: valores < 25 → mediana
    if 'Altura' in df.columns:
        altura_baixa_mask = df['Altura'] < 25
        if altura_baixa_mask.any():
            mediana_altura = df.loc[~altura_baixa_mask, 'Altura'].median()
            df.loc[altura_baixa_mask, 'Altura'] = mediana_altura
            print(f"Substituído {altura_baixa_mask.sum()} valores de 'Altura' < 25 pela mediana {mediana_altura}.")
    
    # PA_Sistolica: valores < 40 ou > 200 → mediana
    if 'PA SISTOLICA' in df.columns:
        pa_sis_outlier_mask = (df['PA SISTOLICA'] < 40) | (df['PA SISTOLICA'] > 200)
        if pa_sis_outlier_mask.any():
            mediana_pa_sis = df.loc[~pa_sis_outlier_mask, 'PA SISTOLICA'].median()
            df.loc[pa_sis_outlier_mask, 'PA SISTOLICA'] = mediana_pa_sis
            print(f"Substituído {pa_sis_outlier_mask.sum()} outliers de 'PA SISTOLICA' pela mediana {mediana_pa_sis}.")
    
    # PA_Diastolica: valores < 40 ou > 120 → mediana
    if 'PA DIASTOLICA' in df.columns:
        pa_dia_outlier_mask = (df['PA DIASTOLICA'] < 40) | (df['PA DIASTOLICA'] > 120)
        if pa_dia_outlier_mask.any():
            mediana_pa_dia = df.loc[~pa_dia_outlier_mask, 'PA DIASTOLICA'].median()
            df.loc[pa_dia_outlier_mask, 'PA DIASTOLICA'] = mediana_pa_dia
            print(f"Substituído {pa_dia_outlier_mask.sum()} outliers de 'PA DIASTOLICA' pela mediana {mediana_pa_dia}.")
    
    # FC: valores < 40 ou > 180 → mediana
    if 'FC' in df.columns:
        fc_outlier_mask = (df['FC'] < 40) | (df['FC'] > 180)
        if fc_outlier_mask.any():
            mediana_fc = df.loc[~fc_outlier_mask, 'FC'].median()
            df.loc[fc_outlier_mask, 'FC'] = mediana_fc
            print(f"Substituído {fc_outlier_mask.sum()} outliers de 'FC' pela mediana {mediana_fc}.")
    
    # IDADE: valores < 0 ou > 120 → média
    if 'IDADE' in df.columns:
        idade_outlier_mask = (df['IDADE'] < 0) | (df['IDADE'] > 120)
        if idade_outlier_mask.any():
            media_idade = df.loc[~idade_outlier_mask, 'IDADE'].mean()
            df.loc[idade_outlier_mask, 'IDADE'] = media_idade
            print(f"Substituído {idade_outlier_mask.sum()} outliers de 'IDADE' pela média {media_idade}.")
    
    return df

def tratar_datas(df):
    """
    Converte colunas com datas para o tipo datetime.

    Também cria nova coluna 'IDADE' baseada na diferença entre datas.

    Parâmetros:
    - df: DataFrame a ser tratado.
    
    Retorna:
    - DataFrame com a data tratada.
    """
    # Converte colunas para datetime
    for col in ["Atendimento", "DN"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculo da idade se tiver ambas as datas
    if "Atendimento" in df.columns and "DN" in df.columns:
        df["IDADE"] = (df["Atendimento"] - df["DN"]).dt.days / 365.25
        # Substitui idades negativas por nan
        df.loc[df["IDADE"] < 0, "IDADE"] = np.nan
    
    return df

def tratar_valores_zerados(df, colunas_numericas=None):
    """
    Substitui valores inválidos (0 e negativos) por NaN em colunas numéricas específicas.

    (Casos em que 0 ou valores negativos não fazem sentido).

    Parâmetros:
    - df: DataFrame a ser tratado.
    - colunas_numericas: Colunas específicas a serem tratadas. O padrão é ocorrer o tratamento de Peso e Altura.
    
    Retorna:
    - DataFrame com valores zerados tratados.
    """
    if colunas_numericas is None:
        colunas_numericas = ["Peso", "Altura"]
    
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].mask(df[col] < 0, np.nan)
    
    return df


def calcular_imc(df, col_peso="Peso", col_altura="Altura"):
    """
    Calcula o IMC com base em Peso e Altura.

    IMC = Peso (kg) / (Altura (m))^2.

    Parâmetros:
    - df: DataFrame a ser tratado.
    - col_altura: Coluna relativa a altura. O valor padrão é "Altura".
    
    Retorna:
    - DataFrame com IMC calculado.

    """
    # Verifica se altura esta em cm e converter para metros se preciso
    if (df[col_altura].dropna() > 10).any():
        # Considera que valores > 10 estao em cm
        altura_em_m = df[col_altura] / 100.0
    else:
        altura_em_m = df[col_altura]

    # Atualiza coluna IMC
    df["IMC"] = df[col_peso] / (altura_em_m ** 2)

    return df


def unificar_categorias(df, col):
    """
    Unifica os valores de uma coluna do DataFrame de acordo com a coluna fornecida.
    
    Parâmetros:
    - df: O DataFrame contendo a coluna a ser unificada.
    - col: O nome da coluna que terá os valores unificados.
      
    Retorna:
    - DataFrame com a coluna unificada.
    """
    if col not in df.columns:
        return df
    
    if col=="SEXO":
        map_dict = {
        "M": "Masculino",
        "Masculino": "Masculino",
        "masculino": "Masculino",
        "F": "Feminino",
        "Feminino": "Feminino",
        "feminino": "Feminino",
        "Indeterminado": "Indeterminado"
        }

    elif col=="SOPRO":
        map_dict = {
            "ausente": "Ausente",
            "Ausente": "Ausente",
            "sistólico": "Sistólico",
            "contínuo": "Contínuo",
            "Contínuo": "Contínuo",
            "diastólico": "Diastólico",
            "Diastólico": "Diastólico"
        }

    elif col=="PULSOS":
        map_dict = {
            "NORMAIS": "Normais",
            "Normais": "Normais",
            "AMPLOS": "Amplos",
            "Amplos": "Amplos",
            "diminuidos": "Diminuídos",
            "diminuídos": "Diminuídos",
            "Diminidos": "Diminuídos",
            "Diminidos ": "Diminuídos",
            " Diminidos ": "Diminuídos",
            "Femorais diminuidos": "Femorais diminuidos",
            "Diminuídos": "Diminuídos",
        }

    elif col=="CLASSE":
        map_dict = {
            "Normais": "Normal",
            "Normal": "Normal",
            "Anormal": "Anormal",
        }
    else:
        map_dict = {}

    df[col] = df[col].map(map_dict).fillna(df[col])
    return df


def mapear_classe_binario(df, col="CLASSE"):
    """
    Mapeia os valores da CLASSE para 0 ou 1;

    Parâmetros:
    - df: DataFrame a ser convertido.
    - col: Coluna a ser convertida. O padrão é a "CLASSE".
    
    Retorna:
    - DataFrame com a coluna convertida.
    """
    if col not in df.columns:
        return df

    map_classe = {
        "Normal": 0,
        "Anormal": 1
    }

    df[col] = df[col].map(map_classe).fillna(df[col])
    return df


def tratar_valores_categoricos_especiais(df):
    """
    Algumas entradas possuem valores especiais como:

    '#VALUE!' e 'Não Calculado'.

    Coverte-os para NaN.

    Parâmetros:
    - df: DataFrame a ser tratado.
    
    Retorna:
    - DataFrame com valores especiais tratados.
    """
    col = "IDADE"
    if col in df.columns:
        df[col] = df[col].replace({
            "#VALUE!": np.nan
        })  
    return df


def imputar_valores_faltantes(df, strategy="mean"):
    """
    Imputa valores faltantes (NaN).
    strategy pode ser: 'mean', 'median' ou 'most_frequent'.

    Para as colunas categóricas é imputado a moda.
    
    Parâmetros:
    - df: DataFrame a ser imputado.
    - strategy: Estratégia de imputação de valores. O valor padrão é a média (mean) para valores numéricos.
    
    Retorna:
    - DataFrame com valores NaN substituídos.
    """
    
    # Separacao das colunas numericas e categoricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    # Imputar colunas numericas -> media
    if len(numeric_cols) > 0:
        imputer_num = SimpleImputer(strategy=strategy)
        df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
    
    # Imputar colunas categoricas -> moda
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    
    return df

def tratar_frequencia_cardiaca(df, col="FC"):
    """
    Processa a coluna de Frequência Cardíaca, convertendo intervalos em médias numéricas.
    
    Parâmetros:
    - df: DataFrame a ser processado.
    - col: Nome da coluna de Frequência Cardíaca. O padrão é "FC".
    
    Retorna:
    - DataFrame com a coluna FC processada.
    """
    if col not in df.columns:
        return df

    # Funcao auxiliar para calcular a media de um intervalo
    def calcular_media_intervalo(valor):
        try:
            # Verifica se o valor eh uma string e contem '-'
            if isinstance(valor, str) and '-' in valor:
                partes = valor.split('-')
                if len(partes) == 2:
                    num1 = float(partes[0].strip())
                    num2 = float(partes[1].strip())
                    return (num1 + num2) / 2.0
                else:
                    return np.nan
            elif isinstance(valor, (int, float)):
                return valor
            else:
                # Tenta converter diretamente para float
                return float(valor)
        except (ValueError, TypeError):
            return np.nan

    # Aplica a funcao auxiliar a toda a coluna
    df[col] = df[col].apply(calcular_media_intervalo)
    
    return df

def codificar_variaveis_categoricas(df, 
                                    colunas_categoricas=None, 
                                    strategy="one-hot",
                                    drop_first=False):
    """
    Codifica as variáveis categóricas de acordo com a estratégia definida. Se for None, o padrão é "one-hot".
    
    Parâmetros:
    - df: DataFrame contendo as colunas a serem codificadas.
    - colunas_categoricas: Lista de colunas categóricas a codificar. Se for None converte as colunas com valores do tipo "object" e "category".
    - strategy: "label" ou "one-hot".
      - "label": usa LabelEncoder em cada coluna.
      - "one-hot": usa pd.get_dummies.
    - drop_first: se True, no one-hot encoding, descarta a primeira coluna 

    Retorna:
    - df_cod: DataFrame com colunas categóricas codificadas.
    """
    
    df_cod = df.copy()
    
    # Deteccao das colunas categoricas
    if colunas_categoricas is None:
        colunas_categoricas = df_cod.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if "CLASSE" in colunas_categoricas:
        colunas_categoricas.remove("CLASSE")

    if strategy == "label":
        for col in colunas_categoricas:
            le = LabelEncoder()
            df_cod[col] = le.fit_transform(df_cod[col])
    
    elif strategy == "one-hot":
        df_cod = pd.get_dummies(df_cod, 
                                columns=colunas_categoricas,
                                drop_first=drop_first)
            
    return df_cod

def normalizar_variaveis_numericas(df, 
                                   colunas_numericas=None, 
                                   strategy="standard"):
    """
    Normaliza as colunas numéricas.
    
    Parâmetros:
    - df: DataFrame contendo as colunas a serem normalizadas.
    - colunas_numericas: Lista de colunas numéricas. Se for None, detecta automaticamente colunas numéricas.
    - strategy: "standard", "minmax" ou "robust".
      - "standard": StandardScaler (média=0, desvio padrão=1)
      - "minmax": MinMaxScaler (0 a 1)
      - "robust": RobustScaler (menos sensível a outliers)
    
    Retorna:
    - df_norm: DataFrame com colunas normalizadas.
    """
    
    df_norm = df.copy()
    
    # Deteccao das colunas numericas
    if colunas_numericas is None:
        colunas_numericas = df_norm.select_dtypes(include=[np.number]).columns.tolist()
    
    # Diferentes estrategias usadas nos testes
    if strategy == "standard":
        scaler = StandardScaler()
    elif strategy == "minmax":
        scaler = MinMaxScaler()
    elif strategy == "robust":
        scaler = RobustScaler()
    
    # fit_transform nas colunas numericas
    df_norm[colunas_numericas] = scaler.fit_transform(df_norm[colunas_numericas])
    
    return df_norm

def preprocessar_df(df):
    """
    Função principal que realiza todos os passos de pré-processamento.

    Parâmetros:
    - df: DataFrame a ser pré-processado.
    
    Retorna:
    - DataFrame com as colunas pré-processadas.
    """
    # Trata as datas e recalcula a idade
    df = tratar_datas(df)

    # Substitui zeros por nan nas colunas peso e altura
    df = tratar_valores_zerados(df, colunas_numericas=["Peso", "Altura"])

    # Trata frequencia cardiaca (FC)
    df = tratar_frequencia_cardiaca(df, col="FC")

    # Tratar e unificar categorias
    df = unificar_categorias(df, col="SEXO")
    df = unificar_categorias(df, col="SOPRO")
    df = unificar_categorias(df, col="PULSOS")
    df = unificar_categorias(df, col="CLASSE")
    df = tratar_valores_categoricos_especiais(df)

    # Substituicao personalizada dos outliers
    df = tratar_outliers_personalizado(df)

    # Imputar valores faltantes
    df = imputar_valores_faltantes(df, strategy="mean")

    # Recalcular IMC
    df = calcular_imc(df, col_peso="Peso", col_altura="Altura")

    # Remover colunas irrelevantes
    colunas_para_remover = ["Convenio", "Atendimento", "DN", "PPA"]
    df.drop(columns=[c for c in colunas_para_remover if c in df.columns],
            inplace=True, errors='ignore')
        
    df = codificar_variaveis_categoricas(df, colunas_categoricas=None, strategy="one-hot", drop_first=False)
    df = normalizar_variaveis_numericas(df, colunas_numericas=None, strategy="standard")
    df = mapear_classe_binario(df, col="CLASSE")
    
    return df
