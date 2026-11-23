"""
Script de Preparação e Persistência da Metodologia de Validação Cruzada
Projeto: Trabalho Acadêmico - Inteligência Artificial | Universidade Feevale
Fase 2: Pré-processamento e Preparação dos Experimentos
Autores: Gabriel Machado e João Tonin
Professor: Rodrigo
Data: Novembro/2026
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

# Configuração de caminhos relativos ao diretório raiz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')
LOGS_DIR = os.path.join(BASE_DIR, 'results', 'logs')

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Configuração de logging
log_file = os.path.join(LOGS_DIR, 'execucao_preprocessamento.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Definição canônica das colunas por tipo semântico
NOMINAL_COLS = ['SEX', 'EDUCATION', 'MARRIAGE']
ORDINAL_COLS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
NUMERIC_COLS = ['LIMIT_BAL', 'AGE'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]
TARGET_COL = 'default payment next month'
AUX_COLS = ['ID', 'feature_group_id']


def gerar_divisao_folds():
    """
    Executa a partição estratificada e agrupada (StratifiedGroupKFold) em 5 folds com seed 42.
    Garante simultaneamente:
    1. Estratificação da proporção de classes (77,88% classe 0 vs 22,12% classe 1).
    2. Agrupamento rigoroso: nenhum registro com preditores idênticos (mesmo feature_group_id)
       fica dividido entre partições de treino e validação, prevenindo vazamento de dados.
    """
    caminho_dados = os.path.join(DATA_PROCESSED_DIR, 'dataset_variacao_a.csv')
    if not os.path.exists(caminho_dados):
        raise FileNotFoundError(f"Dataset processado não encontrado: {caminho_dados}. Execute preprocess.py primeiro.")
    
    logger.info(f"Carregando dados para partição de folds: {caminho_dados}")
    df = pd.read_csv(caminho_dados)
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    folds_assignment = np.zeros(len(df), dtype=int)
    estatisticas_folds = []
    
    y = df[TARGET_COL]
    groups = df['feature_group_id']
    
    logger.info("Executando StratifiedGroupKFold (n_splits=5, shuffle=True, random_state=42)...")
    for fold_idx, (train_indices, val_indices) in enumerate(sgkf.split(df, y, groups), start=1):
        folds_assignment[val_indices] = fold_idx
        
        y_train = y.iloc[train_indices]
        y_val = y.iloc[val_indices]
        
        groups_train = set(groups.iloc[train_indices])
        groups_val = set(groups.iloc[val_indices])
        overlap_grupos = len(groups_train.intersection(groups_val))
        
        # Estatísticas do fold
        n_val = len(val_indices)
        n_val_0 = int((y_val == 0).sum())
        n_val_1 = int((y_val == 1).sum())
        
        n_train = len(train_indices)
        n_train_0 = int((y_train == 0).sum())
        n_train_1 = int((y_train == 1).sum())
        
        logger.info(f"Fold {fold_idx}: Treino={n_train} (C0={n_train_0/n_train*100:.2f}%, C1={n_train_1/n_train*100:.2f}%), "
                    f"Val={n_val} (C0={n_val_0/n_val*100:.2f}%, C1={n_val_1/n_val*100:.2f}%), "
                    f"Sobreposição de Grupos={overlap_grupos}")
        
        assert overlap_grupos == 0, f"Vazamento detectado no Fold {fold_idx}!"
        
        estatisticas_folds.append({
            'fold': fold_idx,
            'total_treino': n_train,
            'classe_0_treino_qtd': n_train_0,
            'classe_0_treino_pct': round((n_train_0 / n_train) * 100, 2),
            'classe_1_treino_qtd': n_train_1,
            'classe_1_treino_pct': round((n_train_1 / n_train) * 100, 2),
            'total_validacao': n_val,
            'classe_0_val_qtd': n_val_0,
            'classe_0_val_pct': round((n_val_0 / n_val) * 100, 2),
            'classe_1_val_qtd': n_val_1,
            'classe_1_val_pct': round((n_val_1 / n_val) * 100, 2),
            'sobreposicao_grupos_treino_val': overlap_grupos
        })
        
    # Salvar mapeamento ID -> fold
    df_folds = pd.DataFrame({
        'ID': df['ID'],
        'fold': folds_assignment,
        'feature_group_id': df['feature_group_id'],
        'target': df[TARGET_COL]
    })
    caminho_mapa = os.path.join(METRICS_DIR, 'validacao_folds.csv')
    df_folds.to_csv(caminho_mapa, index=False, encoding='utf-8')
    logger.info(f"Mapeamento de folds persistido em: {caminho_mapa}")
    
    # Salvar tabela com resumo e distribuição de classes por fold
    df_estatisticas = pd.DataFrame(estatisticas_folds)
    caminho_dist = os.path.join(METRICS_DIR, 'distribuicao_classes_folds.csv')
    df_estatisticas.to_csv(caminho_dist, index=False, encoding='utf-8')
    logger.info(f"Distribuição de classes por fold salva em: {caminho_dist}")
    
    return df_folds, df_estatisticas


def criar_preprocessador_arvores():
    """
    Constrói ColumnTransformer para algoritmos baseados em árvores (Decision Tree e Random Forest):
    - OneHotEncoder nas variáveis nominais (SEX, EDUCATION, MARRIAGE) com handle_unknown='ignore'.
    - Passthrough nas variáveis ordinais e numéricas (árvores são invariantes a transformações monotônicas).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat_nominal', OneHotEncoder(handle_unknown='ignore', sparse_output=False), NOMINAL_COLS)
        ],
        remainder='passthrough'
    )
    return preprocessor


def criar_preprocessador_knn():
    """
    Constrói ColumnTransformer para o algoritmo KNN:
    - OneHotEncoder nas variáveis nominais (SEX, EDUCATION, MARRIAGE) com handle_unknown='ignore'.
    - StandardScaler nas variáveis numéricas contínuas e ordinais, assegurando que
      diferenças de escala (ex: LIMIT_BAL em centenas de milhares vs PAY em unidades)
      não distorçam o cálculo de distâncias euclidianas.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat_nominal', OneHotEncoder(handle_unknown='ignore', sparse_output=False), NOMINAL_COLS),
            ('num_escala', StandardScaler(), NUMERIC_COLS + ORDINAL_COLS)
        ],
        remainder='drop'
    )
    return preprocessor


def obter_pipeline(nome_algoritmo, random_state=42):
    """
    Fábrica de pipelines Scikit-learn para execução controlada sem data leakage.
    Garante que o pré-processamento seja ajustado exclusivamente nos dados de treino de cada fold.
    """
    if nome_algoritmo == 'decision_tree':
        return Pipeline([
            ('prep', criar_preprocessador_arvores()),
            ('clf', DecisionTreeClassifier(random_state=random_state))
        ])
    elif nome_algoritmo == 'random_forest':
        return Pipeline([
            ('prep', criar_preprocessador_arvores()),
            ('clf', RandomForestClassifier(random_state=random_state))
        ])
    elif nome_algoritmo == 'knn':
        return Pipeline([
            ('prep', criar_preprocessador_knn()),
            ('clf', KNeighborsClassifier())
        ])
    elif nome_algoritmo == 'dummy':
        return Pipeline([
            ('clf', DummyClassifier(strategy='most_frequent'))
        ])
    else:
        raise ValueError(f"Algoritmo desconhecido: {nome_algoritmo}")


def executar_preparacao_validacao():
    """Função orquestradora da etapa de preparação da validação."""
    inicio = datetime.now()
    logger.info("=== INÍCIO DA PREPARAÇÃO DE VALIDAÇÃO CRUZADA (FASE 2) ===")
    
    df_folds, df_estatisticas = gerar_divisao_folds()
    
    fim = datetime.now()
    duracao = (fim - inicio).total_seconds()
    logger.info(f"=== PREPARAÇÃO DE VALIDAÇÃO CONCLUÍDA EM {duracao:.2f} SEGUNDOS ===")


if __name__ == '__main__':
    executar_preparacao_validacao()
