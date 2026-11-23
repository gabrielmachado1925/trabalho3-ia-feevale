"""
Script de Validação e Testes Automatizados dos Dados e Pré-processamento
Projeto: Trabalho Acadêmico - Inteligência Artificial | Universidade Feevale
Fase 2: Pré-processamento e Preparação dos Experimentos
Autores: Gabriel Machado e João Tonin
Professor: Rodrigo
Data: Novembro/2026
"""

import os
import sys
import hashlib
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Importar configurações e gerador de pipelines
from prepare_validation import (
    obter_pipeline,
    NOMINAL_COLS,
    ORDINAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    AUX_COLS
)

# Configuração de caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')
LOGS_DIR = os.path.join(BASE_DIR, 'results', 'logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Hashes SHA-256 esperados dos arquivos originais oficiais da UCI
HASHES_ESPERADOS = {
    'default of credit card clients.xls': '30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933',
    'default+of+credit+card+clients.zip': '56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602',
    'data.csv': '45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0'
}


def calcular_sha256(caminho_arquivo):
    """Calcula o hash SHA-256 de um arquivo."""
    hasher = hashlib.sha256()
    with open(caminho_arquivo, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def testar_integridade_arquivos_brutos():
    """Valida a integridade criptográfica dos arquivos brutos da UCI."""
    logger.info("Teste 1: Verificando integridade SHA-256 dos arquivos brutos...")
    for nome_arq, hash_esperado in HASHES_ESPERADOS.items():
        caminho = os.path.join(DATA_RAW_DIR, nome_arq)
        assert os.path.exists(caminho), f"Arquivo não encontrado: {caminho}"
        hash_calculado = calcular_sha256(caminho)
        assert hash_calculado == hash_esperado, (
            f"Hash SHA-256 divergente em {nome_arq}!\n"
            f"Esperado: {hash_esperado}\n"
            f"Calculado: {hash_calculado}"
        )
    logger.info("-> Teste 1 APROVADO: Todos os arquivos brutos estão intactos.")


def testar_preservacao_dimensoes_e_classes():
    """Verifica se os datasets processados preservam exatamente 30.000 instâncias e a distribuição do target."""
    logger.info("Teste 2: Verificando dimensões e integridade de classes nas variações A e B...")
    
    caminho_a = os.path.join(DATA_PROCESSED_DIR, 'dataset_variacao_a.csv')
    caminho_b = os.path.join(DATA_PROCESSED_DIR, 'dataset_variacao_b.csv')
    
    assert os.path.exists(caminho_a), f"Arquivo ausente: {caminho_a}"
    assert os.path.exists(caminho_b), f"Arquivo ausente: {caminho_b}"
    
    df_a = pd.read_csv(caminho_a)
    df_b = pd.read_csv(caminho_b)
    
    for nome, df in [('Variação A', df_a), ('Variação B', df_b)]:
        assert len(df) == 30000, f"{nome}: Quantidade de linhas incorreta ({len(df)} != 30.000)"
        assert df.shape[1] == 26, f"{nome}: Quantidade de colunas incorreta ({df.shape[1]} != 26)"
        assert df.isnull().sum().sum() == 0, f"{nome}: Valores nulos detectados"
        
        vc = df[TARGET_COL].value_counts()
        assert vc[0] == 23364, f"{nome}: Classe 0 com contagem incorreta ({vc[0]} != 23.364)"
        assert vc[1] == 6636, f"{nome}: Classe 1 com contagem incorreta ({vc[1]} != 6.636)"
        
    logger.info("-> Teste 2 APROVADO: 30.000 registros e distribuição 23.364/6.636 rigorosamente preservados.")


def testar_regras_transformacao_categorias():
    """Valida se a Variação A preserva códigos originais e a Variação B consolida exatamente como aprovado."""
    logger.info("Teste 3: Verificando aplicação estrita das regras de categorias...")
    
    df_a = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'dataset_variacao_a.csv'))
    df_b = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'dataset_variacao_b.csv'))
    
    # Variação A deve conter códigos não documentados originais
    assert set(df_a['EDUCATION'].unique()) == {0, 1, 2, 3, 4, 5, 6}, "Variação A não preservou códigos de EDUCATION"
    assert set(df_a['MARRIAGE'].unique()) == {0, 1, 2, 3}, "Variação A não preservou códigos de MARRIAGE"
    
    # Variação B deve conter estritamente as categorias consolidadas aprovadas
    assert set(df_b['EDUCATION'].unique()) == {1, 2, 3, 4}, f"Variação B contém categorias inválidas em EDUCATION: {df_b['EDUCATION'].unique()}"
    assert set(df_b['MARRIAGE'].unique()) == {1, 2, 3}, f"Variação B contém categorias inválidas em MARRIAGE: {df_b['MARRIAGE'].unique()}"
    
    # Quantidades consolidadas na Variação B
    # 4 (Outros) em EDUCATION = 123 (orig 4) + 14 (orig 0) + 280 (orig 5) + 51 (orig 6) = 468
    assert (df_b['EDUCATION'] == 4).sum() == 468, "Consolidação de EDUCATION com contagem divergente"
    # 3 (Outros) em MARRIAGE = 323 (orig 3) + 54 (orig 0) = 377
    assert (df_b['MARRIAGE'] == 3).sum() == 377, "Consolidação de MARRIAGE com contagem divergente"
    
    logger.info("-> Teste 3 APROVADO: Regras de consolidação e preservação executadas com precisão.")


def testar_isolamento_folds_e_ausencia_vazamento():
    """Valida se a partição de 5 folds é reprodutível e livre de vazamento de dados."""
    logger.info("Teste 4: Verificando isolamento de grupos e integridade dos 5 folds...")
    
    caminho_mapa = os.path.join(METRICS_DIR, 'validacao_folds.csv')
    caminho_dist = os.path.join(METRICS_DIR, 'distribuicao_classes_folds.csv')
    
    assert os.path.exists(caminho_mapa), f"Arquivo ausente: {caminho_mapa}"
    assert os.path.exists(caminho_dist), f"Arquivo ausente: {caminho_dist}"
    
    df_folds = pd.read_csv(caminho_mapa)
    assert len(df_folds) == 30000, "Mapeamento de folds não contém 30.000 registros"
    assert set(df_folds['fold'].unique()) == {1, 2, 3, 4, 5}, "Conjunto de folds inválido"
    
    # Verificar sobreposição de grupos para cada fold
    for fold in range(1, 6):
        train_groups = set(df_folds[df_folds['fold'] != fold]['feature_group_id'])
        val_groups = set(df_folds[df_folds['fold'] == fold]['feature_group_id'])
        overlap = train_groups.intersection(val_groups)
        assert len(overlap) == 0, f"VAZAMENTO DE DADOS DETECTADO NO FOLD {fold}! Sobreposição: {len(overlap)}"
        
    logger.info("-> Teste 4 APROVADO: 0 vazamento de perfis idênticos entre treino e validação em todos os 5 folds.")


def testar_viabilidade_algoritmos():
    """
    Executa um ensaio seco (dry-run) de 1 fold para certificar a viabilidade técnica
    dos três algoritmos planejados + dummy com os pipelines Scikit-learn propostos.
    """
    logger.info("Teste 5: Ensaio de viabilidade de execução dos algoritmos (dry-run no Fold 1)...")
    
    df_b = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'dataset_variacao_b.csv'))
    df_folds = pd.read_csv(os.path.join(METRICS_DIR, 'validacao_folds.csv'))
    
    # Separar preditores X e alvo y
    # ID e feature_group_id são RIGOROSAMENTE excluídos de X
    pred_cols = NOMINAL_COLS + ORDINAL_COLS + NUMERIC_COLS
    assert 'ID' not in pred_cols, "CRÍTICO: ID incluído nos preditores!"
    assert 'feature_group_id' not in pred_cols, "CRÍTICO: feature_group_id incluído nos preditores!"
    assert TARGET_COL not in pred_cols, "CRÍTICO: target incluído nos preditores!"
    assert len(pred_cols) == 23, f"Quantidade de preditores divergente ({len(pred_cols)} != 23)"
    
    idx_train = df_folds['fold'] != 1
    idx_val = df_folds['fold'] == 1
    
    X_train = df_b.loc[idx_train, pred_cols]
    y_train = df_b.loc[idx_train, TARGET_COL]
    
    X_val = df_b.loc[idx_val, pred_cols]
    y_val = df_b.loc[idx_val, TARGET_COL]
    
    algoritmos = ['dummy', 'decision_tree', 'random_forest', 'knn']
    for nome in algoritmos:
        pipeline = obter_pipeline(nome)
        # Ajustar pipeline no treino do fold 1
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        acc = accuracy_score(y_val, preds)
        assert len(preds) == len(X_val), f"Tamanho de predições divergente em {nome}"
        assert acc >= 0.0, f"Acurácia inválida em {nome}"
        logger.info(f" - Algoritmo '{nome}': execução técnica concluída com sucesso (acurácia teste fold 1: {acc:.4f})")
        
    logger.info("-> Teste 5 APROVADO: Pipelines e algoritmos 100% compatíveis e prontos para a Fase 3.")


def executar_validacao_completa():
    """Executa a suíte de testes de ponta a ponta."""
    inicio = datetime.now()
    logger.info("========================================================")
    logger.info("INICIANDO SUÍTE DE VALIDAÇÕES AUTOMATIZADAS DA FASE 2")
    logger.info("========================================================")
    
    testar_integridade_arquivos_brutos()
    testar_preservacao_dimensoes_e_classes()
    testar_regras_transformacao_categorias()
    testar_isolamento_folds_e_ausencia_vazamento()
    testar_viabilidade_algoritmos()
    
    duracao = (datetime.now() - inicio).total_seconds()
    logger.info("========================================================")
    logger.info(f"TODAS AS VALIDAÇÕES FORAM APROVADAS COM SUCESSO EM {duracao:.2f}s")
    logger.info("========================================================")


if __name__ == '__main__':
    executar_validacao_completa()
