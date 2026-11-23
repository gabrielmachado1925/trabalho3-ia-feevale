"""
Script de Treinamento e Execução Experimental (40 Rodadas)
Projeto: Trabalho Acadêmico - Inteligência Artificial | Universidade Feevale
Fase 3: Treinamento, Avaliação e Comparação dos Algoritmos
Autores: Gabriel Machado e João Vitor Lamberti Tonin
Professor: Rodrigo
Data: Novembro/2026
"""

import os
import sys
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support
)

# Importar fábrica de pipelines e definições de colunas
from prepare_validation import (
    obter_pipeline,
    NOMINAL_COLS,
    ORDINAL_COLS,
    NUMERIC_COLS,
    TARGET_COL
)

# Configurações de diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')
LOGS_DIR = os.path.join(BASE_DIR, 'results', 'logs')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Configuração de logging
log_file = os.path.join(LOGS_DIR, 'execucao_experimentos.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configurações do ambiente de execução para reprodutibilidade
CONFIG_AMBIENTE = {
    'hardware': 'Apple M2 (8 núcleos: 4 de desempenho, 4 de eficiência)',
    'memoria_ram': '16 GB Unificada',
    'sistema_operacional': 'macOS (Darwin arm64)',
    'python_versao': sys.version.split()[0],
    'random_seed': 42,
    'n_jobs': 1  # Paralelismo sequencial explícito para reprodutibilidade estrita de tempos
}


def executar_experimentos():
    """
    Executa sequencialmente as 40 rodadas experimentais:
    - 2 Variações (A: Códigos Originais, B: Consolidação)
    - 4 Algoritmos (Dummy, Decision Tree, Random Forest, KNN)
    - 5 Folds (StratifiedGroupKFold persistido)
    """
    inicio_geral = time.perf_counter()
    logger.info("================================================================================")
    logger.info("INÍCIO DA EXECUÇÃO EXPERIMENTAL (FASE 3) - 40 RODADAS CONTROLADAS")
    logger.info(f"Ambiente: {CONFIG_AMBIENTE['hardware']} | Python: {CONFIG_AMBIENTE['python_versao']}")
    logger.info("================================================================================")
    
    # Carregar arquivos processados e mapeamento de folds
    df_a = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'dataset_variacao_a.csv'))
    df_b = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'dataset_variacao_b.csv'))
    df_folds = pd.read_csv(os.path.join(METRICS_DIR, 'validacao_folds.csv'))
    
    # Validações de integridade cruzada
    assert len(df_a) == 30000 and len(df_b) == 30000 and len(df_folds) == 30000, "Inconsistência de tamanho nos datasets"
    assert (df_a['ID'] == df_folds['ID']).all(), "Divergência de IDs entre Variação A e Folds"
    assert (df_b['ID'] == df_folds['ID']).all(), "Divergência de IDs entre Variação B e Folds"
    assert (df_a[TARGET_COL] == df_b[TARGET_COL]).all(), "Divergência de target entre Variações A e B"
    
    pred_cols = NOMINAL_COLS + ORDINAL_COLS + NUMERIC_COLS
    assert 'ID' not in pred_cols and 'feature_group_id' not in pred_cols and TARGET_COL not in pred_cols
    assert len(pred_cols) == 23, f"Esperado 23 preditores, encontrado: {len(pred_cols)}"
    
    variacoes = {
        'variacao_a': df_a,
        'variacao_b': df_b
    }
    
    algoritmos = ['dummy', 'decision_tree', 'random_forest', 'knn']
    
    metricas_rodadas = []
    predicoes_oof_lista = []
    
    rodada_count = 0
    total_rodadas = len(variacoes) * len(algoritmos) * 5
    
    for var_nome, df_var in variacoes.items():
        for algo_nome in algoritmos:
            logger.info(f"\n--- Iniciando Algoritmo: {algo_nome.upper()} na {var_nome.upper()} (5 Folds) ---")
            
            oof_preds = np.zeros(len(df_var), dtype=int)
            oof_probas = np.zeros(len(df_var), dtype=float)
            
            for fold in range(1, 6):
                rodada_count += 1
                idx_train = (df_folds['fold'] != fold).values
                idx_val = (df_folds['fold'] == fold).values
                
                X_train = df_var.loc[idx_train, pred_cols]
                y_train = df_var.loc[idx_train, TARGET_COL].values
                
                X_val = df_var.loc[idx_val, pred_cols]
                y_val = df_var.loc[idx_val, TARGET_COL].values
                val_ids = df_var.loc[idx_val, 'ID'].values
                
                # Instanciar pipelineScikit-learn com configurações estritas
                pipeline = obter_pipeline(algo_nome, random_state=CONFIG_AMBIENTE['random_seed'])
                
                # 1. Medição de Treinamento (Pipeline completo: ColumnTransformer fit + Model fit)
                t_train_start = time.perf_counter()
                pipeline.fit(X_train, y_train)
                t_train = time.perf_counter() - t_train_start
                
                # 2. Medição de Predição (Pipeline completo: Transform + Predict)
                t_pred_start = time.perf_counter()
                y_pred = pipeline.predict(X_val)
                t_pred = time.perf_counter() - t_pred_start
                
                # Probabilidades estimadas para a classe positiva (1 - Inadimplente)
                if hasattr(pipeline, "predict_proba"):
                    y_proba = pipeline.predict_proba(X_val)[:, 1]
                else:
                    y_proba = y_pred.astype(float)
                    
                oof_preds[idx_val] = y_pred
                oof_probas[idx_val] = y_proba
                
                # 3. Cálculo das Métricas
                acc = accuracy_score(y_val, y_pred)
                bal_acc = balanced_accuracy_score(y_val, y_pred)
                
                prec_macro = precision_score(y_val, y_pred, average='macro', zero_division=0)
                rec_macro = recall_score(y_val, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
                
                prec_weighted = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                rec_weighted = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                f1_weighted = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                
                # Métricas por classe
                prf = precision_recall_fscore_support(y_val, y_pred, labels=[0, 1], zero_division=0)
                prec_c0, prec_c1 = prf[0][0], prf[0][1]
                rec_c0, rec_c1 = prf[1][0], prf[1][1]
                f1_c0, f1_c1 = prf[2][0], prf[2][1]
                sup_c0, sup_c1 = prf[3][0], prf[3][1]
                
                # Matriz de Confusão
                cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                
                # Métricas probabilísticas
                if algo_nome == 'dummy':
                    auc_val = 0.5
                    ap_val = float(np.mean(y_val))
                else:
                    auc_val = roc_auc_score(y_val, y_proba)
                    ap_val = average_precision_score(y_val, y_proba)
                    
                logger.info(
                    f"[{rodada_count:02d}/{total_rodadas}] {var_nome.upper()} | {algo_nome.upper()} | Fold {fold} -> "
                    f"Acc: {acc:.4f} | F1-Macro: {f1_macro:.4f} | Rec-C1: {rec_c1:.4f} | F1-C1: {f1_c1:.4f} | "
                    f"T_Train: {t_train:.3f}s | T_Pred: {t_pred:.3f}s"
                )
                
                metricas_rodadas.append({
                    'rodada_id': rodada_count,
                    'variacao': var_nome,
                    'algoritmo': algo_nome,
                    'fold': fold,
                    'treino_total': len(X_train),
                    'val_total': len(X_val),
                    'classe_0_val': int(sup_c0),
                    'classe_1_val': int(sup_c1),
                    'acuracia': acc,
                    'acuracia_balanceada': bal_acc,
                    'precisao_macro': prec_macro,
                    'recall_macro': rec_macro,
                    'f1_macro': f1_macro,
                    'precisao_weighted': prec_weighted,
                    'recall_weighted': rec_weighted,
                    'f1_weighted': f1_weighted,
                    'classe_0_precisao': prec_c0,
                    'classe_0_recall': rec_c0,
                    'classe_0_f1': f1_c0,
                    'classe_0_support': int(sup_c0),
                    'classe_1_precisao': prec_c1,
                    'classe_1_recall': rec_c1,
                    'classe_1_f1': f1_c1,
                    'classe_1_support': int(sup_c1),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tp': int(tp),
                    'roc_auc': auc_val,
                    'average_precision': ap_val,
                    'tempo_treinamento_s': t_train,
                    'tempo_predicao_s': t_pred
                })
                
            # Registrar predições out-of-fold consolidadas deste algoritmo/variação
            for idx_row in range(len(df_var)):
                predicoes_oof_lista.append({
                    'ID': int(df_var.loc[idx_row, 'ID']),
                    'variacao': var_nome,
                    'algoritmo': algo_nome,
                    'fold': int(df_folds.loc[idx_row, 'fold']),
                    'y_real': int(df_var.loc[idx_row, TARGET_COL]),
                    'y_pred': int(oof_preds[idx_row]),
                    'y_prob_classe_1': float(oof_probas[idx_row])
                })
                
    # 4. Salvar tabela detalhada por fold (40 linhas)
    df_metricas_folds = pd.DataFrame(metricas_rodadas)
    caminho_metricas_folds = os.path.join(METRICS_DIR, 'metricas_por_fold.csv')
    df_metricas_folds.to_csv(caminho_metricas_folds, index=False, encoding='utf-8')
    logger.info(f"\nTabela detalhada das 40 rodadas salva em: {caminho_metricas_folds}")
    
    # 5. Salvar predições out-of-fold completas (30.000 x 8 combinações = 240.000 linhas)
    df_oof = pd.DataFrame(predicoes_oof_lista)
    caminho_oof = os.path.join(METRICS_DIR, 'predicoes_out_of_fold.csv')
    df_oof.to_csv(caminho_oof, index=False, encoding='utf-8')
    logger.info(f"Predições out-of-fold consolidadas salvas em: {caminho_oof} ({len(df_oof):,} registros)")
    
    # 6. Treinar e persistir os 8 modelos finais no conjunto completo de 30.000 registros
    logger.info("\n================================================================================")
    logger.info("TREINAMENTO E PERSISTÊNCIA DOS 8 MODELOS FINAIS COMPLETOS (30.000 REGISTROS)")
    logger.info("================================================================================")
    
    modelos_persistidos = []
    for var_nome, df_var in variacoes.items():
        X_full = df_var[pred_cols]
        y_full = df_var[TARGET_COL].values
        
        for algo_nome in algoritmos:
            logger.info(f"Treinando modelo final completo: {algo_nome.upper()} ({var_nome})...")
            pipeline_final = obter_pipeline(algo_nome, random_state=CONFIG_AMBIENTE['random_seed'])
            
            t0_full = time.perf_counter()
            pipeline_final.fit(X_full, y_full)
            t_full = time.perf_counter() - t0_full
            
            nome_arquivo_modelo = f"{algo_nome}_{var_nome}.joblib"
            caminho_modelo = os.path.join(MODELS_DIR, nome_arquivo_modelo)
            joblib.dump(pipeline_final, caminho_modelo, compress=3)
            
            tam_bytes = os.path.getsize(caminho_modelo)
            tam_mb = tam_bytes / (1024 * 1024)
            
            logger.info(f" - Salvo em {caminho_modelo} ({tam_mb:.2f} MB, treino={t_full:.2f}s)")
            
            modelos_persistidos.append({
                'algoritmo': algo_nome,
                'variacao': var_nome,
                'arquivo': nome_arquivo_modelo,
                'tamanho_bytes': tam_bytes,
                'tamanho_mb': round(tam_mb, 3),
                'tempo_treinamento_full_s': round(t_full, 3)
            })
            
    df_modelos = pd.DataFrame(modelos_persistidos)
    df_modelos.to_csv(os.path.join(METRICS_DIR, 'modelos_persistidos.csv'), index=False, encoding='utf-8')
    
    duracao_total = time.perf_counter() - inicio_geral
    logger.info("================================================================================")
    logger.info(f"FASE 3 - EXECUÇÃO EXPERIMENTAL CONCLUÍDA COM SUCESSO EM {duracao_total:.2f} SEGUNDOS")
    logger.info("================================================================================")


if __name__ == '__main__':
    executar_experimentos()
