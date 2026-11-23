"""
Script de Validações Automatizadas dos Resultados da Fase 3
Projeto: Trabalho Acadêmico - Inteligência Artificial | Universidade Feevale
Fase 3: Treinamento, Avaliação e Comparação dos Algoritmos
Autores: Gabriel Machado e João Vitor Lamberti Tonin
Professor: Rodrigo
Data: Novembro/2026
"""

import os
import sys
import hashlib
import logging
import pandas as pd
import numpy as np
import joblib

# Configuração de caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')
CHARTS_DIR = os.path.join(BASE_DIR, 'results', 'charts')
LOGS_DIR = os.path.join(BASE_DIR, 'results', 'logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

HASHES_ESPERADOS = {
    'default of credit card clients.xls': '30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933',
    'default+of+credit+card+clients.zip': '56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602',
    'data.csv': '45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0'
}


def calcular_sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def testar_integridade_datasets_brutos():
    logger.info("Teste 1: Verificando integridade dos arquivos originais em data/raw/...")
    for arq, h_esp in HASHES_ESPERADOS.items():
        caminho = os.path.join(DATA_RAW_DIR, arq)
        assert os.path.exists(caminho), f"Arquivo não encontrado: {caminho}"
        h_calc = calcular_sha256(caminho)
        assert h_calc == h_esp, f"Divergência de hash em {arq}: esperado {h_esp}, obtido {h_calc}"
    logger.info("-> Teste 1 APROVADO: Arquivos originais rigorosamente intactos.")


def testar_completude_40_rodadas():
    logger.info("Teste 2: Verificando completude e consistência das 40 rodadas experimentais...")
    caminho = os.path.join(METRICS_DIR, 'metricas_por_fold.csv')
    assert os.path.exists(caminho), f"Arquivo ausente: {caminho}"
    df = pd.read_csv(caminho)
    
    assert len(df) == 40, f"Esperado 40 rodadas, encontrado: {len(df)}"
    assert df['rodada_id'].nunique() == 40, "Identificadores de rodada duplicados"
    
    # 2 variações x 4 algoritmos x 5 folds
    combinacoes = df.groupby(['variacao', 'algoritmo', 'fold']).size()
    assert (combinacoes == 1).all() and len(combinacoes) == 40, "Faltam ou há rodadas duplicadas"
    
    # Validação de intervalos numéricos das métricas
    metricas_intervalo_0_1 = [
        'acuracia', 'acuracia_balanceada', 'precisao_macro', 'recall_macro', 'f1_macro',
        'precisao_weighted', 'recall_weighted', 'f1_weighted',
        'classe_0_precisao', 'classe_0_recall', 'classe_0_f1',
        'classe_1_precisao', 'classe_1_recall', 'classe_1_f1',
        'roc_auc', 'average_precision'
    ]
    for m in metricas_intervalo_0_1:
        assert (df[m] >= 0.0).all() and (df[m] <= 1.0).all(), f"Métrica {m} fora do intervalo [0, 1]!"
        
    # Validação dos tempos
    assert (df['tempo_treinamento_s'] >= 0.0).all(), "Tempo de treinamento negativo detectado"
    assert (df['tempo_predicao_s'] >= 0.0).all(), "Tempo de predição negativo detectado"
    
    logger.info("-> Teste 2 APROVADO: 40 rodadas completas, sem duplicatas e métricas em intervalos válidos.")


def testar_predicoes_out_of_fold():
    logger.info("Teste 3: Verificando integridade das 240.000 predições out-of-fold...")
    caminho = os.path.join(METRICS_DIR, 'predicoes_out_of_fold.csv')
    assert os.path.exists(caminho), f"Arquivo ausente: {caminho}"
    df_oof = pd.read_csv(caminho)
    
    # 8 combinações de (variacao, algoritmo) x 30.000 instâncias = 240.000 registros
    assert len(df_oof) == 240000, f"Esperado 240.000 linhas, encontrado: {len(df_oof)}"
    
    for var in ['variacao_a', 'variacao_b']:
        for algo in ['dummy', 'decision_tree', 'random_forest', 'knn']:
            sub = df_oof[(df_oof['variacao'] == var) & (df_oof['algoritmo'] == algo)]
            assert len(sub) == 30000, f"Contagem out-of-fold incompleta para {var}/{algo}: {len(sub)}"
            assert sub['ID'].nunique() == 30000, f"IDs duplicados em {var}/{algo}"
            assert set(sub['y_pred'].unique()).issubset({0, 1}), f"Classes inválidas previstas em {var}/{algo}"
            assert (sub['y_prob_classe_1'] >= 0.0).all() and (sub['y_prob_classe_1'] <= 1.0).all(), "Probabilidade fora de [0, 1]"
            
            # Verificar preservação estrita do target
            vc_real = sub['y_real'].value_counts()
            assert vc_real[0] == 23364 and vc_real[1] == 6636, f"Divergência de target real em {var}/{algo}"
            
    logger.info("-> Teste 3 APROVADO: Exatamente 30.000 predições out-of-fold por combinação com mapeamento íntegro.")


def testar_matrizes_confusao():
    logger.info("Teste 4: Verificando coerência das matrizes de confusão consolidadas...")
    caminho = os.path.join(METRICS_DIR, 'matrizes_confusao.csv')
    assert os.path.exists(caminho), f"Arquivo ausente: {caminho}"
    df_cm = pd.read_csv(caminho)
    
    assert len(df_cm) == 8, f"Esperado 8 matrizes de confusão, encontrado: {len(df_cm)}"
    for idx, row in df_cm.iterrows():
        soma = row['verdadeiros_negativos_TN'] + row['falsos_positivos_FP'] + row['falsos_negativos_FN'] + row['verdadeiros_positivos_TP']
        assert soma == 30000, f"Soma da matriz de confusão != 30.000 na linha {idx}: {soma}"
        assert row['verdadeiros_negativos_TN'] + row['falsos_positivos_FP'] == 23364, "Soma da classe 0 divergente"
        assert row['falsos_negativos_FN'] + row['verdadeiros_positivos_TP'] == 6636, "Soma da classe 1 divergente"
        
    logger.info("-> Teste 4 APROVADO: Todas as matrizes de confusão fecham perfeitamente em 30.000 instâncias.")


def testar_modelos_persistidos():
    logger.info("Teste 5: Verificando existência, integridade e teste de carga dos 8 modelos salvos em models/...")
    modelos_esperados = [
        'dummy_variacao_a.joblib', 'dummy_variacao_b.joblib',
        'decision_tree_variacao_a.joblib', 'decision_tree_variacao_b.joblib',
        'random_forest_variacao_a.joblib', 'random_forest_variacao_b.joblib',
        'knn_variacao_a.joblib', 'knn_variacao_b.joblib'
    ]
    
    df_b = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'dataset_variacao_b.csv'))
    # Test sample com 5 instâncias
    from prepare_validation import NOMINAL_COLS, ORDINAL_COLS, NUMERIC_COLS
    pred_cols = NOMINAL_COLS + ORDINAL_COLS + NUMERIC_COLS
    X_sample = df_b[pred_cols].head(5)
    
    for mod_nome in modelos_esperados:
        caminho_mod = os.path.join(MODELS_DIR, mod_nome)
        assert os.path.exists(caminho_mod), f"Modelo não encontrado: {caminho_mod}"
        tam = os.path.getsize(caminho_mod)
        assert tam > 0, f"Modelo vazio: {caminho_mod}"
        
        # Testar carregamento via joblib
        mod_carregado = joblib.load(caminho_mod)
        preds_sample = mod_carregado.predict(X_sample)
        assert len(preds_sample) == 5, f"Falha na inferência de teste do modelo {mod_nome}"
        assert set(preds_sample).issubset({0, 1}), f"Predições com classes anômalas em {mod_nome}"
        logger.info(f" - {mod_nome}: {tam / (1024*1024):.2f} MB | Carregado e testado com sucesso.")
        
    logger.info("-> Teste 5 APROVADO: Os 8 modelos finais estão íntegros, funcionais e serializados.")


def testar_existencia_graficos_e_tabelas():
    logger.info("Teste 6: Verificando existência física de todos os 20 gráficos e tabelas de métricas...")
    graficos_esperados = [
        '01_distribuicao_target.png', '02_distribuicao_idade.png', '03_distribuicao_limite_credito.png',
        '04_distribuicao_demograficas.png', '05_taxas_inadimplencia_demograficas.png',
        '06_historico_pagamento_vs_inadimplencia.png', '07_matriz_correlacao_numericas.png',
        '08_evolucao_faturas_pagamentos.png', '09_comparacao_acuracia.png',
        '10_comparacao_precisao_macro.png', '11_comparacao_recall_macro.png',
        '12_comparacao_f1_macro.png', '13_comparacao_tempo_treinamento.png',
        '14_comparacao_tempo_predicao.png', '15_comparacao_recall_classe_inadimplente.png',
        '16_comparacao_f1_classe_inadimplente.png', '17_matrizes_confusao_painel.png',
        '18_desempenho_classes_0_vs_1.png', '19_impacto_consolidacao_variacao_a_vs_b.png',
        '20_curvas_roc_precision_recall.png'
    ]
    for g in graficos_esperados:
        cg = os.path.join(CHARTS_DIR, g)
        assert os.path.exists(cg), f"Gráfico ausente: {cg}"
        assert os.path.getsize(cg) > 1000, f"Gráfico corrompido ou vazio: {cg}"
        
    tabelas_esperadas = [
        'caracterizacao_atributos.csv', 'desbalanceamento_categoricas.csv',
        'distribuicao_categoricas.csv', 'distribuicao_classes_folds.csv',
        'distribuicao_target.csv', 'estatisticas_numericas.csv',
        'resumo_preprocessamento.csv', 'taxa_inadimplencia_demograficas.csv',
        'validacao_folds.csv', 'valores_inconsistentes.csv',
        'metricas_por_fold.csv', 'metricas_experimentos_variacao_a.csv',
        'metricas_experimentos_variacao_b.csv', 'metricas_por_classe.csv',
        'matrizes_confusao.csv', 'comparacao_variacoes.csv',
        'tempos_processamento.csv', 'predicoes_out_of_fold.csv',
        'modelos_persistidos.csv'
    ]
    for t in tabelas_esperadas:
        ct = os.path.join(METRICS_DIR, t)
        assert os.path.exists(ct), f"Tabela ausente: {ct}"
        assert os.path.getsize(ct) > 10, f"Tabela vazia: {ct}"
        
    logger.info("-> Teste 6 APROVADO: Todos os 20 gráficos e 19 tabelas CSV validados com sucesso.")


def executar_validacao_final():
    logger.info("================================================================================")
    logger.info("INICIANDO SUÍTE FINAL DE VALIDAÇÕES DA FASE 3")
    logger.info("================================================================================")
    
    testar_integridade_datasets_brutos()
    testar_completude_40_rodadas()
    testar_predicoes_out_of_fold()
    testar_matrizes_confusao()
    testar_modelos_persistidos()
    testar_existencia_graficos_e_tabelas()
    
    logger.info("================================================================================")
    logger.info("SUCESSO TOTAL: TODAS AS VALIDAÇÕES DA FASE 3 FORAM 100% APROVADAS")
    logger.info("================================================================================")


if __name__ == '__main__':
    executar_validacao_final()
