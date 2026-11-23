"""
Script de Consolidação e Avaliação Estatística dos Resultados Experimentais
Projeto: Trabalho Acadêmico - Inteligência Artificial | Universidade Feevale
Fase 3: Treinamento, Avaliação e Comparação dos Algoritmos
Autores: Gabriel Machado e João Vitor Lamberti Tonin
Professor: Rodrigo
Data: Novembro/2026
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

# Configuração de caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')
LOGS_DIR = os.path.join(BASE_DIR, 'results', 'logs')

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


def formatar_media_desvio(media, desvio, decimais=4):
    """Auxiliar para formatação acadêmica com média e desvio padrão."""
    return f"{media:.{decimais}f} ± {desvio:.{decimais}f}"


def gerar_tabelas_principais(df_folds):
    """
    Gera Tabela 1 (Variação A) e Tabela 2 (Variação B) conforme exigido pelo professor:
    - Algoritmo
    - Acurácia média
    - Precisão macro média
    - Recall macro médio
    - F1-Score macro médio
    - Tempo médio de treinamento em segundos
    """
    logger.info("Gerando Tabela 1 (Variação A) e Tabela 2 (Variação B)...")
    
    ordem_algos = ['dummy', 'decision_tree', 'random_forest', 'knn']
    nomes_apresentacao = {
        'dummy': 'Dummy Classifier (Baseline)',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest',
        'knn': 'K-Nearest Neighbors (KNN)'
    }
    
    for var_nome, arquivo_nome in [
        ('variacao_a', 'metricas_experimentos_variacao_a.csv'),
        ('variacao_b', 'metricas_experimentos_variacao_b.csv')
    ]:
        df_sub = df_folds[df_folds['variacao'] == var_nome]
        
        linhas = []
        for algo in ordem_algos:
            df_algo = df_sub[df_sub['algoritmo'] == algo]
            
            acc_m, acc_s = df_algo['acuracia'].mean(), df_algo['acuracia'].std()
            prec_m, prec_s = df_algo['precisao_macro'].mean(), df_algo['precisao_macro'].std()
            rec_m, rec_s = df_algo['recall_macro'].mean(), df_algo['recall_macro'].std()
            f1_m, f1_s = df_algo['f1_macro'].mean(), df_algo['f1_macro'].std()
            t_m, t_s = df_algo['tempo_treinamento_s'].mean(), df_algo['tempo_treinamento_s'].std()
            
            linhas.append({
                'algoritmo_chave': algo,
                'algoritmo': nomes_apresentacao[algo],
                'acuracia_media': acc_m,
                'acuracia_std': acc_s,
                'acuracia_apresentacao': formatar_media_desvio(acc_m, acc_s, 4),
                'precisao_macro_media': prec_m,
                'precisao_macro_std': prec_s,
                'precisao_macro_apresentacao': formatar_media_desvio(prec_m, prec_s, 4),
                'recall_macro_media': rec_m,
                'recall_macro_std': rec_s,
                'recall_macro_apresentacao': formatar_media_desvio(rec_m, rec_s, 4),
                'f1_macro_media': f1_m,
                'f1_macro_std': f1_s,
                'f1_macro_apresentacao': formatar_media_desvio(f1_m, f1_s, 4),
                'tempo_treinamento_medio_s': t_m,
                'tempo_treinamento_std_s': t_s,
                'tempo_treinamento_apresentacao': formatar_media_desvio(t_m, t_s, 3)
            })
            
        df_tabela = pd.DataFrame(linhas)
        caminho_csv = os.path.join(METRICS_DIR, arquivo_nome)
        df_tabela.to_csv(caminho_csv, index=False, encoding='utf-8')
        logger.info(f"Tabela salva em: {caminho_csv}")


def gerar_metricas_por_classe(df_folds):
    """Gera tabela consolidando desempenho específico para a Classe 0 e Classe 1."""
    logger.info("Gerando tabela de métricas por classe...")
    
    linhas = []
    for var in ['variacao_a', 'variacao_b']:
        for algo in ['dummy', 'decision_tree', 'random_forest', 'knn']:
            sub = df_folds[(df_folds['variacao'] == var) & (df_folds['algoritmo'] == algo)]
            
            for c in [0, 1]:
                p_m, p_s = sub[f'classe_{c}_precisao'].mean(), sub[f'classe_{c}_precisao'].std()
                r_m, r_s = sub[f'classe_{c}_recall'].mean(), sub[f'classe_{c}_recall'].std()
                f_m, f_s = sub[f'classe_{c}_f1'].mean(), sub[f'classe_{c}_f1'].std()
                sup_total = sub[f'classe_{c}_support'].sum()
                
                linhas.append({
                    'variacao': var,
                    'algoritmo': algo,
                    'classe': c,
                    'classe_rotulo': 'Não Inadimplente (Classe 0)' if c == 0 else 'Inadimplente (Classe 1)',
                    'precisao_media': p_m,
                    'precisao_std': p_s,
                    'precisao_apresentacao': formatar_media_desvio(p_m, p_s, 4),
                    'recall_medio': r_m,
                    'recall_std': r_s,
                    'recall_apresentacao': formatar_media_desvio(r_m, r_s, 4),
                    'f1_medio': f_m,
                    'f1_std': f_s,
                    'f1_apresentacao': formatar_media_desvio(f_m, f_s, 4),
                    'suporte_total': int(sup_total)
                })
                
    df_classes = pd.DataFrame(linhas)
    caminho = os.path.join(METRICS_DIR, 'metricas_por_classe.csv')
    df_classes.to_csv(caminho, index=False, encoding='utf-8')
    logger.info(f"Métricas por classe salvas em: {caminho}")


def gerar_matrizes_confusao_consolidadas(df_oof):
    """Gera matrizes de confusão a partir das 30.000 predições out-of-fold."""
    logger.info("Gerando matrizes de confusão consolidadas out-of-fold...")
    
    linhas = []
    for var in ['variacao_a', 'variacao_b']:
        for algo in ['dummy', 'decision_tree', 'random_forest', 'knn']:
            sub = df_oof[(df_oof['variacao'] == var) & (df_oof['algoritmo'] == algo)]
            
            assert len(sub) == 30000, f"Predições OOF incompletas para {var}/{algo}: {len(sub)}"
            
            cm = confusion_matrix(sub['y_real'], sub['y_pred'], labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            # Métricas out-of-fold globais
            acc = (tp + tn) / len(sub)
            sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precisao_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1_pos = 2 * precisao_pos * sensibilidade / (precisao_pos + sensibilidade) if (precisao_pos + sensibilidade) > 0 else 0.0
            
            if algo == 'dummy':
                auc = 0.5
                ap = float(np.mean(sub['y_real']))
            else:
                auc = roc_auc_score(sub['y_real'], sub['y_prob_classe_1'])
                ap = average_precision_score(sub['y_real'], sub['y_prob_classe_1'])
                
            linhas.append({
                'variacao': var,
                'algoritmo': algo,
                'total_amostras': len(sub),
                'verdadeiros_negativos_TN': int(tn),
                'falsos_positivos_FP': int(fp),
                'falsos_negativos_FN': int(fn),
                'verdadeiros_positivos_TP': int(tp),
                'taxa_acuracia_oof': round(acc, 5),
                'sensibilidade_recall_c1': round(sensibilidade, 5),
                'especificidade_c0': round(especificidade, 5),
                'precisao_c1': round(precisao_pos, 5),
                'f1_c1': round(f1_pos, 5),
                'roc_auc_oof': round(auc, 5),
                'average_precision_oof': round(ap, 5)
            })
            
    df_cm = pd.DataFrame(linhas)
    caminho = os.path.join(METRICS_DIR, 'matrizes_confusao.csv')
    df_cm.to_csv(caminho, index=False, encoding='utf-8')
    logger.info(f"Matrizes de confusão salvas em: {caminho}")


def gerar_comparacao_variacoes(df_folds):
    """Compara Variação A vs Variação B para cada algoritmo."""
    logger.info("Gerando tabela comparativa Variação A vs Variação B...")
    
    linhas = []
    algoritmos = ['dummy', 'decision_tree', 'random_forest', 'knn']
    
    for algo in algoritmos:
        sub_a = df_folds[(df_folds['variacao'] == 'variacao_a') & (df_folds['algoritmo'] == algo)]
        sub_b = df_folds[(df_folds['variacao'] == 'variacao_b') & (df_folds['algoritmo'] == algo)]
        
        acc_a, acc_b = sub_a['acuracia'].mean(), sub_b['acuracia'].mean()
        f1_m_a, f1_m_b = sub_a['f1_macro'].mean(), sub_b['f1_macro'].mean()
        rec1_a, rec1_b = sub_a['classe_1_recall'].mean(), sub_b['classe_1_recall'].mean()
        f1_1_a, f1_1_b = sub_a['classe_1_f1'].mean(), sub_b['classe_1_f1'].mean()
        t_a, t_b = sub_a['tempo_treinamento_s'].mean(), sub_b['tempo_treinamento_s'].mean()
        
        linhas.append({
            'algoritmo': algo,
            'acuracia_var_a': acc_a,
            'acuracia_var_b': acc_b,
            'delta_acuracia': acc_b - acc_a,
            'f1_macro_var_a': f1_m_a,
            'f1_macro_var_b': f1_m_b,
            'delta_f1_macro': f1_m_b - f1_m_a,
            'recall_c1_var_a': rec1_a,
            'recall_c1_var_b': rec1_b,
            'delta_recall_c1': rec1_b - rec1_a,
            'f1_c1_var_a': f1_1_a,
            'f1_c1_var_b': f1_1_b,
            'delta_f1_c1': f1_1_b - f1_1_a,
            'tempo_treino_s_var_a': t_a,
            'tempo_treino_s_var_b': t_b,
            'delta_tempo_s': t_b - t_a
        })
        
    df_comp = pd.DataFrame(linhas)
    caminho = os.path.join(METRICS_DIR, 'comparacao_variacoes.csv')
    df_comp.to_csv(caminho, index=False, encoding='utf-8')
    logger.info(f"Comparação de variações salva em: {caminho}")


def gerar_tempos_processamento(df_folds):
    """Gera tabela consolidando os tempos de treino e predição."""
    logger.info("Gerando tabela de tempos de processamento...")
    
    linhas = []
    for var in ['variacao_a', 'variacao_b']:
        for algo in ['dummy', 'decision_tree', 'random_forest', 'knn']:
            sub = df_folds[(df_folds['variacao'] == var) & (df_folds['algoritmo'] == algo)]
            
            t_train_m, t_train_s = sub['tempo_treinamento_s'].mean(), sub['tempo_treinamento_s'].std()
            t_train_tot = sub['tempo_treinamento_s'].sum()
            
            t_pred_m, t_pred_s = sub['tempo_predicao_s'].mean(), sub['tempo_predicao_s'].std()
            t_pred_tot = sub['tempo_predicao_s'].sum()
            
            # Latência por 1.000 amostras (média de amostras de validação ~ 6.000)
            latencia_1k_ms = (t_pred_m / 6000.0) * 1000.0 * 1000.0
            
            linhas.append({
                'variacao': var,
                'algoritmo': algo,
                'tempo_treino_medio_s': t_train_m,
                'tempo_treino_std_s': t_train_s,
                'tempo_treino_total_5folds_s': t_train_tot,
                'tempo_pred_medio_s': t_pred_m,
                'tempo_pred_std_s': t_pred_s,
                'tempo_pred_total_5folds_s': t_pred_tot,
                'latencia_predicao_por_1k_amostras_ms': latencia_1k_ms
            })
            
    df_tempos = pd.DataFrame(linhas)
    caminho = os.path.join(METRICS_DIR, 'tempos_processamento.csv')
    df_tempos.to_csv(caminho, index=False, encoding='utf-8')
    logger.info(f"Tempos de processamento salvos em: {caminho}")


def executar_avaliacao():
    """Função orquestradora da etapa de consolidação estatística."""
    logger.info("=== INÍCIO DA CONSOLIDAÇÃO E AVALIAÇÃO DE RESULTADOS ===")
    
    df_folds = pd.read_csv(os.path.join(METRICS_DIR, 'metricas_por_fold.csv'))
    df_oof = pd.read_csv(os.path.join(METRICS_DIR, 'predicoes_out_of_fold.csv'))
    
    gerar_tabelas_principais(df_folds)
    gerar_metricas_por_classe(df_folds)
    gerar_matrizes_confusao_consolidadas(df_oof)
    gerar_comparacao_variacoes(df_folds)
    gerar_tempos_processamento(df_folds)
    
    logger.info("=== CONSOLIDAÇÃO E AVALIAÇÃO CONCLUÍDAS COM SUCESSO ===")


if __name__ == '__main__':
    executar_avaliacao()
