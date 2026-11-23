"""
Script de Geração de Gráficos Acadêmicos dos Resultados Experimentais
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

# Configurações de ambiente headless para Matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# Configuração de diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')
CHARTS_DIR = os.path.join(BASE_DIR, 'results', 'charts')
LOGS_DIR = os.path.join(BASE_DIR, 'results', 'logs')

os.makedirs(CHARTS_DIR, exist_ok=True)

# Configuração de estilo visual acadêmico
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

logger = logging.getLogger(__name__)

# Paleta de cores padrão dos modelos
CORES_ALGOS = {
    'dummy': '#7f7f7f',          # Cinza (Baseline)
    'decision_tree': '#1f77b4',  # Azul
    'random_forest': '#2ca02c',  # Verde
    'knn': '#ff7f0e'             # Laranja
}
NOMES_ALGOS = {
    'dummy': 'Dummy (Baseline)',
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
    'knn': 'KNN'
}


def plot_barras_comparativas_com_erro(df_folds, metrica_col, titulo, ylabel, nome_arquivo, ylim=(0, 1.0)):
    """Gera gráfico de barras comparando os 4 algoritmos nas duas variações com barras de erro (std)."""
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    
    algos = ['dummy', 'decision_tree', 'random_forest', 'knn']
    x = np.arange(len(algos))
    largura = 0.35
    
    # Médias e desvios Variação A
    sub_a = df_folds[df_folds['variacao'] == 'variacao_a']
    means_a = [sub_a[sub_a['algoritmo'] == a][metrica_col].mean() for a in algos]
    stds_a = [sub_a[sub_a['algoritmo'] == a][metrica_col].std() for a in algos]
    
    # Médias e desvios Variação B
    sub_b = df_folds[df_folds['variacao'] == 'variacao_b']
    means_b = [sub_b[sub_b['algoritmo'] == a][metrica_col].mean() for a in algos]
    stds_b = [sub_b[sub_b['algoritmo'] == a][metrica_col].std() for a in algos]
    
    rects1 = ax.bar(x - largura/2, means_a, largura, yerr=stds_a, capsize=4,
                    label='Variação A (Ref. Códigos Originais)', color='#2b5c8f', edgecolor='#222', alpha=0.9)
    rects2 = ax.bar(x + largura/2, means_b, largura, yerr=stds_b, capsize=4,
                    label='Variação B (Consolidação de Categorias)', color='#41b6c4', edgecolor='#222', alpha=0.9)
    
    ax.set_title(titulo, pad=15, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([NOMES_ALGOS[a] for a in algos])
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc='lower right' if 'Tempo' not in titulo else 'upper right')
    
    # Rótulos nas barras
    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            if h > 0.0001:
                formato = f'{h:.4f}' if 'Tempo' not in titulo else f'{h:.3f}s'
                ax.annotate(formato,
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 5), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=0)
                
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, nome_arquivo))
    plt.close(fig)


def gerar_graficos_individuais(df_folds):
    """Gera gráficos de barras de 1 a 8 comparando as métricas e tempos."""
    # 1. Acurácia
    plot_barras_comparativas_com_erro(
        df_folds, 'acuracia',
        'Comparação da Acurácia Média entre os Algoritmos (Validação Cruzada 5 Folds)',
        'Acurácia Média', '09_comparacao_acuracia.png', ylim=(0.6, 0.9)
    )
    # 2. Precisão Macro
    plot_barras_comparativas_com_erro(
        df_folds, 'precisao_macro',
        'Comparação da Precisão Macro Média entre os Algoritmos',
        'Precisão Macro Média', '10_comparacao_precisao_macro.png', ylim=(0.3, 0.85)
    )
    # 3. Recall Macro
    plot_barras_comparativas_com_erro(
        df_folds, 'recall_macro',
        'Comparação do Recall Macro Médio entre os Algoritmos',
        'Recall Macro Médio', '11_comparacao_recall_macro.png', ylim=(0.4, 0.75)
    )
    # 4. F1-Score Macro (Métrica Principal)
    plot_barras_comparativas_com_erro(
        df_folds, 'f1_macro',
        'Comparação do F1-Score Macro Médio (Métrica Principal de Decisão)',
        'F1-Score Macro Médio', '12_comparacao_f1_macro.png', ylim=(0.35, 0.75)
    )
    # 5. Tempo de Treinamento
    plot_barras_comparativas_com_erro(
        df_folds, 'tempo_treinamento_s',
        'Comparação do Tempo Médio de Treinamento por Fold (Segundos)',
        'Tempo de Treinamento (s)', '13_comparacao_tempo_treinamento.png', ylim=(0, 5.2)
    )
    # 6. Tempo de Predição
    plot_barras_comparativas_com_erro(
        df_folds, 'tempo_predicao_s',
        'Comparação do Tempo Médio de Predição por Fold (~6.000 Amostras)',
        'Tempo de Predição (s)', '14_comparacao_tempo_predicao.png', ylim=(0, 0.22)
    )
    # 7. Recall da Classe Inadimplente (Classe 1)
    plot_barras_comparativas_com_erro(
        df_folds, 'classe_1_recall',
        'Sensibilidade / Recall da Classe Inadimplente (Detecção de Default)',
        'Recall Classe 1 (Inadimplente)', '15_comparacao_recall_classe_inadimplente.png', ylim=(0, 0.55)
    )
    # 8. F1-Score da Classe Inadimplente (Classe 1)
    plot_barras_comparativas_com_erro(
        df_folds, 'classe_1_f1',
        'F1-Score Específico da Classe Inadimplente (Classe 1)',
        'F1-Score Classe 1 (Inadimplente)', '16_comparacao_f1_classe_inadimplente.png', ylim=(0, 0.55)
    )


def gerar_grafico_matrizes_confusao(df_oof):
    """Gera painel com matrizes de confusão para os 3 algoritmos principais em ambas as variações."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=300)
    
    algos = ['decision_tree', 'random_forest', 'knn']
    nomes_titulos = {'decision_tree': 'Decision Tree', 'random_forest': 'Random Forest', 'knn': 'KNN'}
    
    for row_idx, var in enumerate(['variacao_a', 'variacao_b']):
        var_rotulo = 'Variação A (Códigos Originais)' if var == 'variacao_a' else 'Variação B (Consolidação)'
        for col_idx, algo in enumerate(algos):
            ax = axes[row_idx, col_idx]
            sub = df_oof[(df_oof['variacao'] == var) & (df_oof['algoritmo'] == algo)]
            
            cm = confusion_matrix(sub['y_real'], sub['y_pred'], labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            # Formatação percentual
            cm_pct = cm / cm.sum() * 100
            rotulos = np.array([
                [f"VN: {tn:,}\n({cm_pct[0,0]:.1f}%)", f"FP: {fp:,}\n({cm_pct[0,1]:.1f}%)"],
                [f"FN: {fn:,}\n({cm_pct[1,0]:.1f}%)", f"VP: {tp:,}\n({cm_pct[1,1]:.1f}%)"]
            ])
            
            sns.heatmap(cm, annot=rotulos, fmt='', cmap='Blues' if row_idx == 0 else 'YlGnBu',
                        cbar=False, ax=ax, annot_kws={'size': 9, 'fontweight': 'bold'},
                        xticklabels=['Pred. Adimp. (0)', 'Pred. Inadimp. (1)'],
                        yticklabels=['Real Adimp. (0)', 'Real Inadimp. (1)'])
            
            ax.set_title(f"{nomes_titulos[algo]}\n[{var_rotulo}]", fontsize=10, pad=8)
            if col_idx == 0:
                ax.set_ylabel('Classe Real')
            if row_idx == 1:
                ax.set_xlabel('Classe Prevista')
                
    plt.suptitle('Matrizes de Confusão Consolidadas (30.000 Predições Out-Of-Fold)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '17_matrizes_confusao_painel.png'))
    plt.close(fig)


def gerar_grafico_comparacao_classes_0_1(df_folds):
    """Compara Precisão, Recall e F1 entre a Classe 0 (Adimplente) e a Classe 1 (Inadimplente)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)
    
    # Considera a Variação A para análise de classes (padrão de referência)
    sub = df_folds[df_folds['variacao'] == 'variacao_a']
    algos = ['decision_tree', 'random_forest', 'knn']
    nomes = ['Decision Tree', 'Random Forest', 'KNN']
    x = np.arange(len(algos))
    largura = 0.35
    
    metricas = [
        ('precisao', 'Precisão por Classe', axes[0]),
        ('recall', 'Recall (Sensibilidade) por Classe', axes[1]),
        ('f1', 'F1-Score por Classe', axes[2])
    ]
    
    for m_nome, m_titulo, ax in metricas:
        c0_vals = [sub[sub['algoritmo'] == a][f'classe_0_{m_nome}'].mean() for a in algos]
        c1_vals = [sub[sub['algoritmo'] == a][f'classe_1_{m_nome}'].mean() for a in algos]
        
        ax.bar(x - largura/2, c0_vals, largura, label='Classe 0 (Adimplente - 77.9%)', color='#2b5c8f', edgecolor='#222')
        ax.bar(x + largura/2, c1_vals, largura, label='Classe 1 (Inadimplente - 22.1%)', color='#d95f02', edgecolor='#222')
        
        ax.set_title(m_titulo, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(nomes)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Escore')
        
        for i in range(len(algos)):
            ax.annotate(f'{c0_vals[i]:.2f}', (x[i] - largura/2, c0_vals[i]), xytext=(0, 3), 
                        textcoords="offset points", ha='center', fontsize=8)
            ax.annotate(f'{c1_vals[i]:.2f}', (x[i] + largura/2, c1_vals[i]), xytext=(0, 3), 
                        textcoords="offset points", ha='center', fontsize=8, fontweight='bold')
            
        if ax == axes[0]:
            ax.legend(loc='lower left', fontsize=8)
            
    plt.suptitle('Comparação de Desempenho entre as Classes: Impacto do Desbalanceamento (Variação A)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '18_desempenho_classes_0_vs_1.png'))
    plt.close(fig)


def gerar_grafico_impacto_consolidacao():
    """Gera gráfico demonstrando a variação percentual delta (Variação B - Variação A)."""
    df_comp = pd.read_csv(os.path.join(METRICS_DIR, 'comparacao_variacoes.csv'))
    
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    
    # Exclui Dummy (que possui delta zero por definição)
    sub = df_comp[df_comp['algoritmo'] != 'dummy']
    algos = sub['algoritmo'].tolist()
    nomes = [NOMES_ALGOS[a] for a in algos]
    
    x = np.arange(len(algos))
    largura = 0.22
    
    r1 = ax.bar(x - largura, sub['delta_acuracia'] * 100, largura, label='Delta Acurácia (%)', color='#4393c3')
    r2 = ax.bar(x, sub['delta_f1_macro'] * 100, largura, label='Delta F1-Macro (%)', color='#74add1')
    r3 = ax.bar(x + largura, sub['delta_recall_c1'] * 100, largura, label='Delta Recall Inadimplente (%)', color='#d73027')
    
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Impacto da Consolidação de Categorias (Diferença: Variação B - Variação A)', pad=12, fontweight='bold')
    ax.set_ylabel('Variação Absoluta de Pontos Percentuais (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(nomes)
    ax.set_ylim(-1.2, 0.4)
    ax.legend(loc='lower left')
    
    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            pos_y = h - 0.08 if h < 0 else h + 0.02
            ax.annotate(f'{h:+.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, pos_y),
                        ha='center', va='top' if h < 0 else 'bottom', fontsize=8)
            
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '19_impacto_consolidacao_variacao_a_vs_b.png'))
    plt.close(fig)


def gerar_grafico_curvas_roc_pr(df_oof):
    """Gera curvas ROC e Precision-Recall consolidadas out-of-fold."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    algos = ['decision_tree', 'random_forest', 'knn', 'dummy']
    cores = {'dummy': '#888888', 'decision_tree': '#1f77b4', 'random_forest': '#2ca02c', 'knn': '#ff7f0e'}
    
    # Avaliando sob a Variação A (mesmo comportamento para B)
    sub_a = df_oof[df_oof['variacao'] == 'variacao_a']
    y_true = sub_a[sub_a['algoritmo'] == 'decision_tree']['y_real'].values
    
    # 1. Curva ROC
    for algo in algos:
        y_prob = sub_a[sub_a['algoritmo'] == algo]['y_prob_classe_1'].values
        if algo == 'dummy':
            auc_val = 0.500
            ax_roc.plot([0, 1], [0, 1], linestyle='--', color='#888', label=f'Dummy (AUC = {auc_val:.3f})')
        else:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            ax_roc.plot(fpr, tpr, color=cores[algo], linewidth=1.8, label=f"{NOMES_ALGOS[algo]} (AUC = {auc_val:.3f})")
            
    ax_roc.set_title('Curvas ROC Consolidadas (Out-Of-Fold)', fontweight='bold')
    ax_roc.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    ax_roc.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade/Recall)')
    ax_roc.legend(loc='lower right', fontsize=9)
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1.02])
    
    # 2. Curva Precision-Recall
    baseline_pr = float(np.mean(y_true))
    ax_pr.axhline(baseline_pr, linestyle='--', color='#888', label=f'Prevalência Real ({baseline_pr:.3f})')
    
    for algo in ['decision_tree', 'random_forest', 'knn']:
        y_prob = sub_a[sub_a['algoritmo'] == algo]['y_prob_classe_1'].values
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap_val = average_precision_score(y_true, y_prob)
        ax_pr.plot(rec, prec, color=cores[algo], linewidth=1.8, label=f"{NOMES_ALGOS[algo]} (AP = {ap_val:.3f})")
        
    ax_pr.set_title('Curvas Precision-Recall (Detecção de Default)', fontweight='bold')
    ax_pr.set_xlabel('Recall (Sensibilidade da Classe 1)')
    ax_pr.set_ylabel('Precisão da Classe 1')
    ax_pr.legend(loc='upper right', fontsize=9)
    ax_pr.set_xlim([0, 1])
    ax_pr.set_ylim([0, 1.02])
    
    plt.suptitle('Desempenho Probabilístico Out-Of-Fold: Curvas ROC e Precision-Recall (Variação A)', 
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '20_curvas_roc_precision_recall.png'))
    plt.close(fig)


def executar_geracao_graficos():
    """Função orquestradora da etapa de geração de gráficos."""
    logger.info("=== INÍCIO DA GERAÇÃO DE GRÁFICOS ACADÊMICOS (FASE 3) ===")
    
    df_folds = pd.read_csv(os.path.join(METRICS_DIR, 'metricas_por_fold.csv'))
    df_oof = pd.read_csv(os.path.join(METRICS_DIR, 'predicoes_out_of_fold.csv'))
    
    gerar_graficos_individuais(df_folds)
    gerar_grafico_matrizes_confusao(df_oof)
    gerar_grafico_comparacao_classes_0_1(df_folds)
    gerar_grafico_impacto_consolidacao()
    gerar_grafico_curvas_roc_pr(df_oof)
    
    logger.info("=== TODOS OS 12 GRÁFICOS FORAM GERADOS COM SUCESSO EM results/charts/ ===")


if __name__ == '__main__':
    executar_geracao_graficos()
