"""
Script de Análise Exploratória de Dados (EDA)
Projeto: Trabalho Acadêmico - Inteligência Artificial | Universidade Feevale
Tema: Aprendizado de Máquina - Classificação Supervisionada
Dataset: Default of Credit Card Clients (UCI Machine Learning Repository #350)
Autores: Gabriel Machado e João Tonin
Professor: Rodrigo
Data: Novembro/2026
"""

import os
import sys

# Configurações de ambiente para execução headless e cache de fontes do Matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.environ['MPLBACKEND'] = 'Agg'

import math
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração de diretórios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')
CHARTS_DIR = os.path.join(BASE_DIR, 'results', 'charts')
LOGS_DIR = os.path.join(BASE_DIR, 'results', 'logs')

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Configuração de Logging
log_file = os.path.join(LOGS_DIR, 'execucao_exploratoria.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configurações de estilo para gráficos acadêmicos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Dicionário semântico e descritivo das variáveis
METADATA_ATRIBUTOS = {
    'ID': {
        'papel': 'Identificador',
        'tipo_semantico': 'Identificador Único',
        'tipo_armazenamento': 'int64',
        'unidade': 'Adimensional',
        'descricao': 'Identificador único sequencial de cada cliente (1 a 30.000). Não deve ser usado como preditor.'
    },
    'LIMIT_BAL': {
        'papel': 'Preditor',
        'tipo_semantico': 'Numérico Contínuo',
        'tipo_armazenamento': 'int64',
        'unidade': 'Novo Dólar Taiwanês (NT$)',
        'descricao': 'Montante do crédito concedido, incluindo crédito individual e familiar suplementar.'
    },
    'SEX': {
        'papel': 'Preditor',
        'tipo_semantico': 'Categórico Nominal Binário',
        'tipo_armazenamento': 'int64',
        'unidade': 'Categorias (1=Masculino, 2=Feminino)',
        'descricao': 'Gênero do titular do cartão de crédito.'
    },
    'EDUCATION': {
        'papel': 'Preditor',
        'tipo_semantico': 'Categórico Ordinal / Nominal',
        'tipo_armazenamento': 'int64',
        'unidade': 'Categorias (1=Pós-graduação, 2=Universidade, 3=Ensino Médio, 4=Outros)',
        'descricao': 'Grau de escolaridade formal. Valores não documentados: 0, 5 e 6.'
    },
    'MARRIAGE': {
        'papel': 'Preditor',
        'tipo_semantico': 'Categórico Nominal',
        'tipo_armazenamento': 'int64',
        'unidade': 'Categorias (1=Casado, 2=Solteiro, 3=Outros)',
        'descricao': 'Estado civil do cliente. Valor não documentado: 0.'
    },
    'AGE': {
        'papel': 'Preditor',
        'tipo_semantico': 'Numérico Discreto',
        'tipo_armazenamento': 'int64',
        'unidade': 'Anos',
        'descricao': 'Idade cronológica em anos completos.'
    },
    'PAY_0': {
        'papel': 'Preditor',
        'tipo_semantico': 'Categórico Ordinal',
        'tipo_armazenamento': 'int64',
        'unidade': 'Escala de atraso em meses',
        'descricao': 'Status de amortização em setembro/2005 (-1=pontual; 1 a 8=meses de atraso). Códigos -2 e 0 não documentados.'
    },
    'PAY_2': {
        'papel': 'Preditor',
        'tipo_semantico': 'Categórico Ordinal',
        'tipo_armazenamento': 'int64',
        'unidade': 'Escala de atraso em meses',
        'descricao': 'Status de amortização em agosto/2005.'
    },
    'PAY_3': {
        'papel': 'Preditor',
        'tipo_semantico': 'Categórico Ordinal',
        'tipo_armazenamento': 'int64',
        'unidade': 'Escala de atraso em meses',
        'descricao': 'Status de amortização em julho/2005.'
    },
    'PAY_4': {
        'papel': 'Preditor',
        'tipo_semantico': 'Categórico Ordinal',
        'tipo_armazenamento': 'int64',
        'unidade': 'Escala de atraso em meses',
        'descricao': 'Status de amortização em junho/2005.'
    },
    'PAY_5': {
        'papel': 'Preditor',
        'tipo_semantico': 'Categórico Ordinal',
        'tipo_armazenamento': 'int64',
        'unidade': 'Escala de atraso em meses',
        'descricao': 'Status de amortização em maio/2005.'
    },
    'PAY_6': {
        'papel': 'Preditor',
        'tipo_semantico': 'Categórico Ordinal',
        'tipo_armazenamento': 'int64',
        'unidade': 'Escala de atraso em meses',
        'descricao': 'Status de amortização em abril/2005.'
    },
    'BILL_AMT1': {
        'papel': 'Preditor',
        'tipo_semantico': 'Numérico Contínuo',
        'tipo_armazenamento': 'int64',
        'unidade': 'Novo Dólar Taiwanês (NT$)',
        'descricao': 'Valor da fatura em setembro de 2005.'
    },
    'BILL_AMT2': {
        'papel': 'Preditor',
        'tipo_semantico': 'Numérico Contínuo',
        'tipo_armazenamento': 'int64',
        'unidade': 'Novo Dólar Taiwanês (NT$)',
        'descricao': 'Valor da fatura em agosto de 2005.'
    },
    'BILL_AMT3': {
        'papel': 'Preditor',
        'tipo_semantico': 'Numérico Contínuo',
        'tipo_armazenamento': 'int64',
        'unidade': 'Novo Dólar Taiwanês (NT$)',
        'descricao': 'Valor da fatura em julho de 2005.'
    },
    'BILL_AMT4': {
        'papel': 'Preditor',
        'tipo_semantico': 'Numérico Contínuo',
        'tipo_armazenamento': 'int64',
        'unidade': 'Novo Dólar Taiwanês (NT$)',
        'descricao': 'Valor da fatura em junho de 2005.'
    },
    'BILL_AMT5': {
        'papel': 'Preditor',
        'tipo_semantico': 'Numérico Contínuo',
        'tipo_armazenamento': 'int64',
        'unidade': 'Novo Dólar Taiwanês (NT$)',
        'descricao': 'Valor da fatura em maio de 2005.'
    },
    'BILL_AMT6': {
        'papel': 'Preditor',
        'tipo_semantico': 'Numérico Contínuo',
        'tipo_armazenamento': 'int64',
        'unidade': 'Novo Dólar Taiwanês (NT$)',
        'descricao': 'Valor da fatura em abril de 2005.'
    },
    'PAY_AMT1': {
        'papel': 'Preditor',
        'tipo_semantico': 'Numérico Contínuo',
        'tipo_armazenamento': 'int64',
        'unidade': 'Novo Dólar Taiwanês (NT$)',
        'descricao': 'Valor pago em setembro de 2005 referente à fatura anterior.'
    },
    'PAY_AMT2': {
        'papel': 'Preditor',
        'tipo_semantico': 'Numérico Contínuo',
        'tipo_armazenamento': 'int64',
        'unidade': 'Novo Dólar Taiwanês (NT$)',
        'descricao': 'Valor pago em agosto de 2005 referente à fatura anterior.'
    },
    'PAY_AMT3': {
        'papel': 'Preditor',
        'tipo_semantico': 'Numérico Contínuo',
        'tipo_armazenamento': 'int64',
        'unidade': 'Novo Dólar Taiwanês (NT$)',
        'descricao': 'Valor pago em julho de 2005 referente à fatura anterior.'
    },
    'PAY_AMT4': {
        'papel': 'Preditor',
        'tipo_semantico': 'Numérico Contínuo',
        'tipo_armazenamento': 'int64',
        'unidade': 'Novo Dólar Taiwanês (NT$)',
        'descricao': 'Valor pago em junho de 2005 referente à fatura anterior.'
    },
    'PAY_AMT5': {
        'papel': 'Preditor',
        'tipo_semantico': 'Numérico Contínuo',
        'tipo_armazenamento': 'int64',
        'unidade': 'Novo Dólar Taiwanês (NT$)',
        'descricao': 'Valor pago em maio de 2005 referente à fatura anterior.'
    },
    'PAY_AMT6': {
        'papel': 'Preditor',
        'tipo_semantico': 'Numérico Contínuo',
        'tipo_armazenamento': 'int64',
        'unidade': 'Novo Dólar Taiwanês (NT$)',
        'descricao': 'Valor pago em abril de 2005 referente à fatura anterior.'
    },
    'default payment next month': {
        'papel': 'Alvo (Target)',
        'tipo_semantico': 'Categórico Binário',
        'tipo_armazenamento': 'int64',
        'unidade': '0=Não Inadimplente, 1=Inadimplente',
        'descricao': 'Indicador se o cliente entrou em inadimplência no mês subsequente (outubro/2005).'
    }
}


def carregar_dados():
    """Carrega o dataset a partir do arquivo Excel oficial preservado em data/raw."""
    excel_path = os.path.join(DATA_RAW_DIR, 'default of credit card clients.xls')
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Arquivo original não encontrado em {excel_path}")
    
    logger.info(f"Carregando dataset de {excel_path}...")
    # Linha 0 contém X1..X23, Y; linha 1 contém os nomes descritivos
    df = pd.read_excel(excel_path, header=1)
    logger.info(f"Dataset carregado com sucesso: {df.shape[0]} registros e {df.shape[1]} colunas.")
    return df


def caracterizacao_geral(df):
    """Executa a caracterização completa de colunas, tipos e valores ausentes."""
    logger.info("Iniciando caracterização de atributos...")
    registros = []
    
    for col in df.columns:
        meta = METADATA_ATRIBUTOS.get(col, {
            'papel': 'Desconhecido',
            'tipo_semantico': 'Indefinido',
            'tipo_armazenamento': str(df[col].dtype),
            'unidade': 'N/A',
            'descricao': 'Sem descrição'
        })
        n_unicos = df[col].nunique()
        n_nulos = int(df[col].isnull().sum())
        pct_nulos = float(n_nulos / len(df) * 100.0)
        v_min = float(df[col].min())
        v_max = float(df[col].max())
        
        registros.append({
            'atributo': col,
            'papel': meta['papel'],
            'tipo_armazenamento': str(df[col].dtype),
            'tipo_semantico': meta['tipo_semantico'],
            'unidade': meta['unidade'],
            'valores_distintos': n_unicos,
            'valores_ausentes': n_nulos,
            'pct_ausentes': pct_nulos,
            'valor_minimo': v_min,
            'valor_maximo': v_max,
            'descricao': meta['descricao']
        })
        
    df_caracterizacao = pd.DataFrame(registros)
    csv_path = os.path.join(METRICS_DIR, 'caracterizacao_atributos.csv')
    df_caracterizacao.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Tabela de caracterização salva em {csv_path}")
    return df_caracterizacao


def analise_numerica(df):
    """Gera estatísticas descritivas completas para variáveis numéricas contínuas e discretas."""
    logger.info("Calculando estatísticas descritivas dos atributos numéricos...")
    cols_numericas = ['LIMIT_BAL', 'AGE'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]
    
    registros = []
    for col in cols_numericas:
        s = df[col]
        registros.append({
            'atributo': col,
            'tipo_semantico': METADATA_ATRIBUTOS[col]['tipo_semantico'],
            'media': round(float(s.mean()), 2),
            'desvio_padrao': round(float(s.std()), 2),
            'minimo': round(float(s.min()), 2),
            'q25': round(float(s.quantile(0.25)), 2),
            'mediana': round(float(s.median()), 2),
            'q75': round(float(s.quantile(0.75)), 2),
            'maximo': round(float(s.max()), 2),
            'intervalo_iqr': round(float(s.quantile(0.75) - s.quantile(0.25)), 2),
            'assimetria_skewness': round(float(s.skew()), 4),
            'curtose': round(float(s.kurt()), 4)
        })
        
    df_num = pd.DataFrame(registros)
    csv_path = os.path.join(METRICS_DIR, 'estatisticas_numericas.csv')
    df_num.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Estatísticas numéricas salvas em {csv_path}")
    return df_num


def analise_categoricas_e_desbalanceamento(df):
    """
    Analisa a distribuição de frequências e quantifica o desbalanceamento das variáveis categóricas
    utilizando critérios matemáticos explícitos:
    1. Proporção Modal (Modal Ratio): freq_max / N
    2. Razão de Desbalanceamento (Imbalance Ratio - IR): freq_max / freq_min
    3. Entropia de Shannon Normalizada (H_norm): -sum(p * ln(p)) / ln(K)
       onde H_norm = 1 indica distribuição perfeitamente uniforme e H_norm próximo de 0 indica forte concentração.
    """
    logger.info("Calculando distribuição e desbalanceamento de atributos categóricos...")
    cols_categoricas = ['SEX', 'EDUCATION', 'MARRIAGE', 'default payment next month'] + [f'PAY_{i}' for i in [0, 2, 3, 4, 5, 6]]
    
    # 1. Tabela detalhada de frequências
    registros_dist = []
    for col in cols_categoricas:
        vc = df[col].value_counts().sort_index()
        total = len(df)
        for val, count in vc.items():
            registros_dist.append({
                'atributo': col,
                'categoria_codigo': val,
                'frequencia_absoluta': count,
                'frequencia_relativa_pct': round(float(count / total * 100.0), 4)
            })
    df_dist = pd.DataFrame(registros_dist)
    df_dist.to_csv(os.path.join(METRICS_DIR, 'distribuicao_categoricas.csv'), index=False, encoding='utf-8')
    
    # 2. Métricas explícitas de desbalanceamento por atributo
    registros_desbal = []
    N = len(df)
    for col in cols_categoricas:
        vc = df[col].value_counts()
        K = len(vc)
        f_max = vc.max()
        cat_max = vc.idxmax()
        f_min = vc.min()
        cat_min = vc.idxmin()
        
        imbalance_ratio = float(f_max / f_min) if f_min > 0 else np.nan
        modal_ratio = float(f_max / N)
        
        # Entropia normalizada
        p = vc.values / N
        shannon_entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(K) if K > 1 else 1.0
        norm_entropy = float(shannon_entropy / max_entropy) if max_entropy > 0 else 1.0
        
        registros_desbal.append({
            'atributo': col,
            'papel': METADATA_ATRIBUTOS[col]['papel'],
            'quantidade_classes': K,
            'classe_majoritaria': cat_max,
            'freq_majoritaria': f_max,
            'pct_majoritaria': round(modal_ratio * 100, 2),
            'classe_minoritaria': cat_min,
            'freq_minoritaria': f_min,
            'pct_minoritaria': round((f_min / N) * 100, 4),
            'razao_desbalanceamento_IR': round(imbalance_ratio, 2),
            'entropia_normalizada': round(norm_entropy, 4)
        })
        
    df_desbal = pd.DataFrame(registros_desbal).sort_values(by='razao_desbalanceamento_IR', ascending=False)
    csv_path = os.path.join(METRICS_DIR, 'desbalanceamento_categoricas.csv')
    df_desbal.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Métricas de desbalanceamento salvas em {csv_path}")
    return df_dist, df_desbal


def analise_target(df):
    """Gera resumo específico do atributo target."""
    col_target = 'default payment next month'
    vc = df[col_target].value_counts()
    n_total = len(df)
    
    registros = [
        {
            'classe': 0,
            'descricao': 'Não Inadimplente (Adimplente)',
            'contagem': int(vc[0]),
            'percentual': round(float(vc[0] / n_total * 100.0), 2)
        },
        {
            'classe': 1,
            'descricao': 'Inadimplente (Default)',
            'contagem': int(vc[1]),
            'percentual': round(float(vc[1] / n_total * 100.0), 2)
        }
    ]
    df_target = pd.DataFrame(registros)
    csv_path = os.path.join(METRICS_DIR, 'distribuicao_target.csv')
    df_target.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Distribuição do target salva em {csv_path}")
    return df_target


def analise_taxa_inadimplencia(df):
    """Calcula taxa de inadimplência por subcategorias demográficas."""
    logger.info("Calculando taxa de inadimplência por categorias demográficas...")
    col_target = 'default payment next month'
    registros = []
    
    # 1. Gênero
    nomes_sex = {1: 'Masculino', 2: 'Feminino'}
    for val, grp in df.groupby('SEX'):
        tot = len(grp)
        inad = int(grp[col_target].sum())
        registros.append({
            'atributo': 'SEX',
            'codigo': val,
            'rotulo': nomes_sex.get(val, str(val)),
            'total_clientes': tot,
            'inadimplentes': inad,
            'taxa_inadimplencia_pct': round((inad / tot) * 100.0, 2)
        })
        
    # 2. Escolaridade
    nomes_edu = {
        0: '0 - Não Documentado',
        1: '1 - Pós-Graduação',
        2: '2 - Universidade',
        3: '3 - Ensino Médio',
        4: '4 - Outros',
        5: '5 - Não Documentado',
        6: '6 - Não Documentado'
    }
    for val, grp in df.groupby('EDUCATION'):
        tot = len(grp)
        inad = int(grp[col_target].sum())
        registros.append({
            'atributo': 'EDUCATION',
            'codigo': val,
            'rotulo': nomes_edu.get(val, str(val)),
            'total_clientes': tot,
            'inadimplentes': inad,
            'taxa_inadimplencia_pct': round((inad / tot) * 100.0, 2)
        })
        
    # 3. Estado Civil
    nomes_mar = {
        0: '0 - Não Documentado',
        1: '1 - Casado',
        2: '2 - Solteiro',
        3: '3 - Outros'
    }
    for val, grp in df.groupby('MARRIAGE'):
        tot = len(grp)
        inad = int(grp[col_target].sum())
        registros.append({
            'atributo': 'MARRIAGE',
            'codigo': val,
            'rotulo': nomes_mar.get(val, str(val)),
            'total_clientes': tot,
            'inadimplentes': inad,
            'taxa_inadimplencia_pct': round((inad / tot) * 100.0, 2)
        })
        
    df_taxas = pd.DataFrame(registros)
    csv_path = os.path.join(METRICS_DIR, 'taxa_inadimplencia_demograficas.csv')
    df_taxas.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Taxas de inadimplência demográficas salvas em {csv_path}")
    return df_taxas


def catalogar_inconsistencias(df):
    """Identifica e documenta formalmente os valores não documentados e discrepâncias."""
    logger.info("Catalogando inconsistências e valores não previstos...")
    registros = []
    
    # EDUCATION: 0, 5, 6
    for val in [0, 5, 6]:
        cnt = int((df['EDUCATION'] == val).sum())
        registros.append({
            'atributo': 'EDUCATION',
            'valor_encontrado': val,
            'contagem': cnt,
            'percentual': round((cnt / len(df)) * 100.0, 4),
            'status_documentacao': 'Não documentado na UCI (valores oficiais: 1 a 4)',
            'impacto_interpretacao': 'Representa códigos desconhecidos ou categorias adicionais de outros'
        })
        
    # MARRIAGE: 0
    cnt_mar0 = int((df['MARRIAGE'] == 0).sum())
    registros.append({
        'atributo': 'MARRIAGE',
        'valor_encontrado': 0,
        'contagem': cnt_mar0,
        'percentual': round((cnt_mar0 / len(df)) * 100.0, 4),
        'status_documentacao': 'Não documentado na UCI (valores oficiais: 1, 2, 3)',
        'impacto_interpretacao': 'Possível estado civil ausente/desconhecido'
    })
    
    # PAY_X: -2 e 0
    for col in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
        for val, desc in [(-2, 'Inatividade / Saldo zero / Sem consumo'), (0, 'Uso de crédito rotativo / Pagamento mínimo')]:
            cnt = int((df[col] == val).sum())
            registros.append({
                'atributo': col,
                'valor_encontrado': val,
                'contagem': cnt,
                'percentual': round((cnt / len(df)) * 100.0, 4),
                'status_documentacao': 'Não descrito na documentação da UCI (documentado apenas -1 e 1..9)',
                'impacto_interpretacao': desc
            })
            
    # BILL_AMT negativos
    for col in [f'BILL_AMT{i}' for i in range(1, 7)]:
        cnt_neg = int((df[col] < 0).sum())
        registros.append({
            'atributo': col,
            'valor_encontrado': '< 0 (saldo credor)',
            'contagem': cnt_neg,
            'percentual': round((cnt_neg / len(df)) * 100.0, 4),
            'status_documentacao': 'Válido operacionalmente, não explicitado na documentação resumida',
            'impacto_interpretacao': 'Pagamento antecipado ou crédito a favor do cliente gerando saldo negativo na fatura'
        })
        
    df_inconsistencias = pd.DataFrame(registros)
    csv_path = os.path.join(METRICS_DIR, 'valores_inconsistentes.csv')
    df_inconsistencias.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Tabela de inconsistências salva em {csv_path}")
    return df_inconsistencias


def gerar_graficos(df):
    """Gera gráficos com formatação visual rigorosa, títulos em PT-BR e resolução adequada."""
    logger.info("Gerando gráficos exploratórios...")
    target_col = 'default payment next month'
    cor_adimplente = '#2b5c8f'
    cor_inadimplente = '#d95f02'
    
    df_plot = df.copy()
    df_plot['rotulo_target'] = df_plot[target_col].map({0: 'Não Inadimplente', 1: 'Inadimplente'})
    cores_dict = {'Não Inadimplente': cor_adimplente, 'Inadimplente': cor_inadimplente}
    
    # -------------------------------------------------------------
    # Gráfico 1: Distribuição das classes do target
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    contagens = df[target_col].value_counts().sort_index()
    rotulos = ['Não Inadimplente (Classe 0)', 'Inadimplente (Classe 1)']
    cores = [cor_adimplente, cor_inadimplente]
    barras = ax.bar(rotulos, contagens, color=cores, width=0.55, edgecolor='#222222', linewidth=0.8)
    
    for barra in barras:
        altura = barra.get_height()
        pct = (altura / len(df)) * 100
        ax.annotate(f'{altura:,}\n({pct:.2f}%)'.replace(',', '.'),
                    xy=(barra.get_x() + barra.get_width() / 2, altura),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    ax.set_title('Distribuição da Variável Alvo: Inadimplência no Próximo Mês', pad=15)
    ax.set_ylabel('Quantidade de Clientes')
    ax.set_ylim(0, 27000)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '01_distribuicao_target.png'))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Gráfico 2: Distribuição de Idades
    # -------------------------------------------------------------
    fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, 
                                         gridspec_kw={'height_ratios': [0.7, 0.3]}, dpi=300)
    
    sns.histplot(data=df_plot, x='AGE', hue='rotulo_target', bins=35, kde=True, 
                 palette=cores_dict, ax=ax_hist,
                 legend=True, alpha=0.6, edgecolor='white')
    ax_hist.set_title('Distribuição de Idade por Status de Adimplência')
    ax_hist.set_ylabel('Frequência de Clientes')
    
    sns.boxplot(data=df_plot, x='AGE', y='rotulo_target', 
                palette=cores_dict, ax=ax_box, hue='rotulo_target', legend=False)
    ax_box.set_xlabel('Idade (anos)')
    ax_box.set_ylabel('Classe')
    
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '02_distribuicao_idade.png'))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Gráfico 3: Distribuição dos Limites de Crédito (LIMIT_BAL)
    # -------------------------------------------------------------
    fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, 
                                         gridspec_kw={'height_ratios': [0.7, 0.3]}, dpi=300)
    
    sns.histplot(data=df_plot, x='LIMIT_BAL', hue='rotulo_target', bins=30, kde=False, 
                 palette=cores_dict, ax=ax_hist,
                 legend=True, alpha=0.6, edgecolor='white')
    ax_hist.set_title('Distribuição do Limite de Crédito Concedido (NT$)')
    ax_hist.set_ylabel('Frequência de Clientes')
    
    sns.boxplot(data=df_plot, x='LIMIT_BAL', y='rotulo_target', 
                palette=cores_dict, ax=ax_box, hue='rotulo_target', legend=False)
    ax_box.set_xlabel('Limite de Crédito Concedido (Novo Dólar Taiwanês - NT$)')
    ax_box.set_ylabel('Classe')
    
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '03_distribuicao_limite_credito.png'))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Gráfico 4: Distribuição das Principais Variáveis Categóricas
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=300)
    
    # Gênero
    s_sex = df['SEX'].value_counts().sort_index()
    axes[0].bar(['Masculino (1)', 'Feminino (2)'], s_sex, color=['#4393c3', '#f4a582'], edgecolor='#333333')
    axes[0].set_title('Gênero (SEX)')
    axes[0].set_ylabel('Contagem')
    for i, v in enumerate(s_sex):
        axes[0].text(i, v + 250, f'{v:,}\n({v/len(df)*100:.1f}%)'.replace(',', '.'), ha='center', fontsize=9)
    axes[0].set_ylim(0, 22000)
    
    # Escolaridade
    s_edu = df['EDUCATION'].value_counts().sort_index()
    labels_edu = [f'{k}' for k in s_edu.index]
    axes[1].bar(labels_edu, s_edu, color='#7fbf7b', edgecolor='#333333')
    axes[1].set_title('Escolaridade (EDUCATION)\n1=Pós, 2=Univ, 3=Médio, 4=Outros, 0,5,6=N/Doc')
    axes[1].set_xlabel('Código de Escolaridade')
    for i, v in enumerate(s_edu):
        axes[1].text(i, v + 250, f'{v:,}'.replace(',', '.'), ha='center', fontsize=8)
    axes[1].set_ylim(0, 16500)
    
    # Estado Civil
    s_mar = df['MARRIAGE'].value_counts().sort_index()
    labels_mar = [f'{k}' for k in s_mar.index]
    axes[2].bar(labels_mar, s_mar, color='#af8dc3', edgecolor='#333333')
    axes[2].set_title('Estado Civil (MARRIAGE)\n1=Casado, 2=Solteiro, 3=Outros, 0=N/Doc')
    axes[2].set_xlabel('Código de Estado Civil')
    for i, v in enumerate(s_mar):
        axes[2].text(i, v + 250, f'{v:,}'.replace(',', '.'), ha='center', fontsize=8)
    axes[2].set_ylim(0, 18500)
    
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '04_distribuicao_demograficas.png'))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Gráfico 5: Taxa de Inadimplência por Categoria Demográfica
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    
    # Inadimplência por Gênero
    taxa_sex = df.groupby('SEX')[target_col].mean() * 100
    axes[0].bar(['Masculino (1)', 'Feminino (2)'], taxa_sex, color=['#4393c3', '#f4a582'], edgecolor='#333333')
    axes[0].set_title('Taxa de Inadimplência por Gênero')
    axes[0].set_ylabel('Taxa de Inadimplência (%)')
    axes[0].set_ylim(0, 35)
    for i, v in enumerate(taxa_sex):
        axes[0].text(i, v + 0.8, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=10)
    axes[0].axhline(y=22.12, color='red', linestyle='--', linewidth=1, label='Média Geral (22.12%)')
    axes[0].legend(loc='upper right')
    
    # Inadimplência por Escolaridade
    taxa_edu = df.groupby('EDUCATION')[target_col].mean() * 100
    axes[1].bar([str(k) for k in taxa_edu.index], taxa_edu, color='#7fbf7b', edgecolor='#333333')
    axes[1].set_title('Taxa de Inadimplência por Escolaridade')
    axes[1].set_xlabel('Código de Escolaridade')
    axes[1].set_ylim(0, 35)
    for i, v in enumerate(taxa_edu):
        axes[1].text(i, v + 0.8, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
    axes[1].axhline(y=22.12, color='red', linestyle='--', linewidth=1)
    
    # Inadimplência por Estado Civil
    taxa_mar = df.groupby('MARRIAGE')[target_col].mean() * 100
    axes[2].bar([str(k) for k in taxa_mar.index], taxa_mar, color='#af8dc3', edgecolor='#333333')
    axes[2].set_title('Taxa de Inadimplência por Estado Civil')
    axes[2].set_xlabel('Código de Estado Civil')
    axes[2].set_ylim(0, 35)
    for i, v in enumerate(taxa_mar):
        axes[2].text(i, v + 0.8, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
    axes[2].axhline(y=22.12, color='red', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '05_taxas_inadimplencia_demograficas.png'))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Gráfico 6: Histórico de Pagamento Recente (PAY_0) vs Inadimplência
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    grp_pay0 = df.groupby('PAY_0')[target_col].agg(['count', 'mean'])
    grp_pay0['taxa_pct'] = grp_pay0['mean'] * 100
    
    ax2 = ax1.twinx()
    # Barras de volume
    ax1.bar(range(len(grp_pay0)), grp_pay0['count'], color='#b0c4de', alpha=0.7, 
            edgecolor='#333333', width=0.6, label='Volume de Clientes (Eixo Esquerdo)')
    # Linha de taxa
    ax2.plot(range(len(grp_pay0)), grp_pay0['taxa_pct'], color='#b2182b', marker='o', 
             linewidth=2.5, markersize=7, label='Taxa de Inadimplência % (Eixo Direito)')
    
    ax1.set_xticks(range(len(grp_pay0)))
    ax1.set_xticklabels([str(k) for k in grp_pay0.index])
    ax1.set_xlabel('Status de Pagamento em Setembro/2005 (PAY_0)')
    ax1.set_ylabel('Quantidade de Clientes')
    ax2.set_ylabel('Taxa de Inadimplência (%)')
    ax2.set_ylim(0, 100)
    ax2.grid(False)
    
    for i, val in enumerate(grp_pay0['taxa_pct']):
        ax2.annotate(f'{val:.1f}%', (i, val), xytext=(0, 7), textcoords="offset points", 
                     ha='center', fontsize=9, fontweight='bold', color='#7f0000')
        
    ax1.set_title('Relação entre Histórico de Pagamento Recente (PAY_0) e Inadimplência', pad=15)
    
    linhas1, rotulos1 = ax1.get_legend_handles_labels()
    linhas2, rotulos2 = ax2.get_legend_handles_labels()
    ax1.legend(linhas1 + linhas2, rotulos1 + rotulos2, loc='upper left')
    
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '06_historico_pagamento_vs_inadimplencia.png'))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Gráfico 7: Matriz de Correlação das Variáveis Numéricas
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
    cols_corr = ['LIMIT_BAL', 'AGE'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)] + [target_col]
    corr_matrix = df[cols_corr].corr()
    
    # Nomes mais amigáveis para matriz
    nomes_corr = ['LIMIT_BAL', 'AGE', 'FAT_SET', 'FAT_AGO', 'FAT_JUL', 'FAT_JUN', 'FAT_MAI', 'FAT_ABR',
                  'PAG_SET', 'PAG_AGO', 'PAG_JUL', 'PAG_JUN', 'PAG_MAI', 'PAG_ABR', 'TARGET']
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-0.3, vmax=1.0, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, 
                annot=True, fmt='.2f', annot_kws={'size': 7}, ax=ax,
                xticklabels=nomes_corr, yticklabels=nomes_corr)
    ax.set_title('Matriz de Correlação Linear de Pearson (Atributos Numéricos e Alvo)', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '07_matriz_correlacao_numericas.png'))
    plt.close(fig)
    
    # -------------------------------------------------------------
    # Gráfico 8: Evolução Mensal Média de Faturas vs Pagamentos
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    meses = ['Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro']
    # Ordem temporal: BILL_AMT6 (Abril) até BILL_AMT1 (Setembro)
    medias_fatura = [df[f'BILL_AMT{i}'].mean() for i in range(6, 0, -1)]
    medias_pagamento = [df[f'PAY_AMT{i}'].mean() for i in range(6, 0, -1)]
    
    ax.plot(meses, medias_fatura, marker='o', color='#2b5c8f', linewidth=2, label='Média de Fatura (BILL_AMT)')
    ax.plot(meses, medias_pagamento, marker='s', color='#d95f02', linewidth=2, label='Média de Pagamento (PAY_AMT)')
    
    for i, val in enumerate(medias_fatura):
        ax.annotate(f'NT$ {val:,.0f}'.replace(',', '.'), (meses[i], val), xytext=(0, 7), 
                    textcoords="offset points", ha='center', fontsize=9, color='#2b5c8f')
    for i, val in enumerate(medias_pagamento):
        ax.annotate(f'NT$ {val:,.0f}'.replace(',', '.'), (meses[i], val), xytext=(0, -15), 
                    textcoords="offset points", ha='center', fontsize=9, color='#d95f02')
        
    ax.set_title('Evolução Temporal Média: Valor das Faturas vs Pagamentos Realizados (2005)', pad=15)
    ax.set_ylabel('Montante Médio (NT$)')
    ax.set_ylim(-2000, 60000)
    ax.legend(loc='center left')
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '08_evolucao_faturas_pagamentos.png'))
    plt.close(fig)
    
    logger.info("Todos os gráficos foram gerados e salvos com sucesso em results/charts/.")


def executar_pipeline():
    """Executa a sequência completa de análise exploratória reproduzível."""
    inicio = datetime.now()
    logger.info("=== INÍCIO DA ANÁLISE EXPLORATÓRIA ===")
    
    # 1. Carga
    df = carregar_dados()
    
    # 2. Caracterização
    df_carac = caracterizacao_geral(df)
    
    # 3. Estatísticas Numéricas
    df_num = analise_numerica(df)
    
    # 4. Distribuição Categórica e Desbalanceamento
    df_dist, df_desbal = analise_categoricas_e_desbalanceamento(df)
    
    # 5. Target
    df_target = analise_target(df)
    
    # 6. Taxas de Inadimplência
    df_taxas = analise_taxa_inadimplencia(df)
    
    # 7. Inconsistências
    df_incons = catalogar_inconsistencias(df)
    
    # 8. Verificação de Duplicatas
    dup_full = df.duplicated().sum()
    dup_features = df.duplicated(subset=[c for c in df.columns if c != 'ID']).sum()
    logger.info(f"Duplicatas completas (com ID): {dup_full}")
    logger.info(f"Duplicatas parciais (mesmos preditores e target, excluindo ID): {dup_features}")
    
    # 9. Geração de Gráficos
    gerar_graficos(df)
    
    fim = datetime.now()
    duracao = (fim - inicio).total_seconds()
    logger.info(f"=== ANÁLISE EXPLORATÓRIA CONCLUÍDA EM {duracao:.2f} SEGUNDOS ===")


if __name__ == '__main__':
    executar_pipeline()
