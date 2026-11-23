"""
Script de Pré-processamento e Geração de Variações de Dataset
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

# Configuração de caminhos relativos ao diretório raiz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')
LOGS_DIR = os.path.join(BASE_DIR, 'results', 'logs')

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
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


def carregar_dados_brutos():
    """Carrega o dataset oficial original preservado em data/raw/."""
    caminho_xls = os.path.join(DATA_RAW_DIR, 'default of credit card clients.xls')
    if not os.path.exists(caminho_xls):
        raise FileNotFoundError(f"Arquivo bruto não encontrado: {caminho_xls}")
    
    logger.info(f"Carregando dados brutos de: {caminho_xls}")
    df = pd.read_excel(caminho_xls, header=1)
    logger.info(f"Dados brutos carregados: {df.shape[0]} registros e {df.shape[1]} colunas.")
    return df


def identificar_grupos_preditores(df):
    """
    Identifica grupos de registros com preditores (23 variáveis) idênticos.
    Gera um identificador auxiliar 'feature_group_id' para impedir vazamento de dados
    (data leakage) entre partições de treino e teste.
    """
    cols_preditoras = [c for c in df.columns if c not in ['ID', 'default payment next month']]
    # Atribui um identificador único para cada combinação distinta dos 23 preditores
    grupos = df.groupby(cols_preditoras).ngroup()
    logger.info(f"Identificação de grupos de preditores concluída:")
    logger.info(f" - Total de registros: {len(df)}")
    logger.info(f" - Total de perfis preditivos distintos: {grupos.nunique()}")
    logger.info(f" - Quantidade de perfis com mais de um cliente: {(grupos.value_counts() > 1).sum()}")
    return grupos


def construir_variacao_a(df_raw, grupos):
    """
    Variação A - Base de Referência:
    - Preserva os códigos originais de EDUCATION (0 a 6) e MARRIAGE (0 a 3).
    - Preserva todos os demais atributos e valores (incluindo negativos e zeros).
    - Preserva a coluna ID e adiciona feature_group_id para rastreabilidade auxiliar
      (ambos NÃO serão utilizados como preditores no treinamento).
    - NENHUM escalonamento global é aplicado neste arquivo para evitar data leakage.
    """
    logger.info("Construindo Variação A (Base de Referência com códigos originais)...")
    df_a = df_raw.copy()
    df_a['feature_group_id'] = grupos
    
    # Validações estruturais
    assert len(df_a) == 30000, f"Tamanho incorreto na Variação A: {len(df_a)}"
    assert df_a['default payment next month'].value_counts()[0] == 23364, "Divergência na classe 0 da Variação A"
    assert df_a['default payment next month'].value_counts()[1] == 6636, "Divergência na classe 1 da Variação A"
    assert df_a.isnull().sum().sum() == 0, "Valores nulos encontrados na Variação A"
    
    caminho_saida = os.path.join(DATA_PROCESSED_DIR, 'dataset_variacao_a.csv')
    df_a.to_csv(caminho_saida, index=False, encoding='utf-8')
    logger.info(f"Variação A salva com sucesso em: {caminho_saida}")
    return df_a


def construir_variacao_b(df_raw, grupos):
    """
    Variação B - Consolidação de Categorias:
    - Aplica a consolidação metodológica aprovada:
      * EDUCATION: códigos 0, 5 e 6 são mapeados para a categoria 4 (Outros).
      * MARRIAGE: código 0 é mapeado para a categoria 3 (Outros).
    - Preserva todos os demais atributos e valores (incluindo negativos e zeros).
    - Preserva a coluna ID e feature_group_id para rastreabilidade auxiliar.
    - NENHUM escalonamento global é aplicado neste arquivo para evitar data leakage.
    """
    logger.info("Construindo Variação B (Consolidação de Categorias Não Documentadas)...")
    df_b = df_raw.copy()
    
    # Registro de contagens pré-transformação
    contagem_edu_antes = df_b['EDUCATION'].value_counts().sort_index().to_dict()
    contagem_mar_antes = df_b['MARRIAGE'].value_counts().sort_index().to_dict()
    
    # Aplicação das transformações determinísticas aprovadas
    df_b['EDUCATION'] = df_b['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    df_b['MARRIAGE'] = df_b['MARRIAGE'].replace({0: 3})
    
    # Registro de contagens pós-transformação
    contagem_edu_depois = df_b['EDUCATION'].value_counts().sort_index().to_dict()
    contagem_mar_depois = df_b['MARRIAGE'].value_counts().sort_index().to_dict()
    
    logger.info(f"Transformação EDUCATION: Antes={contagem_edu_antes} -> Depois={contagem_edu_depois}")
    logger.info(f"Transformação MARRIAGE: Antes={contagem_mar_antes} -> Depois={contagem_mar_depois}")
    
    df_b['feature_group_id'] = grupos
    
    # Validações estruturais
    assert len(df_b) == 30000, f"Tamanho incorreto na Variação B: {len(df_b)}"
    assert df_b['default payment next month'].value_counts()[0] == 23364, "Divergência na classe 0 da Variação B"
    assert df_b['default payment next month'].value_counts()[1] == 6636, "Divergência na classe 1 da Variação B"
    assert set(df_b['EDUCATION'].unique()) == {1, 2, 3, 4}, f"Categorias inesperadas em EDUCATION: {df_b['EDUCATION'].unique()}"
    assert set(df_b['MARRIAGE'].unique()) == {1, 2, 3}, f"Categorias inesperadas em MARRIAGE: {df_b['MARRIAGE'].unique()}"
    assert df_b.isnull().sum().sum() == 0, "Valores nulos encontrados na Variação B"
    
    caminho_saida = os.path.join(DATA_PROCESSED_DIR, 'dataset_variacao_b.csv')
    df_b.to_csv(caminho_saida, index=False, encoding='utf-8')
    logger.info(f"Variação B salva com sucesso em: {caminho_saida}")
    return df_b, contagem_edu_antes, contagem_edu_depois, contagem_mar_antes, contagem_mar_depois


def gerar_resumo_preprocessamento(df_raw, df_a, df_b, edu_antes, edu_depois, mar_antes, mar_depois):
    """Gera tabela CSV resumindo formalmente as transformações do pré-processamento."""
    logger.info("Gerando resumo comparativo de pré-processamento...")
    registros = [
        {
            'aspecto': 'Total de Instâncias',
            'base_bruta': len(df_raw),
            'variacao_a': len(df_a),
            'variacao_b': len(df_b),
            'observacao': 'Volume total de 30.000 clientes rigorosamente preservado em ambas as variações'
        },
        {
            'aspecto': 'Total de Colunas Exportadas',
            'base_bruta': df_raw.shape[1],
            'variacao_a': df_a.shape[1],
            'variacao_b': df_b.shape[1],
            'observacao': '25 colunas originais + 1 coluna auxiliar feature_group_id para agrupamento de folds'
        },
        {
            'aspecto': 'Atributo ID na Base',
            'base_bruta': 'Presente (1..30000)',
            'variacao_a': 'Preservado para rastreabilidade',
            'variacao_b': 'Preservado para rastreabilidade',
            'observacao': 'Estritamente excluído dos preditores no momento do treinamento'
        },
        {
            'aspecto': 'Categorias de EDUCATION',
            'base_bruta': str(sorted(list(edu_antes.keys()))),
            'variacao_a': str(sorted(list(df_a['EDUCATION'].unique()))),
            'variacao_b': str(sorted(list(df_b['EDUCATION'].unique()))),
            'observacao': f'Variação B consolidou 0 ({edu_antes.get(0, 0)}), 5 ({edu_antes.get(5, 0)}) e 6 ({edu_antes.get(6, 0)}) em 4 (total 4={edu_depois.get(4, 0)})'
        },
        {
            'aspecto': 'Categorias de MARRIAGE',
            'base_bruta': str(sorted(list(mar_antes.keys()))),
            'variacao_a': str(sorted(list(df_a['MARRIAGE'].unique()))),
            'variacao_b': str(sorted(list(df_b['MARRIAGE'].unique()))),
            'observacao': f'Variação B consolidou código 0 ({mar_antes.get(0, 0)}) em 3 (total 3={mar_depois.get(3, 0)})'
        },
        {
            'aspecto': 'Target - Classe 0 (Adimplente)',
            'base_bruta': f"23364 ({23364/30000*100:.2f}%)",
            'variacao_a': f"23364 ({23364/30000*100:.2f}%)",
            'variacao_b': f"23364 ({23364/30000*100:.2f}%)",
            'observacao': 'Distribuição original da classe majoritária 100% inalterada'
        },
        {
            'aspecto': 'Target - Classe 1 (Inadimplente)',
            'base_bruta': f"6636 ({6636/30000*100:.2f}%)",
            'variacao_a': f"6636 ({6636/30000*100:.2f}%)",
            'variacao_b': f"6636 ({6636/30000*100:.2f}%)",
            'observacao': 'Distribuição original da classe minoritária 100% inalterada'
        },
        {
            'aspecto': 'Valores Ausentes (NaN/Nulos)',
            'base_bruta': 0,
            'variacao_a': 0,
            'variacao_b': 0,
            'observacao': 'Nenhum valor nulo introduzido pelo processo'
        },
        {
            'aspecto': 'Escalonamento Global nos Arquivos',
            'base_bruta': 'Não aplicável',
            'variacao_a': 'Não realizado (Prevenção de Leakage)',
            'variacao_b': 'Não realizado (Prevenção de Leakage)',
            'observacao': 'StandardScaler deve ser ajustado estritamente no fold de treino em cada iteração de validação'
        }
    ]
    
    df_resumo = pd.DataFrame(registros)
    caminho_csv = os.path.join(METRICS_DIR, 'resumo_preprocessamento.csv')
    df_resumo.to_csv(caminho_csv, index=False, encoding='utf-8')
    logger.info(f"Tabela de resumo salva em: {caminho_csv}")
    return df_resumo


def executar_preprocessamento():
    """Função orquestradora da etapa de pré-processamento."""
    inicio = datetime.now()
    logger.info("=== INÍCIO DO PRÉ-PROCESSAMENTO (FASE 2) ===")
    
    df_raw = carregar_dados_brutos()
    grupos = identificar_grupos_preditores(df_raw)
    
    df_a = construir_variacao_a(df_raw, grupos)
    df_b, edu_antes, edu_depois, mar_antes, mar_depois = construir_variacao_b(df_raw, grupos)
    
    gerar_resumo_preprocessamento(df_raw, df_a, df_b, edu_antes, edu_depois, mar_antes, mar_depois)
    
    fim = datetime.now()
    duracao = (fim - inicio).total_seconds()
    logger.info(f"=== PRÉ-PROCESSAMENTO CONCLUÍDO EM {duracao:.2f} SEGUNDOS ===")


if __name__ == '__main__':
    executar_preprocessamento()
