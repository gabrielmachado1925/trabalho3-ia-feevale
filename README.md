# Previsão de Inadimplência de Cartão de Crédito com Aprendizado de Máquina

Trabalho acadêmico desenvolvido para a disciplina de **Inteligência Artificial** do curso de **Ciência da Computação** da **Universidade Feevale** (2026.02).

- **Professor:** Rodrigo
- **Integrantes:** Gabriel Machado e João Vitor Lamberti Tonin
- **Tema:** Aprendizado de Máquina – Classificação Supervisionada
- **Dataset:** *Default of Credit Card Clients* (UCI Machine Learning Repository #350)
- **Fonte Oficial:** [https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **DOI:** [10.24432/C55S3H](https://doi.org/10.24432/C55S3H)

---

## 1. Visão Geral do Projeto

O objetivo deste projeto acadêmico é desenvolver, avaliar e comparar rigorosamente modelos de aprendizado de máquina supervisionados para a previsão de inadimplência de clientes de cartão de crédito no mês subsequente, utilizando características demográficas, limites concedidos, histórico de regularidade de pagamentos e evolução financeira de faturas e amortizações.

### Atributo Alvo (Target):
- **Nome:** `default payment next month`
- **Classe 0:** Não Inadimplente (23.364 registros – 77,88%)
- **Classe 1:** Inadimplente (6.636 registros – 22,12%)
- **Desbalanceamento Natural:** Razão $\approx 3,52 : 1$.

### Relatório Acadêmico Oficial:
O relatório completo da atividade, estruturado e diagramado em 19 páginas segundo os padrões da Universidade Feevale, está disponível em [`docs/Relatorio-Atividade-Avaliativa-3-IA.pdf`](docs/Relatorio-Atividade-Avaliativa-3-IA.pdf).

---

## 2. Estrutura do Repositório

```text
.
├── .gitignore                       # Arquivos ignorados pelo Git (.venv, __pycache__, etc.)
├── README.md                        # Apresentação do projeto e instruções completas
├── requirements.txt                 # Dependências e versões exatas do Python
├── data/
│   ├── raw/                         # Arquivos originais intactos e verificados via SHA-256
│   │   ├── default of credit card clients.xls
│   │   ├── default+of+credit+card+clients.zip
│   │   └── data.csv
│   └── processed/                   # Datasets pré-processados da Fase 2
│       ├── dataset_variacao_a.csv   # Variação A: Base de referência (códigos originais preservados)
│       └── dataset_variacao_b.csv   # Variação B: Consolidação de categorias (0,5,6->4 e 0->3)
├── docs/
│   └── Relatorio-Atividade-Avaliativa-3-IA.pdf  # Relatório acadêmico oficial completo (PDF - 19 páginas)
├── models/                          # Modelos persistidos em disco (.joblib - Fase 3)
│   ├── decision_tree_variacao_a.joblib
│   ├── decision_tree_variacao_b.joblib
│   ├── dummy_variacao_a.joblib
│   ├── dummy_variacao_b.joblib
│   ├── knn_variacao_a.joblib
│   ├── knn_variacao_b.joblib
│   ├── random_forest_variacao_a.joblib
│   └── random_forest_variacao_b.joblib
├── results/
│   ├── charts/                      # 23 Gráficos gerados em 300 DPI (inclui painéis para relatório)
│   │   ├── 01_distribuicao_target.png
│   │   ├── 02_distribuicao_idade.png
│   │   ├── 03_distribuicao_limite_credito.png
│   │   ├── 04_distribuicao_demograficas.png
│   │   ├── 05_taxas_inadimplencia_demograficas.png
│   │   ├── 06_historico_pagamento_vs_inadimplencia.png
│   │   ├── 07_matriz_correlacao_numericas.png
│   │   ├── 08_evolucao_faturas_pagamentos.png
│   │   ├── 09_comparacao_acuracia.png
│   │   ├── 10_comparacao_precisao_macro.png
│   │   ├── 11_comparacao_recall_macro.png
│   │   ├── 12_comparacao_f1_macro.png
│   │   ├── 13_comparacao_tempo_treinamento.png
│   │   ├── 14_comparacao_tempo_predicao.png
│   │   ├── 15_comparacao_recall_classe_inadimplente.png
│   │   ├── 16_comparacao_f1_classe_inadimplente.png
│   │   ├── 17_matrizes_confusao_painel.png
│   │   ├── 18_desempenho_classes_0_vs_1.png
│   │   ├── 19_impacto_consolidacao_variacao_a_vs_b.png
│   │   ├── 20_curvas_roc_precision_recall.png
│   │   ├── report_matrizes_confusao.png
│   │   ├── report_curva_roc.png
│   │   └── report_curva_pr.png
│   ├── metrics/                     # 19 Tabelas CSV estruturadas
│   │   ├── caracterizacao_atributos.csv
│   │   ├── comparacao_variacoes.csv
│   │   ├── desbalanceamento_categoricas.csv
│   │   ├── distribuicao_categoricas.csv
│   │   ├── distribuicao_classes_folds.csv
│   │   ├── distribuicao_target.csv
│   │   ├── estatisticas_numericas.csv
│   │   ├── matrizes_confusao.csv
│   │   ├── metricas_experimentos_variacao_a.csv
│   │   ├── metricas_experimentos_variacao_b.csv
│   │   ├── metricas_por_classe.csv
│   │   ├── metricas_por_fold.csv
│   │   ├── modelos_persistidos.csv
│   │   ├── predicoes_out_of_fold.csv
│   │   ├── resumo_preprocessamento.csv
│   │   ├── taxa_inadimplencia_demograficas.csv
│   │   ├── tempos_processamento.csv
│   │   ├── validacao_folds.csv
│   │   └── valores_inconsistentes.csv
│   └── logs/                        # 3 Logs de execução e auditoria
│       ├── execucao_exploratoria.log
│       ├── execucao_preprocessamento.log
│       └── execucao_experimentos.log
└── src/
    ├── explore.py                   # Análise exploratória e estatísticas descritivas (Fase 1)
    ├── preprocess.py                # Geração das Variações A e B de pré-processamento (Fase 2)
    ├── prepare_validation.py        # Partição StratifiedGroupKFold com 5 folds isolados (Fase 2)
    ├── validate_data.py             # Suíte de testes e integridade dos dados (Fase 2)
    ├── train.py                     # Treinamento das 40 rodadas e persistência de modelos (Fase 3)
    ├── evaluate.py                  # Consolidação estatística e tabelas de métricas (Fase 3)
    ├── plot_results.py              # Geração dos gráficos comparativos a 300 DPI (Fase 3)
    └── validate_results.py          # Suíte de validação e testes dos modelos/métricas (Fase 3)
```

---

## 3. Resultados Experimentais Consolidados (Fase 3)

Os experimentos envolveram **40 rodadas completas** (2 Variações $\times$ 4 Algoritmos $\times$ 5 Folds) com **240.000 predições out-of-fold** geradas.

### Tabela 1 – Desempenho dos Algoritmos na Variação A (Códigos Originais)

| Algoritmo | Acurácia Média | Precisão Macro Média | Recall Macro Médio | F1-Score Macro Médio | Tempo Médio de Treino (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0,8175 ± 0,0035** | **0,7470 ± 0,0077** | **0,6585 ± 0,0064** | **0,6823 ± 0,0071** | 4,113 ± 0,210 s |
| **K-Nearest Neighbors (KNN)** | 0,7916 ± 0,0041 | 0,6888 ± 0,0082 | 0,6340 ± 0,0061 | 0,6499 ± 0,0067 | **0,008 ± 0,001 s** |
| **Decision Tree** | 0,7243 ± 0,0045 | 0,6070 ± 0,0070 | 0,6119 ± 0,0065 | 0,6091 ± 0,0066 | 0,432 ± 0,065 s |
| *Dummy Classifier (Baseline)* | *0,7788 ± 0,0027* | *0,3894 ± 0,0014* | *0,5000 ± 0,0000* | *0,4378 ± 0,0009* | *0,001 ± 0,001 s* |

### Tabela 2 – Desempenho dos Algoritmos na Variação B (Consolidação de Categorias)

| Algoritmo | Acurácia Média | Precisão Macro Média | Recall Macro Médio | F1-Score Macro Médio | Tempo Médio de Treino (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0,8157 ± 0,0039** | **0,7437 ± 0,0105** | **0,6547 ± 0,0075** | **0,6780 ± 0,0084** | 4,407 ± 0,057 s |
| **K-Nearest Neighbors (KNN)** | 0,7917 ± 0,0041 | 0,6889 ± 0,0082 | 0,6341 ± 0,0062 | 0,6500 ± 0,0068 | **0,008 ± 0,001 s** |
| **Decision Tree** | 0,7213 ± 0,0050 | 0,6030 ± 0,0067 | 0,6079 ± 0,0053 | 0,6051 ± 0,0060 | 0,415 ± 0,017 s |
| *Dummy Classifier (Baseline)* | *0,7788 ± 0,0027* | *0,3894 ± 0,0014* | *0,5000 ± 0,0000* | *0,4378 ± 0,0009* | *0,001 ± 0,000 s* |

---

## 4. Principais Conclusões dos Experimentos

1. **Melhor Algoritmo Geral:** O **Random Forest** obteve o melhor desempenho em quase todas as métricas: F1-Score Macro de **0,6823**, Acurácia de **81,75%**, Precisão na Classe Inadimplente de **65,28%** e ROC-AUC de **0,7635**.
2. **Pior Algoritmo Geral:** A **Decision Tree sem poda** obteve o menor F1-Score Macro (**0,6091**) e acurácia de **72,43%** (inferior ao baseline ingênuo de 77,88%), sofrendo de sobreajuste (*overfitting*) e gerando 4.358 falsos positivos.
3. **Trade-off Sensibilidade vs Precisão:** A Árvore de Decisão detectou o maior número bruto de inadimplentes (Recall C1 = 41,03%), porém à custa de 4.358 alarmes falsos. O Random Forest detectou 2.478 inadimplentes com muito mais precisão (65,28% vs 38,46%).
4. **Impacto das Variações de Pré-processamento:** A consolidação de `EDUCATION` (0,5,6 $\rightarrow$ 4) e `MARRIAGE` (0 $\rightarrow$ 3) alterou menos de 0,45% das métricas globais, pois afeta apenas 1,33% dos clientes da base e a previsão é dominada pelos atrasos recentes (`PAY_0` a `PAY_6`).
5. **Confronto com Yeh & Lien (2009):** O artigo seminal utilizou o *Sorting Smoothing Method* (SSM) para avaliar a probabilidade contínua de inadimplência, destacando Redes Neurais como superiores e apontando limitações em árvores isoladas e KNN. Nossos resultados corroboram que árvores individuais sem poda sofrem em estabilidade e comprovam que ensembles (*Random Forest*, não avaliado no artigo original) atingem excelência preditiva para este problema.

---

## 5. Instruções de Instalação e Reprodução Integral

### Pré-requisitos
- **Python 3.9+** instalado.
- Gerenciador de pacotes `pip`.

### Passo a passo para execução:

1. **Configurar o ambiente virtual:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Executar todo o pipeline experimental (do zero aos gráficos e testes):**
   ```bash
   # Fase 1: Análise Exploratória
   python src/explore.py

   # Fase 2: Pré-processamento e Folds
   python src/preprocess.py
   python src/prepare_validation.py
   python src/validate_data.py

   # Fase 3: Treinamento, Avaliação, Gráficos e Validação dos Modelos
   python src/train.py
   python src/evaluate.py
   python src/plot_results.py
   python src/validate_results.py
   ```

Ao final da execução, todas as 19 tabelas CSV estarão em `results/metrics/`, todos os 23 gráficos em `results/charts/` e os 8 modelos persistidos em `models/`.
