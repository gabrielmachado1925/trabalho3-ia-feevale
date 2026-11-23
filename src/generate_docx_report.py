#!/usr/bin/env python3
"""
generate_docx_report.py (Versão 4.1 - Revisão Científica e Diagramação)

Gera o relatório acadêmico da Atividade Avaliativa 3 em formato Word (.docx),
preservando estritamente a capa original do modelo institucional (Modelo-Relatorio.docx)
e garantindo compatibilidade estrutural com o Microsoft Word (sem mensagens de recuperação).
"""

import os
import shutil
import zipfile
import xml.etree.ElementTree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"

ET.register_namespace('w', W_NS)
ET.register_namespace('r', R_NS)
ET.register_namespace('a', A_NS)
ET.register_namespace('wp', WP_NS)
ET.register_namespace('pic', PIC_NS)
ET.register_namespace('mc', MC_NS)


def w_elem(tag, attrs=None):
    elem = ET.Element(f"{{{W_NS}}}{tag}")
    if attrs:
        for k, v in attrs.items():
            if ':' in k:
                prefix, name = k.split(':')
                if prefix == 'w':
                    elem.attrib[f"{{{W_NS}}}{name}"] = v
                elif prefix == 'r':
                    elem.attrib[f"{{{R_NS}}}{name}"] = v
            else:
                elem.attrib[f"{{{W_NS}}}{k}"] = v
    return elem


def make_pPr(style=None, jc="both", line=360, first_line=708, before=0, after=0, keep_next=False):
    pPr = w_elem("pPr")
    if style:
        pPr.append(w_elem("pStyle", {"w:val": style}))
    if keep_next:
        pPr.append(w_elem("keepNext"))
    if before > 0 or after > 0 or line > 0:
        spacing_attrs = {}
        if before > 0:
            spacing_attrs["w:before"] = str(before)
        if after > 0:
            spacing_attrs["w:after"] = str(after)
        if line > 0:
            spacing_attrs["w:line"] = str(line)
            spacing_attrs["w:lineRule"] = "auto"
        pPr.append(w_elem("spacing", spacing_attrs))
    if first_line > 0:
        pPr.append(w_elem("ind", {"w:firstLine": str(first_line)}))
    if jc:
        pPr.append(w_elem("jc", {"w:val": jc}))
    return pPr


def make_r(text, bold=False, italic=False, size=24, font="Arial", color=None):
    r = w_elem("r")
    rPr = w_elem("rPr")
    rPr.append(w_elem("rFonts", {"w:ascii": font, "w:hAnsi": font, "w:cs": font}))
    if bold:
        rPr.append(w_elem("b"))
        rPr.append(w_elem("bCs"))
    if italic:
        rPr.append(w_elem("i"))
        rPr.append(w_elem("iCs"))
    if size:
        rPr.append(w_elem("sz", {"w:val": str(size)}))
        rPr.append(w_elem("szCs", {"w:val": str(size)}))
    if color:
        rPr.append(w_elem("color", {"w:val": color}))
    r.append(rPr)
    t = w_elem("t")
    t.text = text
    if text.startswith(" ") or text.endswith(" "):
        t.attrib["{http://www.w3.org/XML/1998/namespace}space"] = "preserve"
    r.append(t)
    return r


def create_paragraph(text, bold=False, italic=False, style=None, jc="both", line=360, first_line=708, before=0, after=0, size=24):
    p = w_elem("p")
    pPr = make_pPr(style=style, jc=jc, line=line, first_line=first_line, before=before, after=after)
    p.append(pPr)
    p.append(make_r(text, bold=bold, italic=italic, size=size))
    return p


def create_page_break():
    p = w_elem("p")
    p.append(make_pPr(jc="left", line=240, first_line=0, before=0, after=0))
    r = w_elem("r")
    r.append(w_elem("br", {"w:type": "page"}))
    p.append(r)
    return p


def create_heading_1(title):
    p = w_elem("p")
    pPr = make_pPr(jc="left", line=280, first_line=0, before=360, after=120, keep_next=True)
    p.append(pPr)
    p.append(make_r(title, bold=True, size=26, font="Arial"))
    return p


def create_heading_2(title):
    p = w_elem("p")
    pPr = make_pPr(jc="left", line=280, first_line=0, before=240, after=80, keep_next=True)
    p.append(pPr)
    p.append(make_r(title, bold=True, size=24, font="Arial"))
    return p


def create_caption(caption_text):
    p = w_elem("p")
    pPr = make_pPr(jc="center", line=240, first_line=0, before=160, after=60, keep_next=True)
    p.append(pPr)
    p.append(make_r(caption_text, bold=True, size=20, font="Arial"))
    return p


def create_source(source_text):
    p = w_elem("p")
    pPr = make_pPr(jc="center", line=240, first_line=0, before=40, after=140)
    p.append(pPr)
    p.append(make_r(source_text, italic=True, size=18, font="Arial"))
    return p


def create_table(headers, rows, col_widths, alignments=None):
    tbl = w_elem("tbl")
    tblPr = w_elem("tblPr")
    tblPr.append(w_elem("tblW", {"w:w": "0", "w:type": "auto"}))
    tblPr.append(w_elem("jc", {"w:val": "center"}))
    
    tblBorders = w_elem("tblBorders")
    for b in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        tblBorders.append(w_elem(b, {"w:val": "single", "w:sz": "4", "w:space": "0", "w:color": "BFBFBF"}))
    tblPr.append(tblBorders)
    
    tblCellMar = w_elem("tblCellMar")
    tblCellMar.append(w_elem("top", {"w:w": "80", "w:type": "dxa"}))
    tblCellMar.append(w_elem("left", {"w:w": "120", "w:type": "dxa"}))
    tblCellMar.append(w_elem("bottom", {"w:w": "80", "w:type": "dxa"}))
    tblCellMar.append(w_elem("right", {"w:w": "120", "w:type": "dxa"}))
    tblPr.append(tblCellMar)
    tbl.append(tblPr)

    tblGrid = w_elem("tblGrid")
    for w in col_widths:
        tblGrid.append(w_elem("gridCol", {"w:w": str(w)}))
    tbl.append(tblGrid)

    # Header Row com tblHeader e cantSplit
    tr_h = w_elem("tr")
    trPr_h = w_elem("trPr")
    trPr_h.append(w_elem("tblHeader"))
    trPr_h.append(w_elem("cantSplit"))
    tr_h.append(trPr_h)

    for i, h in enumerate(headers):
        tc = w_elem("tc")
        tcPr = w_elem("tcPr")
        tcPr.append(w_elem("tcW", {"w:w": str(col_widths[i]), "w:type": "dxa"}))
        tcPr.append(w_elem("shd", {"w:val": "clear", "w:color": "auto", "w:fill": "F2F2F2"}))
        tcPr.append(w_elem("vAlign", {"w:val": "center"}))
        tc.append(tcPr)
        align = alignments[i] if alignments else "center"
        p = w_elem("p")
        p.append(make_pPr(jc=align, line=240, first_line=0, before=40, after=40))
        p.append(make_r(h, bold=True, size=19, font="Arial"))
        tc.append(p)
        tr_h.append(tc)
    tbl.append(tr_h)

    # Data Rows com cantSplit
    for r_idx, row in enumerate(rows):
        tr = w_elem("tr")
        trPr = w_elem("trPr")
        trPr.append(w_elem("cantSplit"))
        tr.append(trPr)
        for i, val in enumerate(row):
            tc = w_elem("tc")
            tcPr = w_elem("tcPr")
            tcPr.append(w_elem("tcW", {"w:w": str(col_widths[i]), "w:type": "dxa"}))
            if r_idx % 2 == 1:
                tcPr.append(w_elem("shd", {"w:val": "clear", "w:color": "auto", "w:fill": "FAFAFA"}))
            tcPr.append(w_elem("vAlign", {"w:val": "center"}))
            tc.append(tcPr)
            align = alignments[i] if alignments else "left"
            p = w_elem("p")
            p.append(make_pPr(jc=align, line=240, first_line=0, before=30, after=30))
            is_bold = (i == 0 and len(val) > 0 and val[0].isdigit()) or ("Random Forest" in val) or ("Média" in val)
            p.append(make_r(val, bold=is_bold, size=18, font="Arial"))
            tc.append(p)
            tr.append(tc)
        tbl.append(tr)

    return tbl


def create_image_drawing(r_id, cx, cy, doc_id, name="Figura"):
    drawing = w_elem("drawing")
    inline = ET.Element(f"{{{WP_NS}}}inline", {
        "distT": "0", "distB": "0", "distL": "0", "distR": "0"
    })
    extent = ET.Element(f"{{{WP_NS}}}extent", {"cx": str(cx), "cy": str(cy)})
    inline.append(extent)

    effectExtent = ET.Element(f"{{{WP_NS}}}effectExtent", {"l": "0", "t": "0", "r": "0", "b": "0"})
    inline.append(effectExtent)

    docPr = ET.Element(f"{{{WP_NS}}}docPr", {"id": str(doc_id), "name": f"{name} {doc_id}"})
    inline.append(docPr)

    cNvGraphicFramePr = ET.Element(f"{{{WP_NS}}}cNvGraphicFramePr")
    graphicLocks = ET.Element(f"{{{A_NS}}}graphicFrameLocks", {"noChangeAspect": "1"})
    cNvGraphicFramePr.append(graphicLocks)
    inline.append(cNvGraphicFramePr)

    graphic = ET.Element(f"{{{A_NS}}}graphic")
    graphicData = ET.Element(f"{{{A_NS}}}graphicData", {"uri": PIC_NS})
    pic = ET.Element(f"{{{PIC_NS}}}pic")

    nvPicPr = ET.Element(f"{{{PIC_NS}}}nvPicPr")
    cNvPr = ET.Element(f"{{{PIC_NS}}}cNvPr", {"id": str(doc_id), "name": f"Picture {doc_id}"})
    cNvPicPr = ET.Element(f"{{{PIC_NS}}}cNvPicPr")
    nvPicPr.append(cNvPr)
    nvPicPr.append(cNvPicPr)
    pic.append(nvPicPr)

    blipFill = ET.Element(f"{{{PIC_NS}}}blipFill")
    blip = ET.Element(f"{{{A_NS}}}blip", {f"{{{R_NS}}}embed": r_id, "cstate": "print"})
    stretch = ET.Element(f"{{{A_NS}}}stretch")
    fillRect = ET.Element(f"{{{A_NS}}}fillRect")
    stretch.append(fillRect)
    blipFill.append(blip)
    blipFill.append(stretch)
    pic.append(blipFill)

    spPr = ET.Element(f"{{{PIC_NS}}}spPr")
    xfrm = ET.Element(f"{{{A_NS}}}xfrm")
    off = ET.Element(f"{{{A_NS}}}off", {"x": "0", "y": "0"})
    ext = ET.Element(f"{{{A_NS}}}ext", {"cx": str(cx), "cy": str(cy)})
    xfrm.append(off)
    xfrm.append(ext)
    spPr.append(xfrm)
    prstGeom = ET.Element(f"{{{A_NS}}}prstGeom", {"prst": "rect"})
    prstGeom.append(ET.Element(f"{{{A_NS}}}avLst"))
    spPr.append(prstGeom)
    pic.append(spPr)

    graphicData.append(pic)
    graphic.append(graphicData)
    inline.append(graphic)
    drawing.append(inline)

    p = w_elem("p")
    p.append(make_pPr(jc="center", line=240, first_line=0, before=80, after=60, keep_next=True))
    p.append(drawing)
    return p


def build_full_report():
    print("Iniciando construção do relatório acadêmico revisado (Fase 4.1)...")
    tmp_dir = "/tmp/docx_work_fase41"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    with zipfile.ZipFile("Modelo-Relatorio.docx", "r") as z:
        z.extractall(tmp_dir)

    doc_xml_path = os.path.join(tmp_dir, "word/document.xml")

    # Ler o XML original para capturar exatamente os atributos da raiz <w:document>
    with open(doc_xml_path, "r", encoding="utf-8") as f:
        original_doc_xml = f.read()

    body_tag_idx = original_doc_xml.find("<w:body>")
    doc_header = original_doc_xml[:body_tag_idx + 8]  # Inclui <w:body>

    # Parse da árvore para manipular elementos do body
    doc_tree = ET.parse(doc_xml_path)
    root = doc_tree.getroot()
    body = root.find(f"{{{W_NS}}}body")

    body_children = list(body)
    cover_elements = body_children[:30]  # Capa intacta
    final_sectPr = body_children[-1]   # sectPr da Seção 2

    for elem in list(body):
        body.remove(elem)
    for elem in cover_elements:
        body.append(elem)

    # Imagens do relatório com dimensões calibradas
    image_specs = [
        ("01_distribuicao_target.png", 12.5, 8.93, "Figura 1 – Distribuição de frequência do atributo alvo (target)"),
        ("03_distribuicao_limite_credito.png", 11.5, 10.06, "Figura 2 – Distribuição do limite de crédito por classe de inadimplência"),
        ("08_evolucao_faturas_pagamentos.png", 13.5, 8.44, "Figura 3 – Evolução temporal das faturas e pagamentos amortizados"),
        ("12_comparacao_f1_macro.png", 13.5, 7.94, "Figura 4 – Comparativo de F1-Score Macro entre os algoritmos e variações"),
        ("report_matrizes_confusao.png", 13.5, 11.48, "Figura 5 – Matrizes de confusão consolidadas dos algoritmos (Variação A)"),
        ("report_curva_roc.png", 12.5, 9.17, "Figura 6 – Curvas ROC dos modelos preditivos (Variação A)"),
        ("report_curva_pr.png", 12.5, 9.17, "Figura 7 – Curvas Precision-Recall para a classe inadimplente (Variação A)"),
    ]

    media_dir = os.path.join(tmp_dir, "word/media")
    os.makedirs(media_dir, exist_ok=True)

    rel_map = {}
    new_rels = []
    doc_id_counter = 200

    for idx, (img_name, width_cm, height_cm, cap_title) in enumerate(image_specs, start=1):
        src_path = os.path.join("results/charts", img_name)
        target_name = f"image_rep_{idx}.png"
        dst_path = os.path.join(media_dir, target_name)
        shutil.copyfile(src_path, dst_path)

        rel_id = f"rIdFig{idx}"
        new_rels.append((rel_id, f"media/{target_name}"))
        rel_map[img_name] = (rel_id, width_cm, height_cm, doc_id_counter + idx, cap_title)

    # Atualizar word/_rels/document.xml.rels com formato limpo sem prefixos estranhos
    rels_xml_path = os.path.join(tmp_dir, "word/_rels/document.xml.rels")
    with open(rels_xml_path, "r", encoding="utf-8") as f:
        rels_content = f.read()

    rel_insert = ""
    for r_id, r_target in new_rels:
        rel_insert += f'<Relationship Id="{r_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{r_target}"/>'
    
    close_tag = "</Relationships>"
    idx_close = rels_content.find(close_tag)
    rels_content_updated = rels_content[:idx_close] + rel_insert + rels_content[idx_close:]
    with open(rels_xml_path, "w", encoding="utf-8") as f:
        f.write(rels_content_updated)

    # Atualizar [Content_Types].xml garantindo que png está cadastrado
    ct_xml_path = os.path.join(tmp_dir, "[Content_Types].xml")
    with open(ct_xml_path, "r", encoding="utf-8") as f:
        ct_content = f.read()

    if 'Extension="png"' not in ct_content:
        idx_types_close = ct_content.find("</Types>")
        ct_content = ct_content[:idx_types_close] + '<Default Extension="png" ContentType="image/png"/>' + ct_content[idx_types_close:]
        with open(ct_xml_path, "w", encoding="utf-8") as f:
            f.write(ct_content)

    def add(elem):
        body.append(elem)

    def add_p(text):
        add(create_paragraph(text))

    def add_figure(img_key):
        r_id, w_cm, h_cm, d_id, cap_text = rel_map[img_key]
        add(create_caption(cap_text))
        cx = int(w_cm * 360000)
        cy = int(h_cm * 360000)
        add(create_image_drawing(r_id, cx, cy, d_id))
        add(create_source("Fonte: Elaborado pelos autores a partir dos dados experimentais (2026)."))

    # =========================================================================
    # CONTEÚDO ACADÊMICO REVISADO (PORTUGUÊS NATURAL, DIRETO E PRECISO)
    # =========================================================================

    # 1 INTRODUÇÃO
    add(create_heading_1("1 INTRODUÇÃO"))
    add_p("O aprendizado de máquina supervisionado é uma área da inteligência artificial aplicada voltada ao desenvolvimento de algoritmos capazes de aprender padrões em dados históricos rotulados para realizar previsões sobre novos exemplos. No setor financeiro, a previsão de inadimplência de crédito é uma aplicação clássica e de grande relevância, permitindo que as instituições avaliem o perfil de risco dos tomadores e tomem decisões fundamentadas sobre limites, cobranças e concessões.")
    add_p("A concessão de crédito em cartões bancários envolve decisões recorrentes tomadas em cenários de incerteza. Modelos preditivos auxiliam nesse processo ao analisar variáveis cadastrais e o histórico de uso do cartão para estimar a probabilidade de um cliente não pagar a fatura no período seguinte.")
    add_p("Este trabalho tem como objetivo desenvolver, avaliar e comparar três algoritmos supervisionados clássicos de aprendizado de máquina — Árvore de Decisão (Decision Tree), Floresta Aleatória (Random Forest) e K-Vizinhos Mais Próximos (K-Nearest Neighbors — KNN) —, além de um classificador de referência baseado na classe mais frequente (DummyClassifier), na tarefa de prever a inadimplência de clientes de cartão de crédito no mês seguinte.")
    add_p("Os objetivos específicos compreendem: (a) caracterizar o conjunto de dados público 'Default of Credit Card Clients', identificando distribuições, tipos de variáveis e casos de desbalanceamento; (b) implementar e avaliar duas variações de pré-processamento para medir o efeito do agrupamento de categorias não documentadas; (c) utilizar validação cruzada estratificada em cinco folds com controle de repetições, prevenindo que perfis preditivos idênticos estejam simultaneamente em treino e teste; (d) comparar os algoritmos com base em Acurácia, Precisão Macro, Recall Macro, F1-Score Macro e tempo de execução; (e) analisar os resultados individuais de cada classe e os trade-offs práticos entre falsos positivos e falsos negativos; e (f) contextualizar os achados em relação ao estudo original de Yeh e Lien (2009).")

    # 2 APRESENTAÇÃO DO DATASET
    add(create_heading_1("2 APRESENTAÇÃO DO DATASET"))
    add_p("O conjunto de dados utilizado é denominado 'Default of Credit Card Clients', registrado sob o identificador ID #350 e DOI 10.24432/C55S3H no repositório UCI Machine Learning Repository (YEH, 2016). A base foi coletada por I-Cheng Yeh e Che-hui Lien, da Universidade Chung Hua, e analisada no periódico Expert Systems with Applications (YEH; LIEN, 2009).")
    add_p("Os dados contêm registros de 30.000 clientes de uma emissora de cartões de crédito em Taiwan, cobrindo o período de abril a setembro de 2005. O objetivo central do conjunto é fornecer informações cadastrais e financeiras para antecipar se o cliente entrará em inadimplência no ciclo seguinte (outubro de 2005).")
    add_p("A escolha desta base se justifica por ser uma base de referência em estudos de aprendizado de máquina, por apresentar um problema prático de classificação com desbalanceamento moderado de classes e por conter séries financeiras de seis meses de faturas e valores amortizados.")
    add_p("O problema proposto é de classificação binária supervisionada. A variável alvo (target) é 'default payment next month', que assume dois valores: (a) Classe 0, indicando cliente adimplente (não inadimplente), aquele que realizou ao menos o pagamento mínimo exigido na fatura de outubro de 2005; e (b) Classe 1, indicando cliente inadimplente, que não realizou o pagamento estipulado.")

    # 3 CARACTERIZAÇÃO DOS DADOS
    add(create_heading_1("3 CARACTERIZAÇÃO DOS DADOS"))
    add_p("A base contém 30.000 instâncias e 25 atributos. O primeiro atributo é 'ID' (identificador numérico de 1 a 30.000), que serve exclusivamente para rastreabilidade cadastral e foi removido das matrizes de preditores fornecidas aos modelos. Desconsiderando o 'ID' e a variável alvo 'default payment next month', restam 23 atributos preditores.")
    add_p("A verificação dos dados confirmou que não existem valores nulos ou faltantes (zero NaNs em todas as colunas). Contudo, a documentação oficial da UCI apresenta categorias não detalhadas: (a) em 'EDUCATION', além das opções documentadas 1 (pós-graduação), 2 (universidade), 3 (ensino médio) e 4 (outros), constam os códigos 0 (14 instâncias), 5 (280 instâncias) e 6 (51 instâncias); e (b) em 'MARRIAGE', além dos códigos 1 (casado), 2 (solteiro) e 3 (outros), verifica-se o código 0 em 54 instâncias.")
    add_p("Nas variáveis de histórico de pagamento ('PAY_0' a 'PAY_6'), a documentação da UCI define que o código -1 representa pagamento em dia e os valores positivos de 1 a 9 indicam atraso na quantidade correspondente de meses. No entanto, os dados também contêm os códigos -2 e 0, não explicados na documentação oficial. Na prática de análise de crédito, sugere-se na literatura que -2 indique cartão sem movimentação e 0 o pagamento mínimo ou uso de crédito rotativo, embora essas interpretações não sejam confirmadas pelo repositório.")
    add_p("Quanto a repetições, foram localizadas 108 linhas associadas a 52 grupos de preditores idênticos. Dentre essas, 35 grupos (71 linhas) têm o mesmo valor de target, enquanto 17 grupos (37 linhas) apresentam perfis idênticos com targets divergentes, evidenciando que perfis com os mesmos dados cadastrais e financeiros podem ter comportamentos de pagamento diferentes.")

    # Tabela 1 – Caracterização dos Atributos
    add(create_caption("Tabela 1 – Caracterização dos Atributos do Dataset"))
    headers_t1 = ["Atributo", "Tipo Semântico", "Descrição Resumida", "Conjunto / Intervalo de Valores", "Valores Nulos"]
    widths_t1 = [1500, 1500, 2900, 2200, 976]
    aligns_t1 = ["left", "left", "left", "center", "center"]
    rows_t1 = [
        ["ID", "Identificador", "Identificador sequencial único do cliente", "1 a 30.000 (inteiro)", "0 (0,0%)"],
        ["LIMIT_BAL", "Contínuo", "Limite de crédito concedido (NT$)", "10.000 a 1.000.000", "0 (0,0%)"],
        ["SEX", "Nominal", "Gênero do titular do cartão", "1 = Masculino; 2 = Feminino", "0 (0,0%)"],
        ["EDUCATION", "Nominal", "Nível de escolaridade formal", "1=Pós-grad., 2=Univ., 3=Médio, 4=Outros, [0,5,6]", "0 (0,0%)"],
        ["MARRIAGE", "Nominal", "Estado civil cadastral", "1 = Casado; 2 = Solteiro; 3 = Outros; [0]", "0 (0,0%)"],
        ["AGE", "Discreto", "Idade cronológica do cliente (anos)", "21 a 79 anos", "0 (0,0%)"],
        ["PAY_0", "Ordinal", "Status de pagamento em Setembro/2005", "-2=Sem uso, -1=Em dia, 0=Mínimo, 1 a 8=Atraso", "0 (0,0%)"],
        ["PAY_2", "Ordinal", "Status de pagamento em Agosto/2005", "-2=Sem uso, -1=Em dia, 0=Mínimo, 1 a 8=Atraso", "0 (0,0%)"],
        ["PAY_3", "Ordinal", "Status de pagamento em Julho/2005", "-2=Sem uso, -1=Em dia, 0=Mínimo, 1 a 8=Atraso", "0 (0,0%)"],
        ["PAY_4", "Ordinal", "Status de pagamento em Junho/2005", "-2=Sem uso, -1=Em dia, 0=Mínimo, 1 a 8=Atraso", "0 (0,0%)"],
        ["PAY_5", "Ordinal", "Status de pagamento em Maio/2005", "-2=Sem uso, -1=Em dia, 0=Mínimo, 1 a 8=Atraso", "0 (0,0%)"],
        ["PAY_6", "Ordinal", "Status de pagamento em Abril/2005", "-2=Sem uso, -1=Em dia, 0=Mínimo, 1 a 8=Atraso", "0 (0,0%)"],
        ["BILL_AMT1", "Contínuo", "Valor da fatura em Setembro/2005 (NT$)", "-165.580 a 964.511", "0 (0,0%)"],
        ["BILL_AMT2", "Contínuo", "Valor da fatura em Agosto/2005 (NT$)", "-69.777 a 983.931", "0 (0,0%)"],
        ["BILL_AMT3", "Contínuo", "Valor da fatura em Julho/2005 (NT$)", "-157.264 a 1.664.089", "0 (0,0%)"],
        ["BILL_AMT4", "Contínuo", "Valor da fatura em Junho/2005 (NT$)", "-170.000 a 891.586", "0 (0,0%)"],
        ["BILL_AMT5", "Contínuo", "Valor da fatura em Maio/2005 (NT$)", "-81.334 a 927.171", "0 (0,0%)"],
        ["BILL_AMT6", "Contínuo", "Valor da fatura em Abril/2005 (NT$)", "-339.603 a 961.664", "0 (0,0%)"],
        ["PAY_AMT1", "Contínuo", "Valor amortizado em Setembro/2005 (NT$)", "0 a 873.552", "0 (0,0%)"],
        ["PAY_AMT2", "Contínuo", "Valor amortizado em Agosto/2005 (NT$)", "0 a 1.684.259", "0 (0,0%)"],
        ["PAY_AMT3", "Contínuo", "Valor amortizado em Julho/2005 (NT$)", "0 a 896.040", "0 (0,0%)"],
        ["PAY_AMT4", "Contínuo", "Valor amortizado em Junho/2005 (NT$)", "0 a 621.000", "0 (0,0%)"],
        ["PAY_AMT5", "Contínuo", "Valor amortizado em Maio/2005 (NT$)", "0 a 426.529", "0 (0,0%)"],
        ["PAY_AMT6", "Contínuo", "Valor amortizado em Abril/2005 (NT$)", "0 a 528.666", "0 (0,0%)"],
        ["default...", "Target Binário", "Inadimplência no mês seguinte", "0 = Adimplente; 1 = Inadimplente", "0 (0,0%)"]
    ]
    add(create_table(headers_t1, rows_t1, widths_t1, aligns_t1))
    add(create_source("Fonte: Elaborado pelos autores a partir do repositório UCI (YEH, 2016)."))

    add_p("A distribuição do atributo alvo registra 23.364 instâncias da Classe 0 (77,88%) e 6.636 da Classe 1 (22,12%), com razão de desbalanceamento de aproximadamente 3,52 clientes adimplentes para cada inadimplente.")
    add_figure("01_distribuicao_target.png")

    add_p("O desbalanceamento dos atributos categóricos foi avaliado sob três critérios: (a) Razão de Desbalanceamento (IR), calculada pela razão entre a frequência da categoria mais comum e a da categoria menos comum; (b) Frequência Modal, que mede o percentual da categoria predominante; e (c) Entropia de Shannon Normalizada, que avalia a dispersão das frequências relativas (valores próximos de 0 indicam maior concentração).")
    add_p("Entre os atributos puramente nominais demográficos ('SEX', 'EDUCATION', 'MARRIAGE'):")
    add_p("- Sob o critério de Razão de Desbalanceamento (IR), o atributo mais desbalanceado é EDUCATION: a categoria mais frequente é a 2 (Universidade), com 14.030 clientes, e a menos frequente é a 0, com 14 clientes. A razão exata é 14.030 / 14 = 1.002,14. Em MARRIAGE, a categoria 2 (Solteiro) tem 15.964 clientes e a categoria 0 tem 54, resultando em IR de 295,63. Em SEX, a categoria 2 (Feminino) tem 18.112 clientes e a 1 tem 11.888 (IR de 1,52).")
    add_p("- Sob o critério de Frequência Modal, a categoria predominante em MARRIAGE é a 2 (Solteiro), que reúne 15.964 clientes, correspondendo a 53,21% da base, seguida pela categoria 1 (Casado) com 13.659 clientes (45,53%). Em SEX, a categoria feminina reúne 60,37% dos registros.")
    add_p("- Sob o critério de Entropia Normalizada, o menor valor entre os nominais ocorre em MARRIAGE (0,5440), seguido por EDUCATION (0,5653), indicando maior concentração em MARRIAGE.")
    add_p("Se forem considerados também os atributos ordinais de histórico de pagamento ('PAY_0' a 'PAY_6'), a categoria 0 concentra entre 49% e 56% das instâncias, com casos raros de códigos extremos (como o código 8 em PAY_5 com apenas 1 registro), elevando a razão IR nesses atributos específicos.")
    add_figure("03_distribuicao_limite_credito.png")
    add_figure("08_evolucao_faturas_pagamentos.png")

    # 4 PRÉ-PROCESSAMENTO DOS DADOS (com quebra de página para evitar início no final da folha)
    add(create_page_break())
    add(create_heading_1("4 PRÉ-PROCESSAMENTO DOS DADOS"))
    add_p("Os procedimentos de pré-processamento foram estruturados para manter a rastreabilidade e evitar o vazamento de informações entre treino e teste:")
    add_p("1. Remoção do ID: A coluna 'ID' foi retirada do conjunto de preditores, evitando que um identificador sequencial sem relevância preditiva influencie o aprendizado.")
    add_p("2. Valores Numéricos Negativos: Valores negativos nas faturas ('BILL_AMT1' a 'BILL_AMT6') foram mantidos, pois indicam créditos decorrentes de pagamentos antecipados ou estornos válidos. Não foram descartados outliers.")
    add_p("3. Codificação e Padronização: As variáveis nominais ('SEX', 'EDUCATION', 'MARRIAGE') foram transformadas por One-Hot Encoding sem descarte de colunas. Para o KNN, os atributos contínuos e ordinais foram padronizados por StandardScaler (média zero e desvio-padrão um). Para a Árvore de Decisão e o Random Forest, os preditores numéricos permaneceram na escala original, uma vez que árvores realizam divisões baseadas em limiares ordenados, sendo invariantes a transformações monotônicas.")
    add_p("4. Ajuste Exclusivo no Treino: Os transformadores que aprendem parâmetros (como as médias do StandardScaler e as categorias do One-Hot Encoding) foram configurados dentro de pipelines com ColumnTransformer, sendo ajustados exclusivamente nos dados de treino de cada fold e aplicados aos dados de validação.")
    add_p("Para atender à solicitação de avaliar variações no pré-processamento, foram estruturadas duas abordagens:")
    add_p("- Variação A (Referência): Mantém os códigos originais de EDUCATION (0 a 6) e MARRIAGE (0 a 3). Após a codificação One-Hot, a matriz resultante possui 33 atributos transformados (20 contínuos/ordinais e 13 colunas binárias).")
    add_p("- Variação B (Consolidação de Categorias): Agrupa os códigos 0, 5 e 6 de EDUCATION na categoria 4 (Outros) e o código 0 de MARRIAGE na categoria 3 (Outros). Essa simplificação reduz o espaço transformado para 29 atributos (20 contínuos/ordinais e 9 colunas binárias).")
    add_p("A contagem direta sobre os dados identificou a quantidade de clientes alterados na Variação B: (a) alterações em EDUCATION: 345 registros; (b) alterações em MARRIAGE: 54 registros; (c) clientes alterados simultaneamente em ambas as variáveis: zero registros; (d) total de clientes únicos alterados: 399 registros; e (e) percentual correspondente da base: 1,3300%.")

    # Tabela 2 – Comparação das Variações de Pré-processamento
    add(create_caption("Tabela 2 – Comparação dos Procedimentos de Pré-processamento entre as Variações A e B"))
    headers_t2 = ["Dimensão de Pré-processamento", "Variação A (Referência)", "Variação B (Consolidação)"]
    widths_t2 = [2800, 3136, 3140]
    aligns_t2 = ["left", "left", "left"]
    rows_t2 = [
        ["Identificador 'ID'", "Excluído dos preditores (mantido em metadados)", "Excluído dos preditores (mantido em metadados)"],
        ["Tratamento de 'EDUCATION'", "Códigos originais preservados (0 a 6)", "Códigos 0, 5 e 6 agrupados na categoria 4 (Outros)"],
        ["Tratamento de 'MARRIAGE'", "Códigos originais preservados (0 a 3)", "Código 0 agrupado na categoria 3 (Outros)"],
        ["Clientes únicos alterados", "0 clientes (0,0%)", "399 clientes únicos (1,3300% da base)"],
        ["One-Hot Encoding ('SEX', 'EDU', 'MAR')", "13 colunas binárias geradas (2 + 7 + 4)", "9 colunas binárias geradas (2 + 4 + 3)"],
        ["Atributos Numéricos e Ordinais", "20 atributos contínuos/ordinais preservados", "20 atributos contínuos/ordinais preservados"],
        ["Dimensionalidade Transformada Total", "33 atributos na entrada dos modelos", "29 atributos na entrada dos modelos"],
        ["Padronização (StandardScaler)", "Ajustado por fold exclusivamente para o KNN", "Ajustado por fold exclusivamente para o KNN"],
        ["Partição de Validação Cruzada", "5 Folds idênticos (StratifiedGroupKFold)", "5 Folds idênticos (StratifiedGroupKFold)"]
    ]
    add(create_table(headers_t2, rows_t2, widths_t2, aligns_t2))
    add(create_source("Fonte: Elaborado pelos autores (2026)."))

    # 5 METODOLOGIA EXPERIMENTAL
    add(create_heading_1("5 METODOLOGIA EXPERIMENTAL"))
    add_p("A comparação experimental utilizou os três algoritmos solicitados e um classificador de referência, instanciados pela biblioteca Scikit-learn:")
    add_p("1. Árvore de Decisão: Utilizou-se o DecisionTreeClassifier, que implementa uma versão otimizada do algoritmo CART (BREIMAN et al., 1984). Foram mantidas as configurações padrão com critério de Gini, divisão 'best', semente aleatória random_state=42 e sem restrição de profundidade máxima (max_depth=None), com min_samples_split=2.")
    add_p("2. Floresta Aleatória (Random Forest): Utilizou-se o RandomForestClassifier (BREIMAN, 2001) com 100 árvores (n_estimators=100), critério de Gini, subamostragem de atributos max_features='sqrt', semente random_state=42 e n_jobs=1.")
    add_p("3. K-Vizinhos Mais Próximos (KNN): Utilizou-se o KNeighborsClassifier com cinco vizinhos (n_neighbors=5), pesos uniformes e distância Euclidiana padrão (COVER; HART, 1967).")
    add_p("4. Classificador de Referência (DummyClassifier): Configurado com strategy='most_frequent', prevendo sempre a classe majoritária (Classe 0) para fornecer uma linha de base comparativa.")
    add_p("A metodologia adotada foi a validação cruzada estratificada em cinco folds com agrupamento via StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42). O agrupamento foi definido pela coluna 'feature_group_id', impedindo que perfis preditivos idênticos fossem divididos entre treino e validação. As mesmas cinco partições foram aplicadas a todos os algoritmos nas duas variações.")
    add_p("Foram executadas 40 rodadas no total (2 variações x 4 algoritmos x 5 folds), gerando 240.000 previsões out-of-fold. Os ensaios foram realizados em ambiente controlado em computador Apple M2 com 16 GB de memória sob macOS e Python 3.9.6, com tempos registrados pelo temporizador time.perf_counter().")
    add_p("As métricas acompanhadas foram Acurácia, Precisão Macro, Recall Macro, F1-Score Macro e tempos de treino e predição. A métrica principal adotada para ordenar os modelos é o F1-Score Macro, por avaliar o equilíbrio entre as classes sem ser inflada pela alta frequência de clientes adimplentes.")

    # 6 RESULTADOS EXPERIMENTAIS
    add(create_page_break())
    add(create_heading_1("6 RESULTADOS EXPERIMENTAIS"))
    add_p("Os resultados médios e seus respectivos desvios-padrão entre os cinco folds de validação cruzada são apresentados nas Tabelas 3 e 4.")

    # 6.1 Variação A
    add(create_heading_2("6.1 Variação A – Categorias Originais"))
    add(create_caption("Tabela 3 – Desempenho dos Algoritmos na Variação A (Códigos Originais de Escolaridade e Estado Civil)"))
    headers_t3 = ["Algoritmo", "Acurácia Média", "Precisão Macro Média", "Recall Macro Médio", "F1-Score Macro Médio", "Tempo Médio de Treino (s)"]
    widths_t3 = [2300, 1350, 1350, 1350, 1350, 1376]
    aligns_t3 = ["left", "center", "center", "center", "center", "center"]
    rows_t3 = [
        ["Random Forest", "81,75% ± 0,35%", "0,7470 ± 0,0077", "0,6585 ± 0,0064", "0,6823 ± 0,0071", "4,113 ± 0,210 s"],
        ["KNN (k=5)", "79,16% ± 0,41%", "0,6888 ± 0,0082", "0,6340 ± 0,0061", "0,6499 ± 0,0067", "0,008 ± 0,001 s"],
        ["Decision Tree", "72,43% ± 0,45%", "0,6070 ± 0,0070", "0,6119 ± 0,0065", "0,6091 ± 0,0066", "0,432 ± 0,065 s"],
        ["Dummy (Baseline)", "77,88% ± 0,27%", "0,3894 ± 0,0014", "0,5000 ± 0,0000", "0,4378 ± 0,0009", "0,001 ± 0,001 s"]
    ]
    add(create_table(headers_t3, rows_t3, widths_t3, aligns_t3))
    add(create_source("Fonte: Dados experimentais obtidos pelos autores via validação cruzada em 5 folds (2026)."))

    # 6.2 Variação B
    add(create_heading_2("6.2 Variação B – Consolidação de Categorias"))
    add(create_caption("Tabela 4 – Desempenho dos Algoritmos na Variação B (Consolidação de Categorias Não Documentadas)"))
    rows_t4 = [
        ["Random Forest", "81,57% ± 0,39%", "0,7437 ± 0,0105", "0,6547 ± 0,0075", "0,6780 ± 0,0084", "4,407 ± 0,057 s"],
        ["KNN (k=5)", "79,17% ± 0,41%", "0,6889 ± 0,0082", "0,6341 ± 0,0062", "0,6500 ± 0,0068", "0,008 ± 0,001 s"],
        ["Decision Tree", "72,13% ± 0,50%", "0,6030 ± 0,0067", "0,6079 ± 0,0053", "0,6051 ± 0,0060", "0,415 ± 0,017 s"],
        ["Dummy (Baseline)", "77,88% ± 0,27%", "0,3894 ± 0,0014", "0,5000 ± 0,0000", "0,4378 ± 0,0009", "0,001 ± 0,000 s"]
    ]
    add(create_table(headers_t3, rows_t4, widths_t3, aligns_t3))
    add(create_source("Fonte: Dados experimentais obtidos pelos autores via validação cruzada em 5 folds (2026)."))

    add_figure("12_comparacao_f1_macro.png")

    # 6.3 Comparação Geral dos Algoritmos
    add(create_heading_2("6.3 Comparação Geral dos Algoritmos"))
    add_p("A análise das métricas indica comportamentos distintos entre os modelos:")
    add_p("- Melhor algoritmo no desempenho geral: O Random Forest obteve o maior equilíbrio geral, com F1-Score Macro de 0,6823 ± 0,0071 e Acurácia de 81,75% ± 0,35% na Variação A, além da maior precisão na classe inadimplente (65,28%).")
    add_p("- Pior algoritmo no desempenho geral: A Árvore de Decisão sem poda obteve o menor F1-Score Macro (0,6091 ± 0,0066) e a menor acurácia (72,43% ± 0,45%), ficando abaixo da taxa base do DummyClassifier (77,88%).")
    add_p("- Algoritmo mais rápido no treinamento: O KNN precisou de apenas 0,008 ± 0,001 s por fold, pois não constrói uma árvore ou regras explícitas durante o treino.")
    add_p("- Algoritmo mais rápido na predição: A Árvore de Decisão teve a menor latência de inferência (0,56 ms por 1.000 amostras), contra 12,80 ms do Random Forest.")
    add_p("- Maior latência na predição: O KNN registrou 23,14 ms por 1.000 amostras na Variação A e 19,23 ms na Variação B, devido ao cálculo de distâncias euclidianas em relação a todos os pontos do conjunto de treinamento.")

    # 6.4 Desempenho por Classe e Matrizes de Confusão
    add(create_heading_2("6.4 Desempenho Individual por Classe e Matrizes de Confusão"))
    add_p("Em virtude do desbalanceamento (77,88% adimplentes e 22,12% inadimplentes), o desempenho difere marcadamente entre as classes:")
    add_p("- Classe 0 (Não Inadimplente): Todos os algoritmos apresentaram desempenho elevado. Na Variação A, o Random Forest obteve precisão de 84,13%, recall de 94,36% e F1-Score de 0,8895, reconhecendo com facilidade os clientes que pagam em dia.")
    add_p("- Classe 1 (Inadimplente): Esta classe se mostrou consideravelmente mais difícil de prever. O Random Forest alcançou precisão de 65,28%, recall de 37,34% e F1-Score de 0,4751. Já a Árvore de Decisão obteve o maior recall na classe devedora (41,03%), detectando 2.723 clientes inadimplentes, mas com baixa precisão (38,46%) e gerando 4.358 falsos alarmes.")

    # Tabela 5 – Desempenho por Classe
    add(create_caption("Tabela 5 – Desempenho Preditivo Individual das Classes 0 e 1 (Variações A e B)"))
    headers_t5 = ["Variação", "Algoritmo", "Precisão C0", "Recall C0", "F1-Score C0", "Precisão C1", "Recall C1", "F1-Score C1"]
    widths_t5 = [1100, 1600, 1060, 1060, 1060, 1060, 1060, 1076]
    aligns_t5 = ["center", "left", "center", "center", "center", "center", "center", "center"]
    rows_t5 = [
        ["Var. A", "Random Forest", "0,8413", "0,9436", "0,8895", "0,6528", "0,3734", "0,4751"],
        ["Var. A", "KNN (k=5)", "0,8326", "0,9166", "0,8726", "0,5449", "0,3514", "0,4273"],
        ["Var. A", "Decision Tree", "0,8293", "0,8135", "0,8213", "0,3846", "0,4103", "0,3970"],
        ["Var. A", "Dummy", "0,7788", "1,0000", "0,8756", "0,0000", "0,0000", "0,0000"],
        ["Var. B", "Random Forest", "0,8397", "0,9435", "0,8886", "0,6476", "0,3659", "0,4676"],
        ["Var. B", "KNN (k=5)", "0,8327", "0,9167", "0,8727", "0,5451", "0,3514", "0,4273"],
        ["Var. B", "Decision Tree", "0,8275", "0,8113", "0,8193", "0,3784", "0,4045", "0,3910"],
        ["Var. B", "Dummy", "0,7788", "1,0000", "0,8756", "0,0000", "0,0000", "0,0000"]
    ]
    add(create_table(headers_t5, rows_t5, widths_t5, aligns_t5))
    add(create_source("Fonte: Dados experimentais obtidos pelos autores a partir das previsões out-of-fold (2026)."))

    # Tabela 6 – Matrizes de Confusão
    add(create_caption("Tabela 6 – Matrizes de Confusão Consolidadas (30.000 Previsões Out-Of-Fold)"))
    headers_t6 = ["Variação", "Algoritmo", "Verd. Negativo (VN)", "Falso Positivo (FP)", "Falso Negativo (FN)", "Verd. Positivo (VP)"]
    widths_t6 = [1100, 1700, 1570, 1570, 1570, 1566]
    aligns_t6 = ["center", "left", "center", "center", "center", "center"]
    rows_t6 = [
        ["Var. A", "Random Forest", "22.046", "1.318", "4.158", "2.478"],
        ["Var. A", "KNN (k=5)", "21.416", "1.948", "4.304", "2.332"],
        ["Var. A", "Decision Tree", "19.006", "4.358", "3.913", "2.723"],
        ["Var. A", "Dummy", "23.364", "0", "6.636", "0"],
        ["Var. B", "Random Forest", "22.043", "1.321", "4.208", "2.428"],
        ["Var. B", "KNN (k=5)", "21.418", "1.946", "4.304", "2.332"],
        ["Var. B", "Decision Tree", "18.955", "4.409", "3.952", "2.684"],
        ["Var. B", "Dummy", "23.364", "0", "6.636", "0"]
    ]
    add(create_table(headers_t6, rows_t6, widths_t6, aligns_t6))
    add(create_source("Fonte: Dados experimentais obtidos pelos autores a partir das previsões out-of-fold (2026)."))

    add_figure("report_matrizes_confusao.png")
    add_figure("report_curva_roc.png")
    add_figure("report_curva_pr.png")

    # 6.5 Impacto do Pré-processamento
    add(create_heading_2("6.5 Impacto do Pré-processamento (Variação A vs. B)"))
    add_p("A comparação entre a Variação A (categorias preservadas) e a Variação B (categorias consolidadas) indicou diferenças pequenas em todas as métricas calculadas. Para o Random Forest, a acurácia média passou de 81,75% para 81,57% (diferença de -0,18 ponto percentual). O F1-Score Macro médio passou de 0,682269 para 0,678040: a diferença exata sobre os valores não arredondados é de -0,004228 (redução de 0,42 ponto percentual, ou -0,62% em termos relativos); se calculada sobre os valores previamente arredondados a quatro casas (0,6780 - 0,6823), a diferença resulta em -0,0043.")
    add_p("Para o KNN, o recall da classe 1 permaneceu idêntico e o F1 Macro teve variação de +0,0001. Na Árvore de Decisão, a acurácia variou em -0,30 ponto percentual e o F1 Macro em -0,0040.")
    add_p("Como as alterações da Variação B atingem apenas 399 clientes (1,33% da base) e as diferenças observadas estão dentro da margem de variação entre os folds (desvios de 0,004 a 0,008), não é possível afirmar superioridade estatística de uma abordagem sobre a outra sem a realização de testes formais de hipótese. Na prática, a consolidação simplifica o esquema de dados sem prejudicar o desempenho de forma substancial.")

    # 7 DISCUSSÃO DOS RESULTADOS
    add(create_page_break())
    add(create_heading_1("7 DISCUSSÃO DOS RESULTADOS"))
    add_p("Os resultados revelam aspectos importantes do aprendizado supervisionado em problemas de crédito:")
    add_p("1. Insuficiência da Acurácia Isolada: O classificador de referência Dummy obtém 77,88% de acurácia prevendo apenas que todos são adimplentes. Apesar do valor percentual relativamente alto, o modelo não identifica nenhum caso de inadimplência (recall e F1 nulos para a Classe 1). Isso reforça que, em problemas com classes desbalanceadas, a acurácia isolada é insuficiente, sendo indispensável analisar o F1-Score Macro e as métricas específicas por classe.")
    add_p("2. Trade-offs entre Falsos Positivos e Falsos Negativos: Na concessão de crédito, os tipos de erro possuem consequências práticas distintas. Um falso negativo (cliente inadimplente classificado como adimplente) pode resultar em prejuízo com o saldo não quitado e despesas de cobrança. Um falso positivo (cliente em dia classificado como inadimplente) pode causar constrangimento caso o cartão seja bloqueado indevidamente, gerando insatisfação e perda de receita com tarifas. Como este trabalho não utilizou uma matriz de custos financeiros reais, não é possível decretar qual erro acarreta maior prejuízo econômico; nota-se, contudo, que o Random Forest adota uma postura mais conservadora (poucos falsos alarmes), enquanto a Árvore de Decisão prioriza a detecção de inadimplentes com muitos alarmes falsos.")
    add_p("3. Risco Teórico de Sobreajuste na Árvore de Decisão: A Árvore de Decisão identificou mais clientes inadimplentes (2.723), mas teve a menor acurácia e precisão. Por ter sido avaliada sem restrição de profundidade máxima ('max_depth=None') e com divisões mínimas padrão, a árvore tende teoricamente a criar grupos muito específicos para acomodar exemplos isolados do conjunto de treino. Embora este protocolo experimental não tenha registrado as métricas no treino para quantificar a discrepância diretamente, a teoria do modelo e a menor acurácia no teste (72,43%) sustentam o risco de sobreajuste aos dados locais.")
    add_p("4. Comportamento do Random Forest: A técnica de Random Forest utiliza o ensacamento de 100 árvores com seleção aleatória de preditores, o que atenua as decisões isoladas de cada árvore e estabiliza as fronteiras de decisão, resultando em melhor capacidade discriminante (ROC-AUC de 0,7635).")
    add_p("5. Limitações Metodológicas: Os resultados estão restritos ao protocolo fixado, sem otimização de hiperparâmetros (como profundidade máxima da árvore ou número de vizinhos no KNN). Além disso, a base reflete o contexto de clientes de Taiwan em 2005, exigindo cautela antes de extrapolar as conclusões para outros cenários ou períodos.")

    # 8 COMPARAÇÃO COM O ESTUDO ORIGINAL
    add(create_heading_1("8 COMPARAÇÃO COM O ESTUDO ORIGINAL"))
    add_p("O artigo original de Yeh e Lien (2009) utilizou esta mesma base de dados, mas adotou um objetivo metodológico diferente: os autores buscaram estimar a probabilidade contínua de inadimplência por meio do método Sorting Smoothing Method (SSM). Esse procedimento ordena as previsões de probabilidade dos modelos e calcula médias móveis para ajustar uma reta de regressão linear Y = A + B * X.")
    add_p("No trabalho de 2009, foram avaliadas seis técnicas: Redes Neurais Artificiais (ANN), Árvores de Classificação e Regressão (CART), K-Vizinhos Mais Próximos (KNN com k=1), Regressão Logística, Análise Discriminante e Naive Bayes. O artigo original não incluiu o algoritmo Random Forest em sua comparação.")
    add_p("Os autores concluíram que as Redes Neurais alcançaram a melhor calibração de probabilidade contínua pelo método SSM, enquanto CART e KNN apresentaram resultados inferiores nesse critério. É importante esclarecer que o coeficiente de determinação R^2 do SSM avalia a aderência da reta de regressão das probabilidades suavizadas, não sendo comparável nem equivalente à Acurácia, à Área sob a Curva ROC (AUC) ou ao F1-Score da classificação binária.")
    add_p("Devido à limitação de acesso ao texto integral original indexado sob paywall da editora, os números internos detalhados de cada técnica não puderam ser verificados de forma independente. Todavia, a comparação metodológica confirma pontos de concordância: ambos os estudos constatam que variáveis de regularidade de pagamento recente guardam forte associação com a inadimplência e que modelos não lineares conseguem extrair padrões relevantes desses dados.")

    # 9 CONCLUSÃO
    add(create_heading_1("9 CONCLUSÃO"))
    add_p("Este trabalho realizou a caracterização dos dados e a comparação experimental de três algoritmos supervisionados e um classificador de referência para previsão de inadimplência em cartões de crédito:")
    add_p("- Melhor algoritmo no desempenho geral: Random Forest, que alcançou o maior F1-Score Macro (0,6823 na Variação A e 0,6780 na Variação B) e a maior Acurácia (81,75% e 81,57%).")
    add_p("- Pior algoritmo entre os três classificadores principais: Árvore de Decisão sem poda, com F1-Score Macro de 0,6091 e Acurácia de 72,43%, ficando abaixo da acurácia do classificador trivial Dummy (77,88%).")
    add_p("- Classe com melhor desempenho: Classe 0 (clientes adimplentes), com F1-Score de 0,8895 no Random Forest e 0,8726 no KNN.")
    add_p("- Classe com pior desempenho: Classe 1 (clientes inadimplentes), com F1-Score variando entre 0,3970 (Árvore), 0,4273 (KNN) e 0,4751 (Random Forest).")
    add_p("- Algoritmo que mais identificou inadimplentes: Árvore de Decisão, com maior recall na Classe 1 (41,03%, identificando 2.723 devedores), porém com alta taxa de falsos positivos (4.358 alarmes falsos).")
    add_p("- Algoritmo com melhor equilíbrio geral: Random Forest, que obteve a maior precisão na Classe 1 (65,28%, identificando 2.478 devedores com 1.318 falsos alarmes).")
    add_p("- Algoritmo mais rápido no treinamento: KNN, com tempo médio de 0,008 segundo por fold.")
    add_p("- Algoritmo mais rápido na predição: Árvore de Decisão, com latência de 0,56 milissegundo por 1.000 amostras.")
    add_p("- Melhor combinação observada: Random Forest associado à Variação A de pré-processamento.")
    add_p("- Impacto do pré-processamento: A consolidação de categorias na Variação B afetou apenas 399 clientes únicos (1,33% da base), com impacto inferior a 0,45 ponto percentual nas métricas, não havendo comprovação de diferença estatística.")
    add_p("- Limitações do estudo: Utilização de parâmetros padrão sem busca de hiperparâmetros, dados restritos a Taiwan em 2005 e ausência de informações de custos financeiros reais para balancear perdas de crédito e atrito com clientes.")

    # REFERÊNCIAS
    add(create_page_break())
    add(create_heading_1("REFERÊNCIAS"))
    refs = [
        "BREIMAN, Leo. Random Forests. Machine Learning, v. 45, n. 1, p. 5–32, 2001. DOI: 10.1023/A:1010933404324.",
        "BREIMAN, Leo; FRIEDMAN, Jerome H.; OLSHEN, Richard A.; STONE, Charles J. Classification and Regression Trees. Belmont: Wadsworth International Group, 1984.",
        "COVER, Thomas; HART, Peter. Nearest neighbor pattern classification. IEEE Transactions on Information Theory, v. 13, n. 1, p. 21–27, 1967. DOI: 10.1109/TIT.1967.1053964.",
        "PEDREGOSA, Fabian et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, v. 12, p. 2825–2830, 2011.",
        "YEH, I-Cheng. Default of credit card clients. UCI Machine Learning Repository, 2016. DOI: 10.24432/C55S3H. Disponível em: <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>. Acesso em: 19 nov. 2026.",
        "YEH, I-Cheng; LIEN, Che-hui. The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, v. 36, n. 2, p. 2473–2480, 2009. DOI: 10.1016/j.eswa.2007.12.020."
    ]
    for r in refs:
        p = w_elem("p")
        p.append(make_pPr(jc="both", line=240, first_line=0, before=60, after=60))
        p.append(make_r(r, size=22, font="Arial"))
        add(p)

    # REPOSITÓRIO DO PROJETO (discreto e sem referências internas)
    add(create_heading_1("REPOSITÓRIO DO PROJETO"))
    add_p("Os códigos-fonte, dados e resultados experimentais estarão disponíveis no repositório do projeto.")
    
    p_repo = w_elem("p")
    p_repo.append(make_pPr(jc="left", line=280, first_line=0, before=100, after=100))
    p_repo.append(make_r("Disponível em: [LINK DO GITHUB A INSERIR].", bold=True, size=22))
    add(p_repo)

    # Adicionar o sectPr final da Seção 2
    body.append(final_sectPr)

    # Serializar os elementos internos do body sem o invólucro do ElementTree
    body_xml_fragments = []
    for child in body:
        body_xml_fragments.append(ET.tostring(child, encoding="utf-8").decode("utf-8"))
    
    new_doc_xml = doc_header + "".join(body_xml_fragments) + "</w:body></w:document>"

    with open(doc_xml_path, "w", encoding="utf-8") as f:
        f.write(new_doc_xml)

    # Gerar o arquivo DOCX final
    out_docx_path = "docs/Relatorio-Atividade-Avaliativa-3-IA.docx"
    with zipfile.ZipFile(out_docx_path, "w", zipfile.ZIP_DEFLATED) as out_zip:
        for root_d, dirs, files in os.walk(tmp_dir):
            for file in files:
                full_path = os.path.join(root_d, file)
                rel_path = os.path.relpath(full_path, tmp_dir)
                out_zip.write(full_path, rel_path)

    print(f"Relatório DOCX revisado gerado com sucesso em: {out_docx_path}")
    print(f"Tamanho do arquivo: {os.path.getsize(out_docx_path) / 1024:.2f} KB")


if __name__ == "__main__":
    build_full_report()
