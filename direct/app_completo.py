"""
Medição de zebrafish e Daphnia (interface Streamlit).

Arquivos necessários na mesma pasta: este arquivo, zebrafish.py (zebrafish), pipeline.py,
detectores.py, medicao.py (Daphnia) e os pesos do detector (yolo11n_best.pt e/ou
fasterrcnn_best.pth; também são procurados na pasta de cima).
Não precisa mais do Daphnia Ruler nem do RealESRGAN_x4plus.pth.

Rodar:
    python -m streamlit run app_completo.py
"""
import argparse
import contextlib
import io
import math
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import medir_zebrafish as zebrafish

BASE = Path(__file__).resolve().parent

# =====================================================================
# CONFIGURAÇÕES FIXAS (o pesquisador não altera nada disso)
# =====================================================================
# --- Daphnia ---
MARGEM_RECORTE = 1.6           # quanto a caixa detectada é ampliada antes de recortar
PX_POR_MM_DAPHNIA = None       # escala padrão sugerida na tela (ex.: 145.0); None = campo vazio

# --- Zebrafish ---
PX_POR_MM = None               # ex.: 85.3 se todas as fotos tiverem a mesma escala; None = só pixels

# --- Tela de resultados ---
POR_PAGINA = 8                 # imagens mostradas por página

# --- Modelos de detecção da Daphnia ---
# "arquivo" é o nome do arquivo de pesos, procurado na pasta deste app e na pasta de cima.
# "confianca" é a confiança mínima da detecção. Só aparecem os modelos cujo arquivo existe.
MODELOS = {
    "YOLO": {
        "tipo": "yolo",
        "arquivo": "yolo11n_best.pt",
        "confianca": 0.5,
        "resumo": "Rápido: cerca de 0,2 s por foto, sem placa de vídeo.",
        "indicado": "Uso normal. Nos testes encontrou todas as 25 Daphnias das 9 fotos.",
        "atencao": "Se na revisão faltar alguma Daphnia, refaça com o Faster R-CNN.",
    },
    "Faster R-CNN": {
        "tipo": "fasterrcnn",
        "arquivo": "fasterrcnn_best.pth",
        "confianca": 0.8,
        "resumo": "Mais lento: cerca de 6 s por foto sem placa de vídeo.",
        "indicado": "Quando o YOLO deixar Daphnias de fora.",
        "atencao": "Nos testes encontrou exatamente as mesmas Daphnias que o YOLO.",
    },
}
DICA_ESCOLHA = ("Comece pelo YOLO. A medição em si é a mesma para os dois; "
                "só muda o modelo que encontra as Daphnias na foto.")
# =====================================================================

EXT_IMG = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
ss = st.session_state

st.set_page_config(page_title="Medição de zebrafish e Daphnia", layout="wide")

st.markdown(
    """
<style>
:root {
    --tinta: #14212B;
    --suave: #5B6B75;
    --linha: #D5DEE2;
    --fundo: #F5F7F8;
    --acento: #0F5C6E;
    --serifa: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
}
.stApp { background: var(--fundo); color: var(--tinta); }
.block-container { max-width: 1120px; padding-top: 3rem; padding-bottom: 5rem; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

h1 { font-family: var(--serifa); font-weight: 600; font-size: 2.4rem;
     letter-spacing: -0.015em; line-height: 1.15; padding: 0 0 .5rem 0; }
h2, h3 { font-family: var(--serifa); font-weight: 600; letter-spacing: -0.005em; }
h2 { font-size: 1.5rem; margin-top: 2rem; }
h3 { font-size: 1.25rem; margin-top: 1.8rem; }
.lead { color: var(--suave); font-size: 1.1rem; line-height: 1.55; max-width: 60ch;
        margin: 0 0 2rem 0; }
.passo { color: var(--suave); max-width: 70ch; line-height: 1.55; margin-bottom: 1rem; }

/* cartões (início e modelos) */
[class*="st-key-cartao"], [class*="st-key-modelo"] {
    background: #fff; border: 1px solid var(--linha); border-radius: 14px;
    padding: 1.7rem 1.7rem 1.3rem; transition: border-color .15s;
}
[class*="st-key-cartao"]:hover, [class*="st-key-modelo"]:hover { border-color: var(--acento); }
[class*="st-key-modelosel"] { border-color: var(--acento); box-shadow: 0 0 0 1px var(--acento); }
[class*="st-key-cartao"] [data-testid="stButton"],
[class*="st-key-modelo"] [data-testid="stButton"] { width: 100%; }
[class*="st-key-cartao"] button, [class*="st-key-modelo"] button { width: 100%; }
.cartao-titulo { font-family: var(--serifa); font-size: 1.7rem; font-weight: 600; line-height: 1.2; }
.cartao-sub { color: var(--suave); font-style: italic; margin-bottom: .9rem; }
.cartao-texto { line-height: 1.55; margin-bottom: 1.2rem; min-height: 4.6em; }
.rotulo { color: var(--suave); font-size: .9rem; margin-top: .6rem; }

/* itens da galeria de resultados */
[class*="st-key-item"] { background: #fff; border: 1px solid var(--linha);
                         border-radius: 12px; padding: 1.2rem 1.3rem; margin-bottom: 1rem; }
.item-nome { font-weight: 600; }
.item-resumo { color: var(--suave); }

/* envio de arquivos */
[data-testid="stFileUploader"] section {
    background: #fff; border: 1px dashed #AFC0C7; border-radius: 12px; padding: 1.6rem;
}

/* botões e números */
[data-testid="stBaseButton-primary"], .stDownloadButton button {
    border-radius: 8px; padding: .6rem 1.5rem; font-weight: 500;
}
[data-testid="stBaseButton-primary"] { background: var(--acento); border-color: var(--acento); }
[data-testid="stMetric"] { background: #fff; border: 1px solid var(--linha);
                           border-radius: 10px; padding: .9rem 1.1rem; }
[data-testid="stMetricValue"] { font-family: var(--serifa); }
hr { border-color: var(--linha); margin: 2rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

ss.setdefault("pagina", "inicio")
ss.setdefault("resultado", None)
ss.setdefault("descartadas", set())


# =====================================================================
# Utilidades
# =====================================================================
def _slug(texto):
    return re.sub(r"[^a-z0-9]+", "_", texto.lower()).strip("_")


def caminho_pesos(arquivo):
    """Procura o arquivo de pesos na pasta do app e na pasta de cima."""
    for pasta in (BASE.parent / 'weights', BASE):
        if (pasta / arquivo).is_file():
            return pasta / arquivo
    return None


def ler_imagem(caminho):
    """Leitura segura para caminhos com acentos (Windows)."""
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR) if dados.size else None


def salvar_vista(img_bgr, destino, lado_max=1400):
    """Salva uma versão reduzida (JPG) só para exibir na tela."""
    h, w = img_bgr.shape[:2]
    f = min(1.0, lado_max / max(h, w))
    if f < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    Path(destino).write_bytes(buf.tobytes())
    return str(destino)


def limpar_resultado():
    res = ss.get("resultado")
    if res:
        shutil.rmtree(res["pasta"], ignore_errors=True)
    ss["resultado"] = None
    ss["descartadas"] = set()
    ss["revisao"] = False
    ss["pag_res"] = 1


def ir_para(pagina):
    limpar_resultado()
    ss["pagina"] = pagina


def montar_zip(arquivos, destino):
    """Junta as imagens enviadas (e o conteúdo de zips) em um único zip para o pipeline."""
    with zipfile.ZipFile(destino, "w") as z:
        for arq in arquivos:
            dados = arq.getvalue()
            if arq.name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(dados)) as zin:
                    for n in zin.namelist():
                        if n.lower().endswith(EXT_IMG):
                            z.writestr(Path(n).name, zin.read(n))
            else:
                z.writestr(Path(arq.name).name, dados)


# =====================================================================
# Medição: zebrafish
# =====================================================================
def medir_zebrafish(arquivos, ao_progredir=None):
    pasta = Path(tempfile.mkdtemp(prefix="zebrafish_"))
    itens, n_ok = [], 0
    for i, arq in enumerate(arquivos):
        img = cv2.imdecode(np.frombuffer(arq.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        item = {"id": arq.name, "orig": None, "anot": None, "resumo": "", "linhas": []}
        if img is None:
            item["resumo"] = "Não foi possível ler esta imagem."
            item["linhas"] = [{"arquivo": arq.name, "comprimento_px": "", "comprimento_mm": ""}]
        else:
            item["orig"] = salvar_vista(img, pasta / f"{i:04d}_orig.jpg")
            etapas = zebrafish.Etapas(Path(tempfile.gettempdir()), ativo=False)
            resultado, erro = zebrafish.processar_imagem(img, etapas)
            if resultado is None:
                item["resumo"] = f"Não foi possível medir ({erro})."
                item["linhas"] = [{"arquivo": arq.name, "comprimento_px": "", "comprimento_mm": ""}]
            else:
                px, anotada = resultado
                mm = f"{px / PX_POR_MM:.3f}" if PX_POR_MM else ""
                item["anot"] = salvar_vista(anotada, pasta / f"{i:04d}_anot.jpg")
                item["resumo"] = f"{px:.1f} px" + (f" ({mm} mm)" if mm else "")
                item["linhas"] = [{"arquivo": arq.name, "comprimento_px": f"{px:.2f}",
                                   "comprimento_mm": mm}]
                n_ok += 1
        itens.append(item)
        if ao_progredir:
            ao_progredir((i + 1) / len(arquivos))
    colunas = ["arquivo", "comprimento_px", "comprimento_mm"]
    if not PX_POR_MM:
        colunas = colunas[:2]
    return {
        "tipo": "zebrafish", "pasta": str(pasta), "itens": itens, "colunas": colunas,
        "metricas": [("Imagens", len(itens)), ("Peixes medidos", n_ok),
                     ("Sem medida", len(itens) - n_ok)],
        "arquivo_csv": "medidas_zebrafish",
    }


# =====================================================================
# Medição: Daphnia
# =====================================================================
def montar_resultado_daphnia(saida, pasta_vistas, nome_modelo, px_por_mm, pipeline):
    """Transforma a saída do pipeline nos itens da galeria (uma foto por item)."""
    itens, total, total_conf = [], 0, 0
    for i, foto in enumerate(saida["por_imagem"]):
        caminho, daphnias = foto["imagem"], foto["daphnias"]
        img = ler_imagem(caminho)
        item = {"id": caminho.name, "orig": None, "anot": None, "resumo": "", "linhas": []}
        if img is None:
            item["resumo"] = "Não foi possível ler esta imagem."
            itens.append(item)
            continue
        item["orig"] = salvar_vista(img, pasta_vistas / f"{i:04d}_orig.jpg")
        if not daphnias:
            item["resumo"] = "Nenhuma Daphnia detectada."
            item["linhas"] = [{"arquivo": caminho.name, "daphnia": "", "comprimento_px": "",
                               "comprimento_mm": "", "conferir": "", "motivo": "nenhuma detectada"}]
        else:
            item["anot"] = salvar_vista(pipeline.desenhar_foto(img, daphnias, px_por_mm),
                                        pasta_vistas / f"{i:04d}_anot.jpg")
            n_conf = sum(d["res"]["conferir"] for d in daphnias)
            total += len(daphnias)
            total_conf += n_conf
            item["resumo"] = (f"{len(daphnias)} Daphnia(s) medida(s)"
                              + (f", {n_conf} para conferir (em laranja)" if n_conf else ""))
            for n, d in enumerate(daphnias, 1):
                L = d["res"]["comprimento_px"]
                item["linhas"].append({
                    "arquivo": caminho.name, "daphnia": n,
                    "comprimento_px": f"{L:.2f}" if L is not None else "",
                    "comprimento_mm": f"{L / px_por_mm:.4f}" if (L is not None and px_por_mm) else "",
                    "conferir": "sim" if d["res"]["conferir"] else "não",
                    "motivo": d["res"]["motivo"],
                })
        itens.append(item)

    colunas = ["arquivo", "daphnia", "comprimento_px", "comprimento_mm", "conferir", "motivo"]
    if not px_por_mm:
        colunas.remove("comprimento_mm")
    return {
        "tipo": "daphnia", "pasta": str(pasta_vistas), "itens": itens, "modelo": nome_modelo,
        "colunas": colunas,
        "metricas": [("Imagens", len(itens)), ("Daphnias medidas", total),
                     ("Para conferir", total_conf)],
        "arquivo_csv": "medidas_daphnia",
    }


def medir_daphnia(arquivos, nome_modelo, pipeline, px_por_mm=None, ao_progredir=None):
    info = MODELOS[nome_modelo]
    wdir = Path(tempfile.mkdtemp(prefix="daphnia_"))
    pasta_vistas = Path(tempfile.mkdtemp(prefix="daphnia_vistas_"))
    try:
        zip_path = wdir / "entrada.zip"
        montar_zip(arquivos, zip_path)
        args = argparse.Namespace(
            model_type=info["tipo"], checkpoint=caminho_pesos(info["arquivo"]),
            entrada=zip_path, workdir=wdir / "saida",
            score_threshold=info["confianca"], crop_scale_factor=MARGEM_RECORTE,
            px_por_mm=px_por_mm, salvar_imagens=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            saida = pipeline.executar_pipeline(args, ao_progredir)
        return montar_resultado_daphnia(saida, pasta_vistas, nome_modelo, px_por_mm, pipeline)
    except Exception:
        shutil.rmtree(pasta_vistas, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(wdir, ignore_errors=True)


# =====================================================================
# Resultados (iguais para os dois organismos)
# =====================================================================
def _alternar(idx):
    if ss[f"desc_{idx}"]:
        ss["descartadas"].add(idx)
    else:
        ss["descartadas"].discard(idx)


def _revisao_mudou():
    if not ss.get("revisao"):
        ss["descartadas"] = set()


def mostrar_resultado(res):
    itens = res["itens"]
    revisao = ss.get("revisao", False)
    descartadas = ss["descartadas"] if revisao else set()

    st.markdown("## Resultado")
    colunas = st.columns(len(res["metricas"]))
    for col, (rotulo, valor) in zip(colunas, res["metricas"]):
        col.metric(rotulo, valor)
    st.write("")

    # --- download e revisão opcional ---
    incluidas = [it for i, it in enumerate(itens) if i not in descartadas]
    linhas = [linha for it in incluidas for linha in it["linhas"]]
    df = pd.DataFrame(linhas, columns=res["colunas"])
    nome_csv = f"{res['arquivo_csv']}_{date.today().isoformat()}.csv"

    c1, c2 = st.columns([2, 3], vertical_alignment="center")
    c1.download_button("Baixar medidas (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=nome_csv, mime="text/csv", type="primary")
    c2.toggle("Quero revisar e descartar imagens", key="revisao", on_change=_revisao_mudou,
              help="Se alguma medição parecer errada, marque a imagem para tirá-la do CSV.")
    if revisao:
        st.caption(f"{len(itens) - len(descartadas)} de {len(itens)} imagens serão incluídas no CSV. "
                   "Marque \"Descartar\" nas imagens cuja medição você não quer usar.")

    # --- galeria ---
    st.markdown("### Imagens medidas")
    n_pag = max(1, math.ceil(len(itens) / POR_PAGINA))
    if n_pag > 1:
        st.selectbox("Página", range(1, n_pag + 1), key="pag_res",
                     format_func=lambda p: f"Página {p} de {n_pag}")
    pag = min(ss.get("pag_res", 1), n_pag)
    ini = (pag - 1) * POR_PAGINA
    for idx in range(ini, min(len(itens), ini + POR_PAGINA)):
        it = itens[idx]
        with st.container(key=f"item_{idx}"):
            topo, marcar = st.columns([4, 1], vertical_alignment="center")
            marcador = " (descartada)" if idx in descartadas else ""
            topo.markdown(f'<span class="item-nome">{it["id"]}</span>{marcador}<br>'
                          f'<span class="item-resumo">{it["resumo"]}</span>',
                          unsafe_allow_html=True)
            if revisao:
                marcar.checkbox("Descartar", value=idx in descartadas, key=f"desc_{idx}",
                                on_change=_alternar, args=(idx,))
            if it["orig"]:
                a, b = st.columns(2)
                a.image(it["orig"], caption="Original")
                if it["anot"]:
                    b.image(it["anot"], caption="Medições")

    st.write("")
    st.button("Fazer nova medição", on_click=limpar_resultado)


# =====================================================================
# Páginas
# =====================================================================
def cabecalho(titulo, subtitulo_html, lead):
    st.button("Voltar ao início", type="tertiary", on_click=ir_para, args=("inicio",))
    st.title(titulo)
    st.markdown(f'<div class="cartao-sub" style="margin-top:-.4rem">{subtitulo_html}</div>'
                f'<p class="lead">{lead}</p>', unsafe_allow_html=True)


def pagina_inicio():
    st.title("Medição de zebrafish e Daphnia")
    st.markdown('<p class="lead">Escolha o organismo, envie suas fotos e baixe os tamanhos '
                "em uma planilha.</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(key="cartao_zebrafish"):
            st.markdown(
                '<div class="cartao-titulo">Zebrafish</div>'
                '<div class="cartao-sub">Danio rerio</div>'
                '<div class="cartao-texto">Mede o comprimento do peixe a partir de fotos com '
                "um peixe por imagem.</div>", unsafe_allow_html=True)
            st.button("Medir zebrafish", key="ir_zebrafish", on_click=ir_para,
                      args=("zebrafish",))
    with c2:
        with st.container(key="cartao_daphnia"):
            st.markdown(
                '<div class="cartao-titulo">Daphnia</div>'
                '<div class="cartao-sub">Daphnia similis</div>'
                '<div class="cartao-texto">Encontra cada Daphnia nas fotos e mede o comprimento '
                "do corpo. Você escolhe o modelo de detecção.</div>", unsafe_allow_html=True)
            st.button("Medir Daphnia", key="ir_daphnia", on_click=ir_para, args=("daphnia",))


def pagina_zebrafish():
    cabecalho("Zebrafish", "<em>Danio rerio</em>",
              "Envie as fotos, com um peixe por imagem. O app mede o comprimento de cada peixe.")
    res = ss.get("resultado")
    if res and res["tipo"] == "zebrafish":
        mostrar_resultado(res)
        return

    arquivos = st.file_uploader("Imagens de zebrafish", type=[e[1:] for e in EXT_IMG],
                                accept_multiple_files=True, key="zf_upload",
                                label_visibility="collapsed")
    st.write("")
    if st.button("Medir", type="primary", disabled=not arquivos):
        barra = st.progress(0.0, text="Medindo as imagens...")
        ss["resultado"] = medir_zebrafish(arquivos, lambda f: barra.progress(f))
        ss["descartadas"], ss["revisao"], ss["pag_res"] = set(), False, 1
        st.rerun()


def pagina_daphnia():
    cabecalho("Daphnia", "<em>Daphnia similis</em>",
              "Escolha o modelo, envie as fotos e aguarde. Ao final você vê cada Daphnia medida "
              "e baixa os tamanhos em uma planilha.")
    res = ss.get("resultado")
    if res and res["tipo"] == "daphnia":
        st.caption(f"Modelo usado: {res['modelo']}. Caixas em laranja estão marcadas "
                   "\"conferir\": olhe a linha vermelha antes de usar essa medida.")
        mostrar_resultado(res)
        return

    # O pipeline só é importado aqui: assim o zebrafish funciona mesmo sem o torch instalado
    try:
        import pipeline
    except Exception as e:
        st.error("Não foi possível carregar o pipeline da Daphnia. "
                 "Falta instalar alguma dependência (torch, ultralytics...).")
        st.code(str(e))
        return

    modelos_ok = {n: m for n, m in MODELOS.items() if caminho_pesos(m["arquivo"])}
    if not modelos_ok:
        st.error("Nenhum modelo encontrado. Coloque os arquivos de pesos na pasta do app "
                 "(ou na pasta de cima) e confira os nomes na lista MODELOS, no início do "
                 "app_completo.py.")
        return
    if ss.get("modelo") not in modelos_ok:
        ss["modelo"] = next(iter(modelos_ok))

    # --- passo 1: modelo ---
    st.markdown("## Qual modelo usar?")
    st.markdown(f'<p class="passo">Os modelos têm o mesmo objetivo, encontrar as Daphnias na '
                f"foto, mas se comportam de modo diferente. {DICA_ESCOLHA}</p>",
                unsafe_allow_html=True)
    colunas = st.columns(len(modelos_ok), gap="large")
    for col, (nome, m) in zip(colunas, modelos_ok.items()):
        escolhido = ss["modelo"] == nome
        with col:
            with st.container(key=f"{'modelosel' if escolhido else 'modelo'}_{_slug(nome)}"):
                st.markdown(
                    f'<div class="cartao-titulo">{nome}</div>'
                    f'<div class="cartao-sub">{m["resumo"]}</div>'
                    f'<div class="rotulo">Indicado quando</div>'
                    f'<div class="cartao-texto" style="min-height:0">{m["indicado"]}</div>'
                    f'<div class="rotulo">Vale saber</div>'
                    f'<div class="cartao-texto">{m["atencao"]}</div>', unsafe_allow_html=True)
                if st.button("Selecionado" if escolhido else "Usar este modelo",
                             key=f"btn_{_slug(nome)}", disabled=escolhido,
                             type="primary" if escolhido else "secondary"):
                    ss["modelo"] = nome
                    st.rerun()

    # --- passo 2: imagens ---
    st.markdown("## Envie as fotos")
    arquivos = st.file_uploader("Imagens de Daphnia (ou um .zip com as imagens)",
                                type=[e[1:] for e in EXT_IMG] + ["zip"],
                                accept_multiple_files=True, key="dp_upload",
                                label_visibility="collapsed")
    # --- passo 3: escala (opcional) ---
    st.markdown("## Escala (opcional)")
    st.markdown('<p class="passo">Se souber quantos pixels equivalem a 1 mm nestas fotos, informe '
                "aqui para a planilha sair também em mm. Todas as fotos do lote devem ter a mesma "
                "magnificação. Deixe 0 para medir só em pixels.</p>", unsafe_allow_html=True)
    px_por_mm = st.number_input("Pixels por mm", min_value=0.0, step=1.0, format="%.2f",
                                value=float(PX_POR_MM_DAPHNIA or 0.0), key="dp_pxmm") or None

    st.write("")
    if st.button("Medir Daphnias", type="primary", disabled=not arquivos):
        barra = st.progress(0.0, text="Carregando o detector...")
        try:
            ss["resultado"] = medir_daphnia(
                arquivos, ss["modelo"], pipeline, px_por_mm,
                lambda f, msg: barra.progress(min(1.0, f), text=msg))
        except Exception as e:
            st.error("A medição não terminou. Tente novamente; se o erro continuar, "
                     "envie a mensagem abaixo para quem mantém o app.")
            st.code(repr(e))
            return
        ss["descartadas"], ss["revisao"], ss["pag_res"] = set(), False, 1
        st.rerun()


# =====================================================================
{"inicio": pagina_inicio, "zebrafish": pagina_zebrafish, "daphnia": pagina_daphnia}[ss["pagina"]]()
