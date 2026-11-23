"""
Interface (Streamlit) para medição de zebrafish e detecção/medição de Daphnia.

Coloque este arquivo na mesma pasta de: medir_zebrafish.py, pipeline.py, detectores.py, filtros.py
Rodar:
    python -m streamlit run app_completo.py
"""
import argparse
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import medir_zebrafish

# ---------------------------------------------------------------------
# CONFIGURAÇÕES FIXAS (o usuário do site não altera; mude aqui se precisar)
# Faixa de plausibilidade: medida / diagonal da caixa detectada
PROPORCAO_MIN = 0.5
PROPORCAO_MAX = 1.0
# ---------------------------------------------------------------------

st.set_page_config(page_title="Medição de organismos", layout="wide")
st.title("Medição de zebrafish e Daphnia")


# =====================================================================
# ABA 1: ZEBRAFISH
# =====================================================================
def aba_zebrafish():
    with st.expander("Parâmetros", expanded=True):
        px_por_mm = st.number_input(
            "Pixels por mm (0 = não converter)", min_value=0.0, value=0.0, step=0.1,
            key="zf_px_mm",
        )
        c1, c2, c3 = st.columns(3)
        fator_fraco = c1.slider(
            "Fator do limiar fraco", 0.20, 1.00, float(medir_zebrafish.FATOR_FRACO), 0.05,
            help="Menor = captura mais cauda, mas entra mais ruído. Maior = mais conservador.",
        )
        abertura = c2.slider(
            "Abertura (remoção de riscos)", 0.002, 0.030, float(medir_zebrafish.ABERTURA_FRAC), 0.001,
            format="%.3f", help="Maior = remove riscos mais grossos da placa.",
        )
        reconstrucao = c3.slider(
            "Reconstrução da cauda", 0.00, 0.10, float(medir_zebrafish.RECONSTRUCAO_FRAC), 0.01,
            help="Maior = recupera mais a ponta da cauda, mas pode trazer pedaços de risco.",
        )

    # passa os valores da interface para o script original
    medir_zebrafish.FATOR_FRACO = fator_fraco
    medir_zebrafish.ABERTURA_FRAC = abertura
    medir_zebrafish.RECONSTRUCAO_FRAC = reconstrucao

    arquivos = st.file_uploader(
        "Envie as imagens de zebrafish", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True, key="zf_upload",
    )
    if not arquivos:
        st.info("Envie uma ou mais imagens para começar.")
        return

    linhas = ["arquivo,comprimento_px,comprimento_mm"]
    for arq in arquivos:
        st.subheader(arq.name)
        img = cv2.imdecode(np.frombuffer(arq.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.warning("Não foi possível ler esta imagem.")
            continue

        etapas = medir_zebrafish.Etapas(Path(tempfile.gettempdir()), ativo=False)
        resultado, erro = medir_zebrafish.processar_imagem(img, etapas)
        if resultado is None:
            st.error(f"Falhou: {erro}")
            linhas.append(f"{arq.name},,")
            continue

        comprimento_px, img_final = resultado
        mm = f"{comprimento_px / px_por_mm:.3f}" if px_por_mm > 0 else ""
        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")
        col2.image(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB), caption="Medido")
        st.markdown(f"**{comprimento_px:.1f} px**" + (f"  ({mm} mm)" if mm else ""))
        linhas.append(f"{arq.name},{comprimento_px:.2f},{mm}")

    st.download_button("Baixar medidas (CSV)", "\n".join(linhas),
                       file_name="medidas_zebrafish.csv", mime="text/csv")


# =====================================================================
# ABA 2: DAPHNIA
# =====================================================================
def aba_daphnia():
    # Importa aqui para que a aba do zebrafish funcione mesmo sem torch instalado
    try:
        import pipeline
    except Exception as e:
        st.error("Não consegui carregar o pipeline da Daphnia. "
                 "Provavelmente falta instalar alguma dependência (torch, ultralytics...).")
        st.code(str(e))
        return

    st.markdown("**1. Lente e calibração das imagens**")
    lente_col, escala_col = st.columns(2)
    aumento = lente_col.selectbox(
        "Aumento da objetiva utilizada",
        [4, 10, 20, 40, 60, 100], index=1,
        format_func=lambda valor: f"{valor}×",
        help="Selecione a objetiva utilizada na aquisição. O aumento não converte "
             "pixels em milímetros sem uma calibração da imagem.",
        key="dp_aumento",
    )
    px_por_mm = escala_col.number_input(
        "Calibração (pixels por mm)", min_value=0.0, value=0.0,
        step=1.0, format="%.4f", key="dp_px_por_mm",
        help="Informe o número de pixels correspondente a 1 mm para essa "
             "objetiva e configuração de captura. Use 0 para deixar o CSV apenas em pixels.",
    )
    if px_por_mm > 0:
        st.caption(f"Conversão ativa: 1 px = {1 / px_por_mm:.6f} mm. "
                   "As medidas originais em pixels também serão preservadas.")
    else:
        st.info("Sem calibração: as medidas em milímetros ficarão vazias no CSV. "
                "O aumento selecionado será registrado para rastreabilidade.")

    st.markdown("**2. Modelo de detecção**")
    c1, c2 = st.columns(2)
    model_type = c1.selectbox("Modelo", ["yolo", "fasterrcnn"],
                              format_func=lambda m: {"yolo": "YOLO",
                                                     "fasterrcnn": "Faster R-CNN"}[m])
    checkpoint = c2.text_input("Caminho do checkpoint (.pt / .pth)",
                               help="Arquivo de pesos treinado. Cole o caminho completo.")

    c3, c4 = st.columns(2)
    score = c3.slider("Limiar de confiança da detecção", 0.05, 0.95, 0.50, 0.05)
    escala_recorte = c4.slider("Fator de escala do recorte", 1.0, 2.5, 1.6, 0.1,
                               help="Quanto aumentar a caixa detectada antes de recortar.")

    st.markdown("**3. Cascata de filtros**")
    filtros_sel = st.multiselect(
        "Filtros (aplicados na ordem em que você escolher)",
        ["sem_filtro", "median", "bilateral", "realesrgan"],
        default=["sem_filtro"],
        help="Cada recorte passa pelo primeiro filtro; os que não derem medida plausível "
             "seguem para o próximo.",
    )
    median_kernel = 3
    weights = "weights/RealESRGAN_x4plus.pth"
    if "median" in filtros_sel:
        median_kernel = st.select_slider("Tamanho do kernel do median", [3, 5, 7, 9], value=3)
    if "realesrgan" in filtros_sel:
        weights = st.text_input("Caminho dos pesos do Real-ESRGAN", weights)

    n_cpu = os.cpu_count() or 1
    with st.expander("Desempenho (CPU)"):
        p1, p2, p3, p4 = st.columns(4)
        cpus = p1.number_input("Núcleos a usar", 1, n_cpu, n_cpu,
                               help="Total de núcleos que o pipeline pode ocupar.")
        workers_det = p2.number_input("Processos de detecção", 1, 8, min(2, n_cpu),
                                      help="Cada processo carrega uma cópia do modelo (usa RAM).")
        workers_esrgan = p3.number_input("Processos Real-ESRGAN", 1, 8, min(2, n_cpu),
                                         help="Cada processo carrega uma cópia do modelo (usa RAM).")
        ruler_workers = p4.number_input("Lotes do Daphnia Ruler", 1, 8, min(2, n_cpu))

    st.markdown("**4. Imagens e pasta de trabalho**")
    zip_up = st.file_uploader("Zip com as imagens de Daphnia", type=["zip"], key="dp_zip")
    workdir = st.text_input("Pasta de trabalho (onde os resultados serão salvos)",
                            "resultados_daphnia")

    if st.button("Rodar Daphnia", type="primary"):
        ckpt = Path(checkpoint.strip().strip('"'))
        if not ckpt.is_file():
            st.error("Checkpoint não encontrado. Confira o caminho.")
            return
        if zip_up is None:
            st.error("Envie o arquivo zip com as imagens.")
            return
        if not filtros_sel:
            st.error("Escolha pelo menos um filtro (pode ser sem_filtro).")
            return

        wdir = Path(workdir.strip().strip('"'))
        wdir.mkdir(parents=True, exist_ok=True)
        zip_path = wdir / "entrada.zip"
        zip_path.write_bytes(zip_up.getbuffer())

        args = argparse.Namespace(
            model_type=model_type, checkpoint=ckpt, dataset_zip=zip_path, workdir=wdir,
            score_threshold=score, crop_scale_factor=escala_recorte,
            daphruler_python=Path(sys.executable), reusar_recortes=False,
            filtros=filtros_sel, nome_execucao=None,
            min_proporcao=PROPORCAO_MIN, max_proporcao=PROPORCAO_MAX,
            median_kernel=median_kernel,
            bilateral_diameter=7, bilateral_sigma_color=50.0, bilateral_sigma_space=50.0,
            realesrgan_weights=Path(weights), realesrgan_outscale=4.0, realesrgan_tile=0,
            device="cpu", cpus=int(cpus),
            workers_deteccao=int(workers_det), workers_esrgan=int(workers_esrgan),
            ruler_workers=int(ruler_workers),
            aumento_objetiva=int(aumento), px_por_mm=float(px_por_mm),
        )

        log = io.StringIO()
        erro = None
        with st.spinner("Rodando... pode demorar alguns minutos."):
            try:
                with contextlib.redirect_stdout(log):
                    pipeline.executar_pipeline_cascata(args)
            except Exception as e:
                erro = e

        st.session_state["dp_log"] = log.getvalue()
        st.session_state["dp_erro"] = None if erro is None else repr(erro)
        csv_path = wdir / "execucoes" / "+".join(filtros_sel) / "avaliacao_cascata.csv"
        st.session_state["dp_df"] = (
            pd.read_csv(csv_path) if erro is None and csv_path.exists() else None
        )

    # Resultado (fica na tela mesmo depois de clicar em baixar)
    if "dp_log" in st.session_state:
        if st.session_state["dp_erro"]:
            st.error(f"Erro durante a execução: {st.session_state['dp_erro']}")
        df = st.session_state.get("dp_df")
        if df is not None:
            st.success("Concluído!")
            resumo = df["status"].value_counts()
            st.write(f"Recortes: {len(df)} | " +
                     " | ".join(f"{k}: {v}" for k, v in resumo.items()))
            st.dataframe(df, use_container_width=True)
            st.download_button("Baixar resultados (CSV)", df.to_csv(index=False),
                               file_name="avaliacao_cascata.csv", mime="text/csv")
        with st.expander("Log da execução"):
            st.code(st.session_state["dp_log"] or "(vazio)")


# =====================================================================
aba1, aba2 = st.tabs(["Zebrafish", "Daphnia"])
with aba1:
    aba_zebrafish()
with aba2:
    aba_daphnia()