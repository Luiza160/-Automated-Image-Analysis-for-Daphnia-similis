"""
Pipeline da Daphnia (versão nova).

  foto -> detector treinado (YOLO ou Faster R-CNN) -> recorte de cada Daphnia
       -> medição direta do corpo (medicao.py) -> planilha + imagens para conferência

Diferenças em relação ao pipeline antigo:
  * não usa mais o Daphnia Ruler nem o Real-ESRGAN (nem a "cascata" de filtros);
  * toda Daphnia detectada recebe uma medida; as duvidosas saem marcadas "conferir";
  * ~0,2 s por foto num computador comum, sem placa de vídeo.
"""
import csv
import math
import re
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

import detectores
import medicao

try:  # esconde os avisos "Unknown field with tag" que o OpenCV mostra para os TIFF do ZEN
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
COLUNAS = ["recorte", "imagem", "daphnia", "confianca_deteccao", "x1", "y1", "x2", "y2",
           "comprimento_px", "comprimento_mm", "conferir", "motivo",
           "ponto1_x", "ponto1_y", "ponto2_x", "ponto2_y"]


# ----------------------------------------------------------------------------- entrada
def ler_imagem(caminho):
    """Lê a imagem como BGR. Funciona com acentos no caminho e com TIFF."""
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    if dados.size == 0:
        return None
    img = cv2.imdecode(dados, cv2.IMREAD_COLOR)
    return img


def preparar_imagens(entrada, pasta_extracao):
    """Aceita um .zip, uma pasta ou uma única imagem. Retorna a lista de imagens, em ordem."""
    entrada = Path(entrada)
    if entrada.is_dir():
        return sorted(p for p in entrada.iterdir() if p.is_file() and p.suffix.lower() in EXTENSOES)
    if entrada.suffix.lower() == ".zip":
        if pasta_extracao.exists():
            shutil.rmtree(pasta_extracao)
        pasta_extracao.mkdir(parents=True)
        with zipfile.ZipFile(entrada) as arq:
            for nome in arq.namelist():
                if nome.lower().endswith(EXTENSOES) and Path(nome).name and "__MACOSX" not in nome:
                    with arq.open(nome) as origem, open(pasta_extracao / Path(nome).name, "wb") as destino:
                        shutil.copyfileobj(origem, destino)
        return sorted(p for p in pasta_extracao.iterdir() if p.suffix.lower() in EXTENSOES)
    if entrada.is_file() and entrada.suffix.lower() in EXTENSOES:
        return [entrada]
    raise ValueError(f"Entrada não reconhecida: {entrada} (use um .zip, uma pasta ou uma imagem)")


# ----------------------------------------------------------------------------- medição
def recortar(img, caixa, fator):
    """Recorta a caixa ampliada por 'fator' (margem de fundo em volta do animal)."""
    x1, y1, x2, y2 = caixa
    w, h = x2 - x1, y2 - y1
    ew, eh = w * (fator - 1) / 2, h * (fator - 1) / 2
    cx1, cy1 = max(0, int(x1 - ew)), max(0, int(y1 - eh))
    cx2, cy2 = min(img.shape[1], int(x2 + ew)), min(img.shape[0], int(y2 + eh))
    return img[cy1:cy2, cx1:cx2], (cx1, cy1)


def medir_imagem(modelo, img_bgr, score_threshold=0.5, crop_scale_factor=1.6,
                 pasta_etapas=None, nome_imagem="imagem"):
    """Detecta e mede todas as Daphnias de uma foto (da esquerda para a direita)."""
    deteccoes = sorted(detectores.predizer(modelo, img_bgr, score_threshold), key=lambda d: d[0][0])
    saida = []
    for n, (caixa, score) in enumerate(deteccoes, 1):
        recorte, (ox, oy) = recortar(img_bgr, caixa, crop_scale_factor)
        diag = math.hypot(caixa[2] - caixa[0], caixa[3] - caixa[1])
        id_seguro = re.sub(r"[^A-Za-z0-9_-]+", "_", nome_imagem).strip("_")[:24]
        destino_etapas = (Path(pasta_etapas) / f"{id_seguro}_d{n:02d}"
                          if pasta_etapas is not None else None)
        res = medicao.medir_recorte(recorte, diag, destino_etapas)
        p1 = p2 = None
        if res["p1"] is not None:
            p1 = (res["p1"][0] + ox, res["p1"][1] + oy)
            p2 = (res["p2"][0] + ox, res["p2"][1] + oy)
        saida.append(dict(caixa=caixa, score=score, recorte=recorte, res=res, p1=p1, p2=p2))
    return saida


# ----------------------------------------------------------------------------- desenho
def desenhar_foto(img_bgr, daphnias, px_por_mm=None):
    """Foto inteira com caixa, linha medida e rótulo de cada Daphnia (verde = ok, laranja = conferir)."""
    img = img_bgr.copy()
    lado = max(img.shape[:2])
    esp = max(2, lado // 600)
    fonte = max(0.6, lado / 1800)
    for n, d in enumerate(daphnias, 1):
        cor = (0, 140, 255) if d["res"]["conferir"] else (60, 150, 0)
        x1, y1, x2, y2 = (int(round(v)) for v in d["caixa"])
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, esp)
        L = d["res"]["comprimento_px"]
        if d["p1"] is not None:
            cv2.line(img, tuple(int(v) for v in d["p1"]), tuple(int(v) for v in d["p2"]),
                     (0, 0, 230), esp, cv2.LINE_AA)
        if L is None:
            texto = f"{n}: sem medida"
        elif px_por_mm:
            texto = f"{n}: {L / px_por_mm:.3f} mm"
        else:
            texto = f"{n}: {L:.0f} px"
        if d["res"]["conferir"]:
            texto += " (conferir)"
        (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, fonte, max(1, esp // 2))
        yt = y1 - 8 if y1 - th - 14 > 0 else min(img.shape[0] - base - 2, y2 + th + 10)
        cv2.rectangle(img, (x1, yt - th - 6), (x1 + tw + 10, yt + base), cor, -1)
        cv2.putText(img, texto, (x1 + 5, yt - 2), cv2.FONT_HERSHEY_SIMPLEX, fonte,
                    (255, 255, 255), max(1, esp // 2), cv2.LINE_AA)
    return img


def salvar_jpg(caminho, img, lado_max=None):
    if lado_max and max(img.shape[:2]) > lado_max:
        f = lado_max / max(img.shape[:2])
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        Path(caminho).write_bytes(buf.tobytes())


def _linha_csv(nome_recorte, imagem, n, d, px_por_mm):
    x1, y1, x2, y2 = d["caixa"]
    L = d["res"]["comprimento_px"]
    r = lambda v: "" if v is None else round(float(v), 1)
    return {
        "recorte": nome_recorte, "imagem": imagem, "daphnia": n,
        "confianca_deteccao": round(d["score"], 3),
        "x1": r(x1), "y1": r(y1), "x2": r(x2), "y2": r(y2),
        "comprimento_px": "" if L is None else round(L, 2),
        "comprimento_mm": "" if (L is None or not px_por_mm) else round(L / px_por_mm, 4),
        "conferir": "sim" if d["res"]["conferir"] else "não",
        "motivo": d["res"]["motivo"],
        "ponto1_x": r(d["p1"][0]) if d["p1"] else "", "ponto1_y": r(d["p1"][1]) if d["p1"] else "",
        "ponto2_x": r(d["p2"][0]) if d["p2"] else "", "ponto2_y": r(d["p2"][1]) if d["p2"] else "",
    }


# ----------------------------------------------------------------------------- principal
def executar_pipeline(args, ao_progredir=None):
    """
    args: objeto com model_type, checkpoint, entrada (zip/pasta/imagem), workdir e, opcionalmente,
          score_threshold (0.5), crop_scale_factor (1.6), px_por_mm (None), salvar_imagens (True).
    ao_progredir(fração, mensagem): opcional, para barras de progresso.
    Retorna dict com 'linhas' (uma por Daphnia, como na planilha), 'por_imagem' (para desenhar),
    'imagens', 'csv' e 'workdir'.
    """
    t0 = time.perf_counter()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    limiar = getattr(args, "score_threshold", 0.5)
    fator = getattr(args, "crop_scale_factor", 1.6)
    px_por_mm = getattr(args, "px_por_mm", None) or None
    salvar = getattr(args, "salvar_imagens", True)

    imagens = preparar_imagens(args.entrada, workdir / "imagens_extraidas")
    print(f"=== {len(imagens)} imagem(ns) encontrada(s) ===")
    if ao_progredir:
        ao_progredir(0.0, "Carregando o detector")
    modelo = detectores.carregar_modelo(args.model_type, args.checkpoint)

    pasta_rec = workdir / "recortes_anotados"
    pasta_fotos = workdir / "fotos_anotadas"
    pasta_etapas = workdir / "etapas_processamento" if getattr(args, "salvar_etapas", False) else None
    if salvar:
        for p in (pasta_rec, pasta_fotos):
            shutil.rmtree(p, ignore_errors=True)
            p.mkdir(parents=True, exist_ok=True)

    linhas, por_imagem, id_recorte = [], [], 0
    for k, caminho in enumerate(imagens):
        if ao_progredir:
            ao_progredir(k / max(1, len(imagens)), f"Medindo {caminho.name} ({k + 1}/{len(imagens)})")
        img = ler_imagem(caminho)
        if img is None:
            print(f"  [aviso] não foi possível ler {caminho.name}")
            continue
        daphnias = medir_imagem(modelo, img, limiar, fator, pasta_etapas, caminho.stem)
        for n, d in enumerate(daphnias, 1):
            nome = f"{id_recorte:04d}_{caminho.stem}_d{n}"
            linhas.append(_linha_csv(nome, caminho.name, n, d, px_por_mm))
            if salvar:
                salvar_jpg(pasta_rec / f"{nome}.jpg", medicao.desenhar_recorte(d["recorte"], d["res"]))
            id_recorte += 1
        if salvar:
            salvar_jpg(pasta_fotos / f"{caminho.stem}.jpg", desenhar_foto(img, daphnias, px_por_mm), 1800)
        # versão leve (sem recorte nem máscara) para as interfaces desenharem o resultado
        por_imagem.append({"imagem": caminho, "daphnias": [
            {"caixa": d["caixa"], "score": d["score"], "p1": d["p1"], "p2": d["p2"],
             "res": {k: v for k, v in d["res"].items() if k != "mascara"}} for d in daphnias]})
        n_conf = sum(d["res"]["conferir"] for d in daphnias)
        print(f"  {caminho.name}: {len(daphnias)} Daphnia(s)" + (f", {n_conf} para conferir" if n_conf else ""))

    csv_saida = workdir / "medidas_daphnia.csv"
    with open(csv_saida, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        w.writerows(linhas)
    if ao_progredir:
        ao_progredir(1.0, "Concluído")

    n_conf = sum(l["conferir"] == "sim" for l in linhas)
    print(f"\n=== RESUMO ===\n  Daphnias medidas: {len(linhas)}  |  para conferir: {n_conf}"
          f"\n  Tempo total: {time.perf_counter() - t0:.1f} s\n  Planilha: {csv_saida}")
    return {"linhas": linhas, "por_imagem": por_imagem, "imagens": imagens, "csv": csv_saida,
            "workdir": workdir}
