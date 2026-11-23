"""Comparison run: YOLO detects/crops each Daphnia, then Daphnia Ruler measures it.

Unlike the production pipeline, this script uses an exact detector box crop
(crop factor 1.0) and writes all comparison outputs to a separate work folder.
"""
import argparse
import csv
import tempfile
from pathlib import Path

import cv2
import numpy as np

import detectores
import main
import pipeline


def salvar_png(path, image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Não foi possível codificar a imagem: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes())


def salvar_etapa(pasta, nome, image):
    """Save a diagnostic array as a readable PNG."""
    if pasta is None:
        return
    a = np.asarray(image)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    elif a.dtype != np.uint8:
        a = np.nan_to_num(a.astype(np.float32))
        if a.ndim == 2:
            a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    salvar_png(pasta / f"{nome}.png", a)


def medir_com_ruler(path, pasta_etapas=None):
    """Instrumented copy of Daphnia Ruler's default head_method."""
    from daphruler import utils
    from scipy import ndimage
    from skimage import measure, morphology
    from skimage.filters import scharr

    loaded = utils.import_image(str(path))
    img, gray = loaded["img"], loaded["gray"]
    salvar_etapa(pasta_etapas, "02_ruler_redimensionada_720px", img)
    salvar_etapa(pasta_etapas, "03_tons_de_cinza", gray)

    # daphruler.utils.create_mask(), with an image saved after each operation.
    grad = scharr(gray)
    mediana = np.median(grad)
    limiar = 2.5 * mediana
    salvar_etapa(pasta_etapas, "04_gradiente_scharr", grad)
    edges = grad >= limiar
    salvar_etapa(pasta_etapas, "05_limiar_2_5x_mediana", edges)
    edges = morphology.closing(edges, morphology.square(3))
    salvar_etapa(pasta_etapas, "06_fechamento_3x3", edges)
    edges = ndimage.binary_fill_holes(edges)
    salvar_etapa(pasta_etapas, "07_buracos_preenchidos", edges)
    edges = morphology.erosion(edges, morphology.square(3))
    salvar_etapa(pasta_etapas, "08_erosao_3x3", edges)

    props = utils.create_props(edges, gray)
    if not props:
        raise ValueError("nenhuma região conectada após a máscara inicial")

    # Instrument daphruler.utils.erode_mask() and save each opening iteration.
    edges_u8 = np.uint8(edges)
    edges_res = edges_u8
    kernel_size = 2
    label_img = morphology.label(edges, connectivity=2, background=0)
    iteracao = 0
    while props[0].solidity < 0.93:
        iteracao += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        edges_res = cv2.morphologyEx(edges_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        label_img = morphology.label(edges_res, connectivity=2, background=0)
        props = measure.regionprops(label_img, gray)
        props = sorted(props, reverse=True, key=lambda region: region.area)
        if not props:
            raise ValueError("as aberturas eliminaram todas as regiões")
        salvar_etapa(pasta_etapas,
                     f"09_abertura_{iteracao:02d}_kernel_{kernel_size:02d}_solidez_{props[0].solidity:.3f}",
                     label_img == props[0].label)
        kernel_size += 1

    salvar_etapa(pasta_etapas, "10_regiao_final_selecionada", label_img == props[0].label)
    binary = utils.plt_binary(edges_res, label_img, props)
    anotada = img.copy()
    anotada = utils.plt_contour(binary, anotada)
    salvar_etapa(pasta_etapas, "11_contorno", anotada.copy())
    anotada = utils.plt_elipse(anotada, props)
    salvar_etapa(pasta_etapas, "12_elipse_ajustada", anotada.copy())
    anotada = utils.plt_majaxis(anotada, props)
    salvar_etapa(pasta_etapas, "13_eixo_maior", anotada.copy())
    anotada = utils.plt_minaxis(anotada, props)
    resultado = utils.make_res(anotada, props, loaded["scf"], str(path))
    salvar_etapa(pasta_etapas, "14_elipse_eixos_resultado", resultado["image"])
    return resultado


def parse_args():
    p = argparse.ArgumentParser(
        description="Compara YOLO + recorte exato + método padrão do Daphnia Ruler."
    )
    p.add_argument("--entrada", type=Path, required=True,
                   help="pasta, ZIP ou imagem de entrada")
    p.add_argument("--workdir", type=Path, required=True,
                   help="pasta separada para os resultados deste teste")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="pesos YOLO; padrão: yolo11n_best.pt")
    p.add_argument("--score-threshold", type=float, default=0.5,
                   help="confiança mínima do YOLO (padrão: 0.5)")
    p.add_argument("--salvar-etapas", action="store_true",
                   help="salva imagens intermediárias do Daphnia Ruler para cada recorte")
    return p.parse_args()


def main_run():
    args = parse_args()
    checkpoint = args.checkpoint or main.achar_pesos("yolo11n_best.pt")
    modelo = detectores.carregar_modelo("yolo", checkpoint)
    args.workdir.mkdir(parents=True, exist_ok=True)
    pasta_recortes = args.workdir / "recortes_ruler"
    pasta_recortes.mkdir(parents=True, exist_ok=True)
    pasta_etapas = args.workdir / "etapas_ruler" if args.salvar_etapas else None
    imagens = pipeline.preparar_imagens(args.entrada, args.workdir / "imagens_extraidas")
    print(f"=== {len(imagens)} imagem(ns); YOLO + corte exato + Daphnia Ruler ===")

    try:
        import daphruler
    except ImportError as exc:
        raise SystemExit(
            "Daphnia Ruler não está instalado neste Python. Instale o pacote daphruler."
        ) from exc

    linhas = []
    # O Ruler usa cv2.imread, que falha em caminhos com acentos no Windows.
    # Os recortes temporários têm nomes ASCII em uma pasta temporária simples.
    with tempfile.TemporaryDirectory(prefix="daphnia_ruler_") as tmp:
        tmp = Path(tmp)
        for image_index, image_path in enumerate(imagens, 1):
            img = pipeline.ler_imagem(image_path)
            if img is None:
                print(f"[aviso] não foi possível ler {image_path.name}")
                continue
            detections = sorted(
                detectores.predizer(modelo, img, args.score_threshold),
                key=lambda d: d[0][0],
            )
            for n, (box, score) in enumerate(detections, 1):
                crop, _ = pipeline.recortar(img, box, 1.0)  # sem margem de 1,6x
                crop_path = tmp / f"crop_{image_index:04d}_{n:03d}.png"
                salvar_png(crop_path, crop)
                pasta_etapa_daphnia = (pasta_etapas / f"{image_path.stem}_d{n:02d}"
                                       if pasta_etapas is not None else None)
                salvar_etapa(pasta_etapa_daphnia, "01_recorte_exato_yolo", crop)
                out = {
                    "imagem": image_path.name,
                    "daphnia": n,
                    "confianca_deteccao": round(float(score), 4),
                    "x1": round(float(box[0]), 1), "y1": round(float(box[1]), 1),
                    "x2": round(float(box[2]), 1), "y2": round(float(box[3]), 1),
                    "comprimento_px": "", "solidez": "", "erro": "",
                }
                try:
                    ruler = medir_com_ruler(crop_path, pasta_etapa_daphnia)
                    out["comprimento_px"] = round(float(ruler["full.Length"]), 2)
                    out["solidez"] = round(float(ruler["solidity"]), 4)
                    nome = f"{image_path.stem}_d{n:02d}_ruler.png"
                    anotada = ruler["image"].copy()
                    texto = f"Medida: {float(ruler['full.Length']):.2f} px"
                    cv2.rectangle(anotada, (4, 4), (270, 31), (255, 255, 255), -1)
                    cv2.putText(anotada, texto, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 2, cv2.LINE_AA)
                    salvar_png(pasta_recortes / nome, anotada)
                except Exception as exc:
                    out["erro"] = f"{type(exc).__name__}: {exc}"
                    print(f"[aviso] {image_path.name}, d{n}: {out['erro']}")
                linhas.append(out)
            print(f"  {image_path.name}: {len(detections)} detecção(ões)")

    csv_path = args.workdir / "medidas_ruler.csv"
    cols = ["imagem", "daphnia", "confianca_deteccao", "x1", "y1", "x2", "y2",
            "comprimento_px", "solidez", "erro"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(linhas)
    print(f"\nCSV: {csv_path}\nRecortes anotados: {pasta_recortes}")


if __name__ == "__main__":
    main_run()
