"""
Compara as medidas automáticas com medidas feitas à mão no ImageJ.

Precisa de:
  --fotos    pasta com as fotos originais (.tif/.png/.jpg)
  --medidas  pasta com recortes salvos pelo ImageJ (.tif) contendo a linha medida (ROI de linha),
             como a pasta "Medidas". Cada recorte é localizado automaticamente na foto de origem.

Uso:
    python validar.py --fotos "../Cont Ap@GO" --medidas "../Cont Ap@GO/Medidas"

Saída: tabela no terminal + validacao.csv. Requer: pip install tifffile
"""
import argparse
import csv
import math
import struct
from pathlib import Path

import cv2
import numpy as np

import detectores
import pipeline

BASE = Path(__file__).resolve().parent


def linha_imagej(caminho):
    """Lê o recorte e a linha (ROI) gravada pelo ImageJ. Retorna (img_cinza, (x1,y1,x2,y2), px_por_mm)."""
    import tifffile
    with tifffile.TiffFile(caminho) as t:
        img = t.pages[0].asarray()
        meta = t.imagej_metadata or {}
        xr = t.pages[0].tags.get("XResolution")
        px_por_mm = (xr.value[0] / xr.value[1]) if (xr and meta.get("unit") == "mm") else None
    roi = meta.get("ROI")
    if not roi or roi[:4] != b"Iout" or roi[6] != 3:     # 3 = linha reta
        return img, None, px_por_mm
    x1, y1, x2, y2 = struct.unpack(">ffff", roi[18:34])
    return img, (x1, y1, x2, y2), px_por_mm


def cinza(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if img.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fotos", type=Path, required=True)
    p.add_argument("--medidas", type=Path, required=True)
    p.add_argument("--model-type", choices=["yolo", "fasterrcnn"], default="yolo")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--saida", type=Path, default=Path("validacao.csv"))
    a = p.parse_args()
    if a.checkpoint is None:
        nome = "yolo11n_best.pt" if a.model_type == "yolo" else "fasterrcnn_best.pth"
        a.checkpoint = next((d / nome for d in (BASE, BASE.parent) if (d / nome).is_file()), BASE / nome)
    limiar = 0.5 if a.model_type == "yolo" else 0.8

    fotos = {f: pipeline.ler_imagem(f) for f in sorted(a.fotos.iterdir())
             if f.suffix.lower() in pipeline.EXTENSOES}
    fotos_cinza = {f: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for f, im in fotos.items() if im is not None}

    referencias = []
    for f in sorted(a.medidas.glob("*.tif"), key=lambda s: (len(s.stem), s.stem)):
        img, linha, escala = linha_imagej(f)
        if linha is None:
            print(f"  {f.name}: sem linha medida, ignorado")
            continue
        modelo_c = cinza(img)
        melhor = (-1, None, None)
        for foto, g in fotos_cinza.items():
            r = cv2.matchTemplate(g, modelo_c, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx > melhor[0]:
                melhor = (mx, foto, loc)
        if melhor[0] < 0.9:
            print(f"  {f.name}: não encontrei este recorte nas fotos (semelhança {melhor[0]:.2f}), ignorado")
            continue
        ox, oy = melhor[2]
        x1, y1, x2, y2 = linha
        referencias.append(dict(arquivo=f.name, foto=melhor[1], escala=escala,
                                x1=x1 + ox, y1=y1 + oy, x2=x2 + ox, y2=y2 + oy,
                                manual_px=math.hypot(x2 - x1, y2 - y1)))

    modelo = detectores.carregar_modelo(a.model_type, a.checkpoint)
    medidas_por_foto = {}
    for foto in sorted({r["foto"] for r in referencias}):
        medidas_por_foto[foto] = pipeline.medir_imagem(modelo, fotos[foto], limiar)

    linhas = []
    print(f"\n{'referência':28s} {'foto':26s} {'manual':>8s} {'auto':>8s} {'erro':>7s}  conferir")
    for r in referencias:
        mx, my = (r["x1"] + r["x2"]) / 2, (r["y1"] + r["y2"]) / 2
        achada = [d for d in medidas_por_foto[r["foto"]]
                  if d["caixa"][0] <= mx <= d["caixa"][2] and d["caixa"][1] <= my <= d["caixa"][3]]
        auto = achada[0]["res"]["comprimento_px"] if achada else None
        erro = (auto - r["manual_px"]) / r["manual_px"] * 100 if auto else None
        conferir = achada[0]["res"]["conferir"] if achada else True
        esc = r["escala"]
        linhas.append({"referencia": r["arquivo"], "foto": r["foto"].name,
                       "manual_px": round(r["manual_px"], 1), "auto_px": round(auto, 1) if auto else "",
                       "manual_mm": round(r["manual_px"] / esc, 4) if esc else "",
                       "auto_mm": round(auto / esc, 4) if (esc and auto) else "",
                       "erro_pct": round(erro, 1) if erro is not None else "",
                       "conferir": "sim" if conferir else "não",
                       "situacao": "ok" if achada else "Daphnia não detectada"})
        print(f"{r['arquivo']:28s} {r['foto'].name[:26]:26s} {r['manual_px']:8.1f} "
              f"{(f'{auto:8.1f}' if auto else '       -')} {(f'{erro:+6.1f}%' if erro is not None else '      -')}"
              f"  {'sim' if conferir else ''}")

    erros = np.array([l["erro_pct"] for l in linhas if l["erro_pct"] != ""], float)
    if erros.size:
        print(f"\nReferências comparadas: {erros.size} de {len(linhas)}")
        print(f"Erro mediano (absoluto): {np.median(np.abs(erros)):.1f}%   "
              f"tendência média: {erros.mean():+.1f}%   desvio-padrão: {erros.std(ddof=1) if erros.size > 1 else 0:.1f}%")
        print(f"Dentro de 5%: {(np.abs(erros) <= 5).sum()}   dentro de 10%: {(np.abs(erros) <= 10).sum()}   "
              f"pior caso: {erros[np.argmax(np.abs(erros))]:+.1f}%")
    with open(a.saida, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()) if linhas else ["referencia"])
        w.writeheader()
        w.writerows(linhas)
    print(f"Tabela salva em: {a.saida}")


if __name__ == "__main__":
    main()
