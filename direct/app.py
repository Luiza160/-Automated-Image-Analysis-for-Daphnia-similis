# app.py  —  python app.py  →  http://localhost:5000
# A página fica em static/index.html.
import csv, io, os, threading, uuid, zipfile, traceback
from argparse import Namespace
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, Response, abort

import medir_zebrafish as zebrafish
import pipeline     # daphnia (detector + medição direta do corpo, medicao.py)

BASE = Path(__file__).parent.resolve()
JOBS_DIR = BASE / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
EXT = zebrafish.EXTENSOES
JOBS = {}


def achar_pesos(nome):
    """Procura o arquivo de pesos nesta pasta e na pasta de cima."""
    for pasta in (BASE.parent / 'weights', BASE):
        if (pasta / nome).is_file():
            return pasta / nome
    return BASE / nome


# ---- Configuração do pipeline da Daphnia (ajuste por variáveis de ambiente) ----
_TIPO = os.getenv("MODEL_TYPE", "yolo")                            # yolo | fasterrcnn
CFG = {
    "model_type": _TIPO,
    "checkpoint": Path(os.getenv("CHECKPOINT") or
                       achar_pesos("yolo11n_best.pt" if _TIPO == "yolo" else "fasterrcnn_best.pth")),
    "score_threshold": float(os.getenv("SCORE_THRESHOLD", "0.5" if _TIPO == "yolo" else "0.8")),
}

app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="/static")


# ------------------------------- Zebrafish -------------------------------
def rodar_zebrafish(job, arquivos, px_por_mm):
    saida = job["dir"] / "saida"
    saida.mkdir(exist_ok=True)
    for i, f in enumerate(arquivos):
        job["mensagem"] = f"Medindo {f.name} ({i + 1}/{len(arquivos)})"
        item = {"nome": f.name, "px": None, "mm": None, "imagem": None, "erro": None}
        try:
            img = zebrafish.ler_imagem(f)
            if img is None:
                item["erro"] = "não foi possível ler a imagem"
            else:
                etapas = zebrafish.Etapas(saida / f.stem, ativo=False)
                res, erro = zebrafish.processar_imagem(img, etapas)
                if res is None:
                    item["erro"] = erro
                else:
                    px, img_final = res
                    destino = saida / f"medido_{f.stem}.png"
                    zebrafish.salvar_imagem(destino, img_final)
                    item["px"] = round(px, 2)
                    item["mm"] = round(px / px_por_mm, 3) if px_por_mm else None
                    item["imagem"] = f"/api/arquivo/{job['id']}/saida/{destino.name}"
        except Exception as e:  # uma imagem ruim não derruba o lote
            item["erro"] = str(e)
        job["resultados"].append(item)
        job["progresso"] = (i + 1) / len(arquivos)


# -------------------------------- Daphnia --------------------------------
def montar_args(job, entrada, px_por_mm):
    return Namespace(
        model_type=CFG["model_type"], checkpoint=CFG["checkpoint"],
        entrada=entrada, workdir=job["dir"] / "daphnia",
        score_threshold=CFG["score_threshold"], crop_scale_factor=1.6,
        px_por_mm=px_por_mm, salvar_imagens=True,
    )


def rodar_daphnia(job, arquivos, px_por_mm):
    # junta tudo numa pasta: imagens soltas + conteúdo de qualquer .zip enviado
    pasta = job["dir"] / "upload"
    for f in [a for a in arquivos if a.suffix.lower() == ".zip"]:
        with zipfile.ZipFile(f) as z:
            for n in z.namelist():
                if n.lower().endswith(EXT) and Path(n).name and "__MACOSX" not in n:
                    (pasta / Path(n).name).write_bytes(z.read(n))

    def progresso(fracao, mensagem):
        job["progresso"], job["mensagem"] = fracao, mensagem

    args = montar_args(job, pasta, px_por_mm)
    saida = pipeline.executar_pipeline(args, progresso)

    for row in saida["linhas"]:
        ok = row["comprimento_px"] != ""
        px = float(row["comprimento_px"]) if ok else None
        job["resultados"].append({
            "nome": f"{row['imagem']} · Daphnia {row['daphnia']}", "px": px,
            "mm": round(px / px_por_mm, 3) if (ok and px_por_mm) else None,
            "imagem": f"/api/arquivo/{job['id']}/daphnia/recortes_anotados/{row['recorte']}.jpg",
            "erro": None if ok else (row["motivo"] or "sem medida"),
            "aviso": f"conferir: {row['motivo']}" if (ok and row["conferir"] == "sim") else None,
        })
    job["progresso"] = 1.0


def executar(job, arquivos, organismo, px_por_mm):
    try:
        (rodar_zebrafish if organismo == "zebrafish" else rodar_daphnia)(job, arquivos, px_por_mm)
        job["status"] = "concluido"
        job["mensagem"] = "Análise concluída"
    except Exception as e:
        traceback.print_exc()
        job["status"] = "erro"
        job["mensagem"] = f"Falha na análise: {e}"


# --------------------------------- Rotas ---------------------------------
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/analisar")
def analisar():
    organismo = request.form.get("organismo", "zebrafish")
    if organismo not in ("zebrafish", "daphnia"):
        abort(400, "organismo inválido")
    px_por_mm = request.form.get("px_por_mm", type=float) or None

    job_id = uuid.uuid4().hex[:10]
    pasta = JOBS_DIR / job_id / "upload"
    pasta.mkdir(parents=True)
    arquivos = []
    for f in request.files.getlist("arquivos"):
        nome = Path(f.filename).name
        if Path(nome).suffix.lower() in EXT + (".zip",):
            destino = pasta / nome
            f.save(destino)
            arquivos.append(destino)
    if not arquivos:
        abort(400, "nenhuma imagem válida enviada")

    # zebrafish aceita .zip também: extrai as imagens
    if organismo == "zebrafish":
        expandidos = []
        for a in arquivos:
            if a.suffix.lower() == ".zip":
                with zipfile.ZipFile(a) as z:
                    for n in z.namelist():
                        if n.lower().endswith(EXT) and Path(n).name:
                            alvo = pasta / Path(n).name
                            alvo.write_bytes(z.read(n))
                            expandidos.append(alvo)
            else:
                expandidos.append(a)
        arquivos = expandidos

    job = {"id": job_id, "dir": JOBS_DIR / job_id, "organismo": organismo,
           "status": "processando", "progresso": 0.0, "mensagem": "Iniciando",
           "resultados": [], "px_por_mm": px_por_mm}
    JOBS[job_id] = job
    threading.Thread(target=executar, args=(job, arquivos, organismo, px_por_mm), daemon=True).start()
    return jsonify({"id": job_id})


@app.get("/api/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id) or abort(404)
    return jsonify({k: job[k] for k in ("id", "organismo", "status", "progresso",
                                        "mensagem", "resultados", "px_por_mm")})


@app.get("/api/arquivo/<job_id>/<path:rel>")
def arquivo(job_id, rel):
    job = JOBS.get(job_id) or abort(404)
    return send_from_directory(job["dir"], rel)  # send_from_directory bloqueia path traversal


@app.get("/api/exportar/<job_id>.csv")
def exportar(job_id):
    job = JOBS.get(job_id) or abort(404)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["arquivo", "comprimento_px", "comprimento_mm", "observacao"])
    for r in job["resultados"]:
        w.writerow([r["nome"], r["px"] if r["px"] is not None else "",
                    r["mm"] if r["mm"] is not None else "", r["erro"] or r.get("aviso") or ""])
    return Response("﻿" + buf.getvalue(), mimetype="text/csv",  # BOM: Excel lê os acentos
                    headers={"Content-Disposition": f"attachment; filename=medidas_{job_id}.csv"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
