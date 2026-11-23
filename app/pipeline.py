import csv
import math
import os
import shutil
import zipfile
import subprocess
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import detectores
import filtros


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def _nucleos(args):
    return args.cpus or os.cpu_count() or 1


def _limitar_threads(threads):
    """Limita as threads de OpenCV e PyTorch no processo atual."""
    try:
        import cv2
        cv2.setNumThreads(threads)
    except ImportError:
        pass
    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        pass


def extrair_zip(zip_path, target_dir):
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    extensoes_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    with zipfile.ZipFile(zip_path.resolve()) as archive:
        for file in archive.namelist():
            if file.lower().endswith(extensoes_validas):
                filename = Path(file).name
                if filename:
                    target_path = target_dir / filename
                    with archive.open(file) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)

    image_paths = sorted([p for p in target_dir.glob("*") if p.suffix.lower() in extensoes_validas])
    return image_paths


# ----------------------------------------------------------------------------
# Detecção paralela (N processos x M threads ~ núcleos)
# ----------------------------------------------------------------------------
_MODELO = None


def _init_detector(model_type, checkpoint, device, threads):
    global _MODELO
    import torch
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _MODELO = detectores.carregar_modelo(model_type, checkpoint, device)


def _detectar(item):
    caminho, score = item
    return caminho, detectores.predizer(_MODELO, caminho, score)


def detectar_paralelo(img_paths, args):
    nucleos = _nucleos(args)
    n_workers = max(1, min(args.workers_deteccao, len(img_paths)))
    threads = max(1, nucleos // n_workers)
    print(f"  Detecção: {n_workers} processo(s) x {threads} thread(s)")

    itens = [(p, args.score_threshold) for p in img_paths]
    ctx = mp.get_context("spawn")
    resultados = {}
    with ctx.Pool(n_workers, initializer=_init_detector,
                  initargs=(args.model_type, args.checkpoint,
                            getattr(args, "device_deteccao", "cpu"), threads)) as pool:
        for caminho, boxes in pool.imap(_detectar, itens, chunksize=1):
            resultados[caminho] = boxes
            print(f"  {caminho.name}: {len(boxes)} detecção(ões)", flush=True)
    return resultados


# ----------------------------------------------------------------------------
# Aplicação paralela dos filtros
# ----------------------------------------------------------------------------
_FILTRO = None
_ORIG_DIR = None


def _aplicar_item(filtro, item, originais_dir):
    nome, src, dst = item
    with Image.open(src) as im:
        img = im.convert("RGB")
    saida = filtro.aplicar(img)
    saida.save(dst)
    with Image.open(originais_dir / nome) as orig:
        escala = saida.width / orig.width
    return nome, escala


def _init_worker(nome_filtro, args, originais_dir, threads):
    global _FILTRO, _ORIG_DIR
    _limitar_threads(threads)
    _FILTRO = filtros.obter_filtro(nome_filtro, args)
    _ORIG_DIR = originais_dir


def _processar(item):
    return _aplicar_item(_FILTRO, item, _ORIG_DIR)


def aplicar_filtro_paralelo(nome_filtro, filtro, args, itens, originais_dir):
    nucleos = _nucleos(args)
    escalas = {}

    if nome_filtro == "realesrgan":
        # Pesado: poucos processos, cada um com várias threads e seu próprio modelo.
        n_workers = max(1, min(args.workers_esrgan, len(itens)))
        threads = max(1, nucleos // n_workers)
        print(f"  Real-ESRGAN: {n_workers} processo(s) x {threads} thread(s)")
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_init_worker,
                      initargs=(nome_filtro, args, originais_dir, threads)) as pool:
            for i, (nome, esc) in enumerate(pool.imap_unordered(_processar, itens, chunksize=1), 1):
                escalas[nome] = esc
                print(f"    [{i}/{len(itens)}] {nome}", flush=True)
    else:
        # Filtros leves de OpenCV (liberam o GIL): threads bastam, sem custo de spawn.
        n_threads = max(1, min(nucleos, len(itens)))
        print(f"  {nome_filtro}: {n_threads} thread(s)")
        _limitar_threads(1)
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            for nome, esc in ex.map(lambda it: _aplicar_item(filtro, it, originais_dir), itens):
                escalas[nome] = esc
    return escalas


# ----------------------------------------------------------------------------
# Daphnia Ruler
# ----------------------------------------------------------------------------
def rodar_daphruler(pasta, python_exe):
    if not any(pasta.glob("*.png")):
        return {}
    cmd = [str(python_exe), "-m", "daphruler", "-p", str(pasta)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ruler_csv = next(pasta.rglob("measurement_results_*"), None)
    if not ruler_csv:
        return {}

    medidas = {}
    with open(ruler_csv) as f:
        for row in csv.DictReader(f):
            nome = row["ID"].replace("\\", "/").split("/")[-1]
            valor = row.get("body.Length.h", "NA")
            medidas[nome] = None if valor in ("NA", "", None) else float(valor)
    return medidas


def _link_ou_copia(origem, destino):
    try:
        os.link(origem, destino)  # hardlink: sem gastar disco
    except OSError:
        shutil.copy(origem, destino)


def rodar_daphruler_paralelo(pasta, python_exe, n_lotes):
    pngs = sorted(pasta.glob("*.png"))
    if not pngs:
        return {}
    n_lotes = max(1, min(n_lotes, len(pngs)))
    if n_lotes == 1:
        return rodar_daphruler(pasta, python_exe)

    lotes_root = pasta.parent / "lotes_ruler"
    if lotes_root.exists():
        shutil.rmtree(lotes_root)
    lotes = []
    for i in range(n_lotes):
        d = lotes_root / f"lote_{i}"
        d.mkdir(parents=True)
        lotes.append(d)
    for j, p in enumerate(pngs):
        _link_ou_copia(p, lotes[j % n_lotes] / p.name)

    print(f"  Daphnia Ruler: {n_lotes} lote(s) em paralelo")
    medidas = {}
    with ThreadPoolExecutor(max_workers=n_lotes) as ex:
        for parcial in ex.map(lambda d: rodar_daphruler(d, python_exe), lotes):
            medidas.update(parcial)
    return medidas


# ----------------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------------
def executar_pipeline_cascata(args):
    px_por_mm = float(getattr(args, "px_por_mm", 0.0))
    aumento = getattr(args, "aumento_objetiva", None)
    if not math.isfinite(px_por_mm) or px_por_mm < 0:
        raise ValueError("px_por_mm deve ser um número finito maior ou igual a zero.")
    if aumento is not None and aumento <= 0:
        raise ValueError("O aumento da objetiva deve ser positivo.")
    # Falha cedo: evita rodar a detecção inteira para só depois descobrir que faltam os pesos
    if "realesrgan" in args.filtros and not Path(args.realesrgan_weights).is_file():
        raise FileNotFoundError(
            f"Pesos do Real-ESRGAN não encontrados: {Path(args.realesrgan_weights).resolve()}\n"
            "Baixe RealESRGAN_x4plus.pth e/ou informe o caminho com --realesrgan-weights."
        )

    args.workdir.mkdir(parents=True, exist_ok=True)
    originais_dir = args.workdir / "recortes_originais"
    meta_csv = args.workdir / "crop_metadata.csv"

    if args.reusar_recortes and meta_csv.exists() and originais_dir.exists():
        print("Reutilizando recortes existentes de detecções anteriores...")
        with open(meta_csv, encoding="utf-8") as f:
            metadata = list(csv.DictReader(f))
            for m in metadata:
                m["diagonal_caixa_px"] = float(m["diagonal_caixa_px"])
    else:
        img_paths = extrair_zip(args.dataset_zip, args.workdir / "imagens_extraidas")

        if originais_dir.exists():
            shutil.rmtree(originais_dir)
        originais_dir.mkdir(parents=True)

        print("\n=== Executando Detecção nos Alvos ===")
        deteccoes = detectar_paralelo(img_paths, args)

        metadata = []
        crop_id = 0
        for p in img_paths:
            boxes = deteccoes[p]
            img = Image.open(p).convert("RGB")
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                ew, eh = w * (args.crop_scale_factor - 1) / 2, h * (args.crop_scale_factor - 1) / 2
                cx1, cy1 = max(0, int(x1 - ew)), max(0, int(y1 - eh))
                cx2, cy2 = min(img.width, int(x2 + ew)), min(img.height, int(y2 + eh))

                crop_name = f"{crop_id:04d}_{p.stem}_d{i}.png"
                img.crop((cx1, cy1, cx2, cy2)).save(originais_dir / crop_name)
                metadata.append({"crop_name": crop_name, "image_name": p.name,
                                 "diagonal_caixa_px": math.hypot(w, h)})
                crop_id += 1

        with open(meta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["crop_name", "image_name", "diagonal_caixa_px"])
            writer.writeheader()
            writer.writerows(metadata)

    meta_por_nome = {m["crop_name"]: m for m in metadata}
    pendentes = [m["crop_name"] for m in metadata]
    fonte = {n: originais_dir / n for n in pendentes}

    run_dir = args.workdir / "execucoes" / (args.nome_execucao or "+".join(args.filtros))
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{len(metadata)} Daphnias recortadas prontas para a cascata de filtros.")

    resolvidos = {}
    resumo_etapas = []

    for k, nome_filtro in enumerate(args.filtros):
        if not pendentes:
            print("\nNenhum recorte pendente remanescente. Encerrando cascata mais cedo!")
            break

        filtro = filtros.obter_filtro(nome_filtro, args)  # leve: o modelo só carrega nos workers
        filtros_acumulados = "+".join(args.filtros[:k + 1])
        etapa_dir = run_dir / f"etapa_{k}_{filtro.nome}"
        recortes_dir = etapa_dir / "recortes"
        recortes_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Etapa {k}: {filtros_acumulados} ({len(pendentes)} recortes pendentes) ===")

        itens = [(n, fonte[n], recortes_dir / n) for n in pendentes]
        escalas = aplicar_filtro_paralelo(nome_filtro, filtro, args, itens, originais_dir)
        for n in pendentes:
            fonte[n] = recortes_dir / n

        print(f"  Rodando Daphnia Ruler na pasta da Etapa {k}...")
        medidas = rodar_daphruler_paralelo(recortes_dir, args.daphruler_python, args.ruler_workers)

        contagem = {"plausivel": 0, "implausivel": 0, "sem_medida": 0}
        novos_pendentes = []

        for nome in pendentes:
            meta = meta_por_nome[nome]
            if medidas.get(nome) is None:
                status = "sem_medida"
                novos_pendentes.append(nome)
            else:
                # Normaliza o tamanho medido para a escala do recorte original
                medida_px_original = medidas[nome] / escalas[nome]
                proporcao = medida_px_original / float(meta["diagonal_caixa_px"])

                if args.min_proporcao <= proporcao <= args.max_proporcao:
                    status = "plausivel"
                    resolvidos[nome] = {
                        "crop_name": nome, "status": "plausivel", "etapa_sucesso": k,
                        "filtro": nome_filtro, "medida_px_original": round(medida_px_original, 2),
                        "proporcao": round(proporcao, 3)
                    }
                else:
                    status = "implausivel"
                    novos_pendentes.append(nome)

            contagem[status] += 1

        print(f"  Resultado Etapa {k}: plausivel={contagem['plausivel']}, "
              f"implausivel={contagem['implausivel']}, sem_medida={contagem['sem_medida']}")
        resumo_etapas.append((k, filtros_acumulados, len(pendentes), contagem["plausivel"]))
        pendentes = novos_pendentes

    # Quem terminou sem solução satisfatória após todos os filtros
    for nome in pendentes:
        resolvidos[nome] = {
            "crop_name": nome, "status": "nao_resolvido", "etapa_sucesso": "N/A",
            "filtro": "N/A", "medida_px_original": "N/A", "proporcao": "N/A"
        }

    output_csv = run_dir / "avaliacao_cascata.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_name", "status", "etapa_sucesso",
                                               "filtro", "medida_px_original", "proporcao",
                                               "aumento_objetiva", "px_por_mm", "comprimento_mm"])
        writer.writeheader()
        for m in metadata:
            linha = resolvidos[m["crop_name"]].copy()
            linha["aumento_objetiva"] = aumento if aumento is not None else ""
            linha["px_por_mm"] = px_por_mm if px_por_mm > 0 else ""
            medida_px = linha["medida_px_original"]
            linha["comprimento_mm"] = (
                round(float(medida_px) / px_por_mm, 6)
                if px_por_mm > 0 and isinstance(medida_px, (int, float)) else ""
            )
            writer.writerow(linha)

    print("\n=== RESUMO GERAL DA CASCATA ===")
    total_geral = len(metadata)
    acumulado = 0
    for k, nome, entrada, plaus in resumo_etapas:
        acumulado += plaus
        pct = 100 * acumulado / total_geral if total_geral else 0.0
        print(f"  Etapa {k} ({nome}): {plaus}/{entrada} novos plausíveis | "
              f"Acumulado: {acumulado}/{total_geral} ({pct:.1f}%)")
    print(f"  Não resolvidos após todos os filtros: {len(pendentes)}")
    print(f"Planilha de dados salva com sucesso em: {output_csv}")