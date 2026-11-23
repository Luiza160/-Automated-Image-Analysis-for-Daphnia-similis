"""
Medição do comprimento de zebrafish em lote.

Uso:
    python medir_zebrafish.py CAMINHO_DA_PASTA
    python medir_zebrafish.py CAMINHO_DA_PASTA --px-por-mm 85.3
    python medir_zebrafish.py CAMINHO_DA_PASTA --sem-etapas

Dependências: pip install opencv-python numpy
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

# ======================= PARÂMETROS AJUSTÁVEIS =======================
EXTENSOES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

CLAHE_CLIP = 3.0
CLAHE_GRID = (8, 8)
BLUR_KERNEL = (11, 11)          # deve ser ímpar

FUNDO_KERNEL_FRAC = 0.30        # tamanho do filtro de fundo (fração da imagem reduzida).
                                # aumente se o peixe for muito grande na foto

FATOR_FRACO = 0.55              # limiar fraco = Otsu * fator.
                                # MENOR = mais cauda (mas mais ruído/riscos); MAIOR = mais conservador

ABERTURA_FRAC = 0.008           # largura mínima do que é "peixe" (fração do lado maior da imagem).
                                # ~0.008 = 18 px numa imagem de 2200 px. MAIOR = remove riscos mais grossos
RECONSTRUCAO_FRAC = 0.03        # quanto recuperar a partir do corpo (fração do lado maior).
                                # devolve a ponta fina da cauda. MAIOR = recupera cauda maior,
                                # mas deixa pedaços de risco

CLOSING_KERNEL = 9
CLOSING_ITER = 2
DILATACAO_KERNEL = 0            # 0 = sem dilatação (medida sem viés)

AREA_MINIMA_PEIXE = 500         # área mínima do maior contorno (px²)

USAR_FRAGMENTOS = False         # True = inclui pedaços próximos ao peixe (ex.: cauda solta)
PERIMETRO_MINIMO_FRAGMENTO = 40
MARGEM_FRAGMENTO = 0.15
# =====================================================================


def ler_imagem(caminho):
    """Leitura segura para caminhos com acentos (Windows)."""
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    if dados.size == 0:
        return None
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def salvar_imagem(caminho, img, tentativas=4):
    """Escrita segura para caminhos com acentos, com novas tentativas
    (útil quando OneDrive/antivírus travam o arquivo por instantes)."""
    ok, buf = cv2.imencode(caminho.suffix or ".png", img)
    if not ok:
        print(f"[Aviso] Falha ao codificar {caminho.name}")
        return False
    dados = buf.tobytes()
    for t in range(tentativas):
        try:
            with open(caminho, "wb") as f:
                f.write(dados)
            return True
        except OSError as e:
            if t < tentativas - 1:
                time.sleep(0.5 * (t + 1))
            else:
                print(f"[Aviso] Não foi possível salvar {caminho.name}: {e}")
    return False


class Etapas:
    """Salva imagens intermediárias numeradas automaticamente (01_..., 02_...)."""

    def __init__(self, pasta, ativo=True):
        self.pasta = pasta
        self.ativo = ativo
        self.n = 0
        if ativo:
            pasta.mkdir(parents=True, exist_ok=True)

    def salvar(self, nome, img):
        self.n += 1
        if self.ativo:
            salvar_imagem(self.pasta / f"{self.n:02d}_{nome}.png", img)


def estimar_fundo(img_gray):
    """Estima o fundo com mediana grande (calculada em versão reduzida, por velocidade)."""
    h, w = img_gray.shape
    escala = min(1.0, 256 / max(h, w))
    pequena = cv2.resize(img_gray, (max(1, int(w * escala)), max(1, int(h * escala))),
                         interpolation=cv2.INTER_AREA)
    k = int(max(pequena.shape) * FUNDO_KERNEL_FRAC)
    k = max(3, k | 1)  # ímpar e >= 3
    fundo = cv2.medianBlur(pequena, k)
    return cv2.resize(fundo, (w, h), interpolation=cv2.INTER_LINEAR)


def histerese(forte, fraca):
    """Mantém só as regiões fracas que contêm algum pixel forte."""
    n, rotulos = cv2.connectedComponents(fraca, connectivity=8)
    rot_fortes = np.unique(rotulos[forte > 0])
    rot_fortes = rot_fortes[rot_fortes != 0]
    return (np.isin(rotulos, rot_fortes).astype(np.uint8)) * 255


def remover_riscos(mascara):
    """Remove estruturas finas (riscos da placa) ligadas ao peixe.

    1) Abertura: apaga tudo mais fino que o kernel, deixando só o corpo.
    2) Reconstrução limitada: devolve da máscara original apenas o que está
       a poucos pixels do corpo (recupera a ponta fina da cauda, mas não
       deixa os riscos longos voltarem).
    """
    h, w = mascara.shape
    lado = max(h, w)

    k = max(3, int(lado * ABERTURA_FRAC) | 1)
    kernel_a = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    nucleo = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_a)

    # fica só com o maior componente (o corpo do peixe)
    n, rot, stats, _ = cv2.connectedComponentsWithStats(nucleo, connectivity=8)
    if n <= 1:
        return mascara  # abertura apagou tudo: mantém a máscara original
    maior = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    atual = ((rot == maior).astype(np.uint8)) * 255

    # reconstrução geodésica limitada
    passos = max(1, int(lado * RECONSTRUCAO_FRAC))
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(passos):
        novo = cv2.bitwise_and(cv2.dilate(atual, kernel_3), mascara)
        if np.array_equal(novo, atual):
            break
        atual = novo
    return atual


def selecionar_contornos(contornos):
    principal = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(principal) < AREA_MINIMA_PEIXE:
        return None
    if not USAR_FRAGMENTOS:
        return [principal]

    x, y, w, h = cv2.boundingRect(principal)
    m = int(MARGEM_FRAGMENTO * max(w, h))
    x0, y0, x1, y1 = x - m, y - m, x + w + m, y + h + m

    escolhidos = [principal]
    for c in contornos:
        if c is principal or cv2.arcLength(c, True) < PERIMETRO_MINIMO_FRAGMENTO:
            continue
        cx, cy, cw, ch = cv2.boundingRect(c)
        if x0 <= cx + cw / 2 <= x1 and y0 <= cy + ch / 2 <= y1:
            escolhidos.append(c)
    return escolhidos


def maior_distancia(contornos):
    """Maior distância entre dois pontos do conjunto de contornos."""
    pontos = np.vstack(contornos)
    hull = cv2.convexHull(pontos)
    pts = hull[:, 0, :].astype(np.float64)
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    i, j = np.unravel_index(dist.argmax(), dist.shape)
    p1 = (int(pts[i][0]), int(pts[i][1]))
    p2 = (int(pts[j][0]), int(pts[j][1]))
    return float(dist[i, j]), p1, p2, hull


def processar_imagem(img, etapas):
    etapas.salvar("original", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    etapas.salvar("escala_de_cinza", gray)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    gray_eq = clahe.apply(gray)
    etapas.salvar("clahe", gray_eq)

    blur = cv2.GaussianBlur(gray_eq, BLUR_KERNEL, 0)
    etapas.salvar("gaussian_blur", blur)

    # Diferença para o fundo estimado (remove gradientes de iluminação)
    fundo = estimar_fundo(blur)
    etapas.salvar("fundo_estimado", fundo)
    diff = cv2.absdiff(fundo, blur)
    etapas.salvar("diferenca_para_fundo", cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX))

    # Limiar forte (Otsu) e limiar fraco (fração do Otsu)
    otsu_val, forte = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    limiar_fraco = max(2, otsu_val * FATOR_FRACO)
    _, fraca = cv2.threshold(diff, limiar_fraco, 255, cv2.THRESH_BINARY)
    etapas.salvar("mascara_forte_otsu", forte)
    etapas.salvar("mascara_fraca", fraca)

    mascara = histerese(forte, fraca)
    etapas.salvar("mascara_histerese", mascara)

    # Remove os riscos da placa (finos) e mantém o peixe (grosso)
    mascara = remover_riscos(mascara)
    etapas.salvar("mascara_sem_riscos", mascara)

    kernel_c = np.ones((CLOSING_KERNEL, CLOSING_KERNEL), np.uint8)
    final = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_c, iterations=CLOSING_ITER)
    etapas.salvar("morfologia_closing", final)

    if DILATACAO_KERNEL > 0:
        kernel_d = np.ones((DILATACAO_KERNEL, DILATACAO_KERNEL), np.uint8)
        final = cv2.dilate(final, kernel_d, iterations=1)
        etapas.salvar("morfologia_dilatacao", final)

    contornos, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, "nenhum contorno encontrado"

    img_todos = img.copy()
    cv2.drawContours(img_todos, contornos, -1, (0, 255, 0), 2)
    etapas.salvar("todos_contornos", img_todos)

    escolhidos = selecionar_contornos(contornos)
    if escolhidos is None:
        return None, "objeto muito pequeno"

    img_sel = img.copy()
    cv2.drawContours(img_sel, escolhidos, -1, (0, 0, 255), 2)
    etapas.salvar("contorno_selecionado", img_sel)

    comprimento, p1, p2, hull = maior_distancia(escolhidos)

    img_hull = img.copy()
    cv2.drawContours(img_hull, [hull], -1, (255, 0, 255), 2)
    etapas.salvar("convex_hull", img_hull)

    resultado = img.copy()
    cv2.drawContours(resultado, escolhidos, -1, (0, 0, 255), 2)
    cv2.line(resultado, p1, p2, (0, 255, 0), 3)
    cv2.circle(resultado, p1, 5, (255, 0, 0), -1)
    cv2.circle(resultado, p2, 5, (255, 0, 0), -1)
    etapas.salvar("resultado_final", resultado)

    return (comprimento, resultado), None


def medir_zebrafish_lote(caminho_pasta, px_por_mm=None, salvar_etapas=True):
    pasta = Path(caminho_pasta).expanduser().resolve()

    if not pasta.is_dir():
        print(f"Erro: o caminho '{pasta}' não existe ou não é uma pasta.")
        return

    imagens = sorted(
        f for f in pasta.iterdir() if f.is_file() and f.suffix.lower() in EXTENSOES
    )
    if not imagens:
        print(f"Nenhuma imagem encontrada na pasta: {pasta}")
        return

    print(f"Iniciando medição de {len(imagens)} imagens em: {pasta}\n")

    pasta_saida = pasta / "resultados_medicao"
    pasta_saida.mkdir(exist_ok=True)

    linhas = []
    for img_path in imagens:
        img = ler_imagem(img_path)
        if img is None:
            print(f"[Aviso] Não foi possível ler {img_path.name}")
            continue

        etapas = Etapas(pasta_saida / f"etapas_{img_path.stem}", ativo=salvar_etapas)
        resultado, erro = processar_imagem(img, etapas)

        if resultado is None:
            print(f"[Aviso] {img_path.name}: {erro}")
            linhas.append([img_path.name, "", ""])
            continue

        comprimento_px, img_final = resultado
        salvar_imagem(pasta_saida / f"medido_{img_path.stem}.png", img_final)

        comprimento_mm = f"{comprimento_px / px_por_mm:.3f}" if px_por_mm else ""
        linhas.append([img_path.name, f"{comprimento_px:.2f}", comprimento_mm])

        msg = f"{comprimento_px:.2f} px"
        if px_por_mm:
            msg += f" ({comprimento_mm} mm)"
        print(f"[Sucesso] {img_path.name} -> {msg}")

    with open(pasta_saida / "medidas.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["arquivo", "comprimento_px", "comprimento_mm"])
        escritor.writerows(linhas)

    print(f"\nProcessamento concluído! Resultados e medidas.csv em: {pasta_saida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mede o comprimento de zebrafish em imagens.")
    parser.add_argument("pasta", nargs="?", default=".",
                        help="pasta com as imagens (padrão: pasta atual)")
    parser.add_argument("--px-por-mm", type=float, default=None,
                        help="escala da imagem, para converter pixels em mm")
    parser.add_argument("--sem-etapas", action="store_true",
                        help="não salva as imagens intermediárias")
    args = parser.parse_args()

    medir_zebrafish_lote(args.pasta, args.px_por_mm, not args.sem_etapas)