"""
Mede o comprimento de UMA Daphnia dentro do recorte de uma detecção.

Substitui o Daphnia Ruler + Real-ESRGAN do pipeline antigo. Não usa rede neural:
são operações simples de imagem, por isso leva ~0,03 s por Daphnia em qualquer computador.

Passos:
  1. Redimensiona o recorte para que a caixa da detecção tenha sempre a mesma diagonal
     (DIAG_PADRAO). Assim o método se comporta igual em qualquer magnificação.
  2. Estima o fundo a partir das bordas do recorte (só há fundo ali, por causa da margem
     do recorte), como um plano inclinado por canal de cor: corrige iluminação desigual.
  3. Suaviza (apaga o ruído da câmera) e calcula a diferença de cor de cada pixel para o
     fundo. Limiar automático (Otsu) separa "animal" de "fundo". Como usa a cor do corpo
     inteiro, e não só a linha do contorno, o corpo aparece cheio mesmo com contorno fraco.
  4. Fecha falhas, preenche buracos e faz uma abertura leve para remover ruído fino
     sem encolher excessivamente a borda visível do corpo.
  5. Comprimento = eixo longitudinal do corpo, da ponta da cabeça até o início da
     cauda, sem incluir o espinho nem medir diagonalmente até a barriga.
  6. Marca para conferência manual quando o resultado é suspeito (ver motivos abaixo).

Validação (9 fotos, 25 Daphnias, 17 medidas manuais do ImageJ): erro mediano 2,7%,
16 de 17 dentro de 10% da medida manual, sem tendência a medir a mais ou a menos.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology

# ---------------- parâmetros (validados com as fotos de teste) ----------------
DIAG_PADRAO = 200.0        # diagonal da caixa (px) depois de redimensionar o recorte
SUAVIZACAO = 2.0           # sigma do filtro gaussiano, em px na escala padrão
BORDA_FUNDO = 6            # largura (px) da faixa nas bordas usada para estimar o fundo
ABERTURA_FRAC = 0.025      # abertura leve: reduz ruído sem cortar tanto o contorno do corpo
PROPORCAO_MIN = 0.5        # comprimento esperado entre 50% e 100% da diagonal da caixa
PROPORCAO_MAX = 1.0
DISCORDANCIA_MAX = 0.15    # paquímetro x elipse: se discordarem mais que 15%, conferir
SOLIDEZ_MIN = 0.78         # máscaras muito recortadas podem conter sobreposição ou ruído
LIMIAR_BORDA_FRAC = 0.90   # relaxa levemente Otsu para recuperar borda de baixo contraste
LIMIAR_CRESCIMENTO_FRAC = 0.68  # limiar mais permissivo, usado só junto à máscara
CRESCIMENTO_BORDA_PX = 5   # limite da recuperação local da borda na escala padrão
# -------------------------------------------------------------------------------


def _faixa_borda(a, b):
    """Valores da faixa de largura b nas quatro bordas do array a."""
    return np.concatenate([a[:b].reshape(-1, *a.shape[2:]), a[-b:].reshape(-1, *a.shape[2:]),
                           a[:, :b].reshape(-1, *a.shape[2:]), a[:, -b:].reshape(-1, *a.shape[2:])])


def _estimar_fundo(img_f):
    """Ajusta um plano (a*x + b*y + c) por canal usando só os pixels da borda."""
    h, w = img_f.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    bx = _faixa_borda(xx, BORDA_FUNDO).astype(np.float32)
    by = _faixa_borda(yy, BORDA_FUNDO).astype(np.float32)
    valores = _faixa_borda(img_f, BORDA_FUNDO)
    A = np.c_[bx, by, np.ones_like(bx)]
    fundo = np.empty_like(img_f)
    for c in range(img_f.shape[2]):
        coef, *_ = np.linalg.lstsq(A, valores[:, c], rcond=None)
        fundo[..., c] = coef[0] * xx + coef[1] * yy + coef[2]
    return fundo


def mascara_corpo(img_bgr, etapas=None):
    """Máscara do corpo (sem espinho e antenas) de um recorte já na escala padrão."""
    h, w = img_bgr.shape[:2]
    suave = cv2.GaussianBlur(img_bgr.astype(np.float32), (0, 0), SUAVIZACAO)
    fundo = _estimar_fundo(suave)
    diferenca = np.sqrt(((suave - fundo) ** 2).sum(2))
    if etapas is not None:
        etapas.extend([
            ("03_suavizada", suave),
            ("04_fundo_estimado", fundo),
            ("05_diferenca_para_fundo", diferenca),
        ])

    # Otsu sobre a diferença (escalada para 0-255 com resolução de 1/4)
    limiar, _ = cv2.threshold(np.clip(diferenca * 4, 0, 255).astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara = diferenca > (limiar / 4) * LIMIAR_BORDA_FRAC
    if etapas is not None:
        etapas.append(("06_limiar_otsu", mascara))

    raio = max(2, int(ABERTURA_FRAC * DIAG_PADRAO))
    mascara = morphology.closing(mascara, morphology.disk(max(2, raio // 2)))
    if etapas is not None:
        etapas.append(("07_fechamento", mascara.copy()))
    mascara = ndi.binary_fill_holes(mascara)
    if etapas is not None:
        etapas.append(("08_buracos_preenchidos", mascara.copy()))
    mascara = morphology.opening(mascara, morphology.disk(raio))
    if etapas is not None:
        etapas.append(("09_abertura", mascara.copy()))

    # Recupera bordas fracas que Otsu deixou de fora, mas somente numa faixa
    # estreita conectada ao corpo já segmentado. Isso evita crescer pelo fundo
    # inteiro e mantém a recuperação local ao contorno existente.
    limiar_fraco = (limiar / 4) * LIMIAR_BORDA_FRAC * LIMIAR_CRESCIMENTO_FRAC
    candidatos = diferenca > limiar_fraco
    faixa = morphology.dilation(mascara, morphology.disk(CRESCIMENTO_BORDA_PX))
    mascara = ndi.binary_propagation(mascara, mask=(candidatos & faixa))
    if etapas is not None:
        etapas.append(("10_borda_recuperada", mascara.copy()))

    rotulos = measure.label(mascara)
    if rotulos.max() == 0:
        return None
    # região grande e perto do centro do recorte (o animal detectado, não sujeira)
    regioes = measure.regionprops(rotulos)
    cy, cx = h / 2, w / 2
    escolhida = max(regioes, key=lambda r: r.area / (1 + np.hypot(r.centroid[0] - cy,
                                                                  r.centroid[1] - cx) / max(h, w)))
    resultado = rotulos == escolhida.label
    if etapas is not None:
        etapas.append(("11_componente_escolhido", resultado.copy()))
    return resultado


def _salvar_etapas(pasta, etapas):
    """Salva os intermediários em PNG, normalizando arrays para visualização."""
    if pasta is None:
        return
    from pathlib import Path
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    for indice, (nome, imagem) in enumerate(etapas, 1):
        a = np.asarray(imagem)
        if a.dtype == bool:
            a = a.astype(np.uint8) * 255
        elif a.dtype != np.uint8:
            a = np.nan_to_num(a.astype(np.float32))
            if a.ndim == 2:
                a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                a = np.clip(a, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", a)
        if ok:
            destino = pasta / f"{nome}.png"
            try:
                destino.write_bytes(buf.tobytes())
            except OSError:
                # Fallback curto para ambientes Windows/OneDrive com problemas
                # ao criar arquivos usando nomes longos.
                (pasta / f"{indice:02d}.png").write_bytes(buf.tobytes())


def comprimento_paquimetro(mascara):
    """Estima o eixo do corpo usando o perfil de largura da máscara.

    Pixels muito finos na ponta do recorte (por exemplo, antenas) deixam de
    definir o extremo. O ponto final fica onde a seção transversal volta a
    ultrapassar uma fração da largura máxima do corpo.
    """
    pts = np.argwhere(mascara)[:, ::-1].astype(float)  # (x, y)
    centro = pts.mean(axis=0)
    _, vetores = np.linalg.eigh(np.cov(pts.T))
    eixo = vetores[:, -1]
    proj = (pts - centro) @ eixo
    transversal = (pts - centro) @ np.array([-eixo[1], eixo[0]])

    # Perfil da largura perpendicular ao eixo. Percentis reduzem a influência
    # de ruído; uma mediana suaviza mudanças de um único pixel entre faixas.
    inicio, fim = np.percentile(proj, [0.5, 99.5])
    n_faixas = int(np.clip(round((fim - inicio) / 3.0), 24, 100))
    bordas = np.linspace(inicio, fim, n_faixas + 1)
    centros = (bordas[:-1] + bordas[1:]) / 2
    larguras = np.zeros(n_faixas, dtype=float)
    for i in range(n_faixas):
        secao = transversal[(proj >= bordas[i]) & (proj < bordas[i + 1])]
        if secao.size:
            larguras[i] = np.percentile(secao, 90) - np.percentile(secao, 10)
    validas = larguras > 0
    if validas.sum() < 5:
        a, b = np.percentile(proj, [1.0, 99.0])
    else:
        larguras = np.interp(np.arange(n_faixas), np.flatnonzero(validas), larguras[validas])
        larguras = ndi.median_filter(larguras, size=5, mode="nearest")
        largura_max = float(larguras.max())
        limite = max(2.0, largura_max * 0.28)

        # Nas duas pontas, ignora somente a extensão cuja largura permanece
        # abaixo do limite; a máscara inteira continua sendo usada no contorno.
        indices = np.flatnonzero(larguras >= limite)
        a, b = centros[indices[0]], centros[indices[-1]]

    faixa = max(1.5, (b - a) * 0.015)
    frente = pts[np.abs(proj - a) <= faixa].mean(axis=0)
    traseira = pts[np.abs(proj - b) <= faixa].mean(axis=0)
    return float(np.linalg.norm(traseira - frente)), frente, traseira


def medir_recorte(recorte_bgr, diag_caixa, pasta_etapas=None):
    """
    Mede a Daphnia no recorte (BGR, escala original da foto).
    diag_caixa: diagonal da caixa da detecção, em px da foto original.

    Retorna dict com:
      comprimento_px  (float ou None), em px da foto original
      p1, p2          pontos extremos da medida, em coordenadas do recorte original
      conferir        True se o resultado deve ser conferido por uma pessoa
      motivo          texto explicando por que conferir ("" se não precisa)
      mascara, escala máscara na escala padrão e fator de escala usado (para desenhar)
    """
    escala = DIAG_PADRAO / float(diag_caixa)
    interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    pequeno = cv2.resize(recorte_bgr, None, fx=escala, fy=escala, interpolation=interp)
    etapas = [("01_recorte_original", recorte_bgr), ("02_tamanho_padronizado", pequeno)]

    mascara = mascara_corpo(pequeno, etapas if pasta_etapas is not None else None)
    if mascara is None:
        _salvar_etapas(pasta_etapas, etapas)
        return dict(comprimento_px=None, p1=None, p2=None, conferir=True,
                    motivo="não foi possível separar o animal do fundo", mascara=None, escala=escala)

    L, p1, p2 = comprimento_paquimetro(mascara)
    eixo_img = pequeno.copy()
    a, b = tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int))
    cv2.line(eixo_img, a, b, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(eixo_img, a, 3, (0, 255, 0), -1)
    cv2.circle(eixo_img, b, 3, (255, 0, 0), -1)
    etapas.append(("12_eixo_e_extremos", eixo_img))
    final = pequeno.copy()
    cv2.line(final, a, b, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(final, f"Distancia estimada: {L / escala:.1f} px", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    etapas.append(("13_distancia_final", final))
    _salvar_etapas(pasta_etapas, etapas)
    elipse = measure.regionprops(mascara.astype(np.uint8))[0].axis_major_length
    contorno = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    casco = cv2.convexHull(max(contorno, key=cv2.contourArea)) if contorno else None
    solidez = (np.count_nonzero(mascara) / cv2.contourArea(casco)) if casco is not None and cv2.contourArea(casco) else 1.0

    motivos = []
    proporcao = L / DIAG_PADRAO
    if not PROPORCAO_MIN <= proporcao <= PROPORCAO_MAX:
        motivos.append(f"medida fora do esperado ({proporcao:.0%} da caixa)")
    if mascara[0].any() or mascara[-1].any() or mascara[:, 0].any() or mascara[:, -1].any():
        motivos.append("animal encosta na borda do recorte (pode estar cortado)")
    if abs(L - elipse) / L > DISCORDANCIA_MAX:
        motivos.append("formato irregular")
    if solidez < SOLIDEZ_MIN:
        motivos.append("contorno irregular ou possível sobreposição; conferir máscara")

    return dict(comprimento_px=L / escala, p1=p1 / escala, p2=p2 / escala,
                conferir=bool(motivos), motivo="; ".join(motivos), mascara=mascara, escala=escala)


def desenhar_recorte(recorte_bgr, res):
    """Recorte com o contorno do corpo (azul) e a linha medida (vermelha, ou laranja se conferir)."""
    img = recorte_bgr.copy()
    esp = max(1, int(round(max(img.shape[:2]) / 250)))
    if res.get("mascara") is not None:
        m = cv2.resize(res["mascara"].astype(np.uint8), (img.shape[1], img.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
        contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contornos, -1, (230, 150, 0), esp)
    if res.get("p1") is not None:
        cor = (0, 140, 255) if res["conferir"] else (0, 0, 230)
        a = tuple(int(round(v)) for v in res["p1"])
        b = tuple(int(round(v)) for v in res["p2"])
        cv2.line(img, a, b, cor, esp + 1, cv2.LINE_AA)
        cv2.circle(img, a, esp + 2, cor, -1)
        cv2.circle(img, b, esp + 2, cor, -1)
    return img
