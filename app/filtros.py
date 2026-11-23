import numpy as np
import cv2
from PIL import Image


class SemFiltro:
    nome = "sem_filtro"

    def aplicar(self, img):
        return img


class RealESRGANFiltro:
    """Super-resolução Real-ESRGAN. O modelo só é carregado na primeira chamada
    de aplicar(), então o processo principal pode instanciar o filtro (para ler
    o nome) sem gastar RAM. Cada worker carrega a sua própria cópia."""

    def __init__(self, weights, outscale=4.0, tile=0, device="cpu"):
        self.weights = weights
        self.outscale = outscale
        self.tile = tile
        self.device = device
        self.upsampler = None
        self.nome = f"realesrgan_x{outscale:g}"

    @staticmethod
    def _compat_torchvision():
        """basicsr importa torchvision.transforms.functional_tensor, removido no
        torchvision >= 0.17. Recriamos o módulo apontando para o equivalente atual."""
        import sys
        import types
        try:
            import torchvision.transforms.functional_tensor  # noqa: F401
        except ImportError:
            from torchvision.transforms import functional as F
            shim = types.ModuleType("torchvision.transforms.functional_tensor")
            shim.rgb_to_grayscale = F.rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = shim

    def _carregar(self):
        import torch
        self._compat_torchvision()
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        dev = torch.device(self.device)
        if dev.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("Dispositivo 'xpu' pedido, mas torch.xpu.is_available() é False. "
                               "Instale a build XPU do PyTorch ou use --device cpu.")
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Dispositivo 'cuda' pedido, mas não há GPU CUDA disponível.")
        rede = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=str(self.weights),
            model=rede,
            tile=self.tile,
            half=(dev.type == "cuda"),  # half só funciona em GPU
            device=dev,
        )

    def aplicar(self, img):
        if self.upsampler is None:
            self._carregar()
        bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        saida, _ = self.upsampler.enhance(bgr, outscale=self.outscale)
        return Image.fromarray(cv2.cvtColor(saida, cv2.COLOR_BGR2RGB))


class MedianFiltro:
    """Redução de ruído impulsivo preservando relativamente as bordas."""
    nome = "median"

    def __init__(self, kernel_size=3):
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size deve ser ímpar e >= 3")
        self.kernel_size = kernel_size

    def aplicar(self, img):
        arr = np.asarray(img.convert("RGB"))
        out = cv2.medianBlur(arr, self.kernel_size)
        return Image.fromarray(out)


class BilateralFiltro:
    """Suavização com preservação de bordas."""
    nome = "bilateral"

    def __init__(self, diameter=7, sigma_color=50.0, sigma_space=50.0):
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def aplicar(self, img):
        arr = np.asarray(img.convert("RGB"))
        out = cv2.bilateralFilter(arr, self.diameter, self.sigma_color, self.sigma_space)
        return Image.fromarray(out)


def obter_filtro(nome, args):
    if nome == "sem_filtro":
        return SemFiltro()
    if nome == "realesrgan":
        return RealESRGANFiltro(args.realesrgan_weights, args.realesrgan_outscale,
                                args.realesrgan_tile, args.device)
    if nome == "median":
        return MedianFiltro(args.median_kernel)
    if nome == "bilateral":
        return BilateralFiltro(args.bilateral_diameter, args.bilateral_sigma_color,
                               args.bilateral_sigma_space)
    raise ValueError(f"Filtro desconhecido: {nome}")