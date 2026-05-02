"""
OmniParser V2 Service — Set-of-Marks (SoM) Grounding
=====================================================
Wrapper alrededor de OmniParser V2 (Microsoft) para enriquecer screenshots
con bounding boxes numerados, transformando el problema de "adivinar coordenadas"
en "seleccionar elemento de una lista numerada".

Flujo con OmniParser ACTIVO:
  screenshot_raw
    │
    ▼  OmniParserService.parse()
  ┌─ icon_detect (YOLO) → bounding boxes de elementos interactivos
  ├─ icon_caption (Florence-2) → descripción funcional de cada elemento
  └─ annotated_screenshot → imagen con etiquetas numéricas encima de cada elemento

  VLM recibe: annotated_screenshot + lista textual:
    "[1] Button 'Buscar' (verde, esquina superior derecha)"
    "[2] Input 'Email' (campo vacío)"
    "[3] Link 'Olvidé mi contraseña'"

  VLM responde: "click element #3"  ← selección, no predicción de coords

Flujo con OmniParser INACTIVO (fallback):
  screenshot con cuadrícula de coordenadas (comportamiento original)

Activar:
  1. Descargar pesos: huggingface-cli download microsoft/OmniParser-v2.0
  2. Colocar en /omniparser/weights/ (icon_detect/ + icon_caption_florence/)
  3. Setear OMNIPARSER_ENABLED=true en .env

Referencia: https://github.com/microsoft/OmniParser
"""

import asyncio
import base64
import io
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from loguru import logger


@dataclass
class ParsedElement:
    """Un elemento UI detectado por OmniParser."""
    id: int
    label: str              # Descripción funcional del elemento (Florence-2)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) en coordenadas de imagen
    center_x: int           # Centro X en coordenadas de imagen
    center_y: int           # Centro Y en coordenadas de imagen
    interactable: bool = True


@dataclass
class OmniParserResult:
    """Resultado del parsing de un screenshot."""
    elements: List[ParsedElement] = field(default_factory=list)
    annotated_b64: str = ""          # Screenshot con etiquetas SoM en base64
    element_list_text: str = ""      # Texto formateado para el prompt del VLM
    raw_screenshot_b64: str = ""     # Screenshot original sin anotaciones

    def get_element_by_id(self, element_id: int) -> Optional[ParsedElement]:
        for el in self.elements:
            if el.id == element_id:
                return el
        return None

    def resolve_click_coords(self, element_id: int) -> Optional[Tuple[int, int]]:
        """Resuelve un ID de elemento a coordenadas (x, y) en imagen space."""
        el = self.get_element_by_id(element_id)
        if el:
            return (el.center_x, el.center_y)
        return None


class OmniParserService:
    """
    Singleton lazy-init para OmniParser V2.

    Carga los modelos solo cuando se necesitan por primera vez.
    Si los pesos no están disponibles, las llamadas retornan None silenciosamente.
    """

    _instance: Optional["OmniParserService"] = None
    _initialized: bool = False
    _available: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _try_init(self, model_dir: str) -> bool:
        """Intenta inicializar los modelos OmniParser. Retorna True si exitoso."""
        if self._initialized:
            return self._available

        self._initialized = True

        try:
            import torch
            from ultralytics import YOLO
            from transformers import AutoProcessor, AutoModelForCausalLM

            icon_detect_path = os.path.join(model_dir, "icon_detect", "model.pt")
            icon_caption_path = os.path.join(model_dir, "icon_caption_florence")

            if not os.path.exists(icon_detect_path) or not os.path.isdir(icon_caption_path):
                logger.info(
                    f"[OmniParser] Pesos no encontrados en {model_dir}. "
                    "Descarga en progreso (background) — usando cuadrícula de coordenadas por ahora."
                )
                self._available = False
                return False

            logger.info("[OmniParser] Cargando modelos...")
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._yolo = YOLO(icon_detect_path)
            self._yolo.to(self._device)

            self._caption_processor = AutoProcessor.from_pretrained(
                icon_caption_path, trust_remote_code=True, local_files_only=True
            )
            self._caption_model = AutoModelForCausalLM.from_pretrained(
                icon_caption_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            ).to(self._device)

            self._available = True
            if self._device == "cpu":
                logger.warning(
                    "[OmniParser] ⚠️ Running on CPU — Florence-2 captioning is very slow "
                    "(~2-5s per element). With 10-20 elements per screenshot, each parse "
                    "may take 20-100s. Consider enabling GPU (--gpus all in Docker) "
                    "or disabling OmniParser (omniparser_enabled=False) and using the "
                    "coordinate grid fallback instead."
                )
            logger.success("[OmniParser] ✓ Modelos cargados correctamente.")
            return True

        except ImportError as e:
            logger.warning(
                f"[OmniParser] Dependencias no instaladas ({e}). "
                "Instalar: pip install ultralytics transformers. "
                "Fallback: cuadrícula de coordenadas."
            )
            self._available = False
            return False
        except Exception as e:
            logger.error(f"[OmniParser] Error al inicializar: {e}")
            self._available = False
            return False

    def is_available(self, model_dir: str) -> bool:
        return self._try_init(model_dir)

    def _caption_element(self, crop_b64: str) -> str:
        """Genera descripción funcional de un crop de elemento."""
        import time
        t0 = time.time()
        try:
            import torch
            from PIL import Image

            img_bytes = base64.b64decode(crop_b64)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            inputs = self._caption_processor(
                text="<CAPTION>",
                images=image,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._caption_model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )
            caption = self._caption_processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0].strip()
            elapsed = time.time() - t0
            if elapsed > 1.0:
                logger.debug(f"[OmniParser] Caption took {elapsed:.2f}s (device={self._device}) — slow CPU detected")
            return caption or "UI element"
        except Exception as e:
            logger.debug(f"[OmniParser] Error captioning element: {e}")
            return "UI element"

    async def ensure_weights(self, model_dir: str) -> bool:
        """
        Background task: descarga microsoft/OmniParser-v2.0 si los pesos no existen.
        Llamar una vez desde el lifespan de FastAPI con asyncio.create_task().
        No bloquea el event loop — usa run_in_executor.
        Cuando termina, resetea _initialized para que el próximo is_available() cargue los modelos.

        Returns True si los pesos existen o fueron descargados exitosamente.
        """
        icon_detect_path = os.path.join(model_dir, "icon_detect", "model.pt")
        if os.path.exists(icon_detect_path):
            logger.info(f"[OmniParser] Pesos ya existen en {model_dir}. Download skipped.")
            return True

        # Ensure directory exists with write permissions
        os.makedirs(model_dir, exist_ok=True)

        def _download():
            import time
            from huggingface_hub import snapshot_download

            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            if not hf_token:
                logger.warning(
                    "[OmniParser] HF_TOKEN no configurado. "
                    "Descarga pública requerida. Si el modelo es gated, la descarga fallará. "
                    "Configura HF_TOKEN en .env o como variable de entorno."
                )

            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"[OmniParser] Descargando microsoft/OmniParser-v2.0 (intent {attempt}/{max_retries})...")
                    snapshot_download(
                        repo_id="microsoft/OmniParser-v2.0",
                        local_dir=model_dir,
                        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
                        token=hf_token,
                    )
                    logger.success(f"[OmniParser] ✓ Pesos descargados en {model_dir}. Se cargarán en el próximo paso.")
                    # Reset so the next is_available() call loads the models
                    self._initialized = False
                    self._available = False
                    return True
                except Exception as e:
                    logger.error(f"[OmniParser] Error descarga intent {attempt}/{max_retries}: {e}")
                    if attempt < max_retries:
                        wait_sec = 2 ** attempt  # exponential backoff: 2, 4, 8s
                        logger.info(f"[OmniParser] Reintentando en {wait_sec}s...")
                        time.sleep(wait_sec)
                    else:
                        logger.error(
                            "[OmniParser] Falló descarga después de 3 intentos. "
                            "OmniParser permanecerá desactivado. Fallback: cuadrícula de coordenadas."
                        )
                        return False

        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(None, _download)
        return success

    def _parse_sync(self, screenshot_b64: str) -> OmniParserResult:
        """Síncrono — ejecutar via run_in_executor."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np

            img_bytes = base64.b64decode(screenshot_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img)
            w, h = img.size

            results = self._yolo.predict(img_np, conf=0.25, iou=0.45, verbose=False)
            boxes = results[0].boxes if results else None

            elements: List[ParsedElement] = []
            draw_img = img.copy()
            draw = ImageDraw.Draw(draw_img)

            # Try system fonts first, then fallback to PIL default
            def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
                font_paths = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "/System/Library/Fonts/Helvetica.ttc",  # macOS
                    "C:/Windows/Fonts/arial.ttf",           # Windows
                ]
                for fp in font_paths:
                    if os.path.exists(fp):
                        try:
                            return ImageFont.truetype(fp, size)
                        except Exception:
                            continue
                return ImageFont.load_default()

            try:
                font = _load_font(14)
                font_small = _load_font(11)
            except Exception:
                font = ImageFont.load_default()
                font_small = font

            colors = [
                "#FF4B4B", "#4B9EFF", "#4BFF6B", "#FFB84B",
                "#B84BFF", "#FF4BCF", "#4BFFF0", "#FF8C4B",
            ]

            if boxes is not None:
                _cpu_slow = self._device != "cuda"
                _boxes_list = boxes.xyxy.cpu().numpy()
                _total_elements = len(_boxes_list)
                if _cpu_slow and _total_elements > 8:
                    # CPU only: cap at 8 elements to avoid 20-40s hangs per screenshot
                    logger.warning(
                        f"[OmniParser] Detected {_total_elements} elements — capping to 8 on CPU "
                        f"to keep latency reasonable. GPU would process all {_total_elements}."
                    )
                    _boxes_list = _boxes_list[:8]
                for idx, box in enumerate(_boxes_list):
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    element_id = idx + 1
                    color = colors[idx % len(colors)]

                    crop = img.crop((x1, y1, x2, y2))
                    crop_buf = io.BytesIO()
                    crop.save(crop_buf, format="PNG")
                    crop_b64 = base64.b64encode(crop_buf.getvalue()).decode()
                    caption = self._caption_element(crop_b64)

                    elements.append(ParsedElement(
                        id=element_id,
                        label=caption,
                        bbox=(x1, y1, x2, y2),
                        center_x=cx,
                        center_y=cy,
                        interactable=True,
                    ))

                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    badge_x, badge_y = x1, max(0, y1 - 18)
                    badge_w = len(str(element_id)) * 9 + 6
                    draw.rectangle(
                        [badge_x, badge_y, badge_x + badge_w, badge_y + 18],
                        fill=color,
                    )
                    draw.text((badge_x + 3, badge_y + 2), str(element_id), fill="white", font=font)

            buf = io.BytesIO()
            draw_img.save(buf, format="PNG", optimize=True)
            annotated_b64 = base64.b64encode(buf.getvalue()).decode()

            element_lines = []
            for el in elements:
                element_lines.append(f"[{el.id}] {el.label} — center({el.center_x},{el.center_y})")
            element_list_text = "\n".join(element_lines) if element_lines else "(No interactable elements detected)"

            logger.info(f"[OmniParser] Parsed {len(elements)} elements from screenshot.")
            return OmniParserResult(
                elements=elements,
                annotated_b64=annotated_b64,
                element_list_text=element_list_text,
                raw_screenshot_b64=screenshot_b64,
            )

        except Exception as e:
            logger.error(f"[OmniParser] Error parsing screenshot: {e}")
            return OmniParserResult(raw_screenshot_b64=screenshot_b64)

    async def parse(self, screenshot_b64: str) -> OmniParserResult:
        """Async: parsea el screenshot y retorna elementos numerados + imagen anotada."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._parse_sync, screenshot_b64)


_omniparser_service = OmniParserService()


def get_omniparser() -> OmniParserService:
    """Retorna la instancia singleton de OmniParserService."""
    return _omniparser_service
