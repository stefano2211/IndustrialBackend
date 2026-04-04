"""
Generador de Dataset Sintético VL (Computer Use)
=================================================
Este script genera un dataset sintético para probar el entrenamiento
del modelo Qwen2.5-VL en el pipeline de Macrohard (Digital Optimus Local).

Crea 3 imágenes (screenshots simulados de SAP GUI):
  1. SAP Home Screen
  2. Transaction MB51 Input Screen
  3. MB51 Results Screen

Y crea un dataset JSONL (data/vl_synthetic.jsonl) y lo envía a MinIO
a través de ApiLLMOps, para simular las interacciones.
"""

import os
import json
import base64
import io
import httpx
import asyncio
from datetime import datetime
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("FATAL: PIL (Pillow) no está instalado. Instalándolo temporalmente...")
    os.system("pip install Pillow")
    from PIL import Image, ImageDraw, ImageFont

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENS_DIR = os.path.join(BASE_DIR, "app", "static", "demo", "screens")
DATASET_PATH = os.path.join(BASE_DIR, "data", "vl_synthetic.jsonl")

# Configuraciones API
MOTHERSHIP_URL = "http://localhost:8001/api/v1/vl/upload"
TENANT_ID = "aura_tenant_01"
API_KEY = "default-mothership-secret-key"

os.makedirs(SCREENS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

# ── Generación de Screens ──────────────────────────────────────────────────

def draw_sap_window(title: str):
    """Crea una base de ventana SAP estilo gris industrial (Win95)."""
    img = Image.new("RGB", (1024, 768), color=(212, 208, 200))
    draw = ImageDraw.Draw(img)
    
    # Barra de título
    draw.rectangle([0, 0, 1024, 30], fill=(0, 0, 128))
    draw.text((10, 8), f"SAP Easy Access - {title}", fill=(255, 255, 255))
    
    # Menú bar
    draw.rectangle([0, 30, 1024, 55], fill=(236, 233, 216))
    draw.text((10, 38), "Menu  Edit  Favorites  Extras  System  Help", fill=(0, 0, 0))
    draw.line([0, 55, 1024, 55], fill=(160, 160, 160), width=1)
    
    # Header bar
    draw.rectangle([0, 56, 1024, 95], fill=(236, 233, 216))
    draw.rectangle([10, 65, 200, 85], fill=(255, 255, 255), outline=(120, 120, 120))
    draw.text((210, 68), "🔍 | 💾 ◀ ▶ ✖", fill=(0, 0, 0))
    draw.line([0, 95, 1024, 95], fill=(160, 160, 160), width=1)
    
    return img, draw

def generate_screen_1_home():
    img, draw = draw_sap_window("User Menu for Aura Agent")
    draw.text((15, 68), "MB51", fill=(0,0,0)) # Simula que el usuario tipeó el TCode
    
    # Contenido Home
    draw.text((50, 150), "> Favorites", fill=(0,0,0))
    draw.text((50, 180), "> SAP Menu", fill=(0,0,0))
    draw.text((70, 210),   " > Logistics", fill=(0,0,0))
    draw.text((70, 240),   " > Accounting", fill=(0,0,0))
    draw.text((70, 270),   " > Human Resources", fill=(0,0,0))
    
    path = os.path.join(SCREENS_DIR, "01_home.png")
    img.save(path)
    return path

def generate_screen_2_input():
    img, draw = draw_sap_window("Material Document List")
    draw.text((15, 68), "", fill=(0,0,0))
    
    # Formulario
    draw.text((50, 150), "Material:", fill=(0,0,0))
    draw.rectangle([200, 145, 500, 165], fill=(255, 255, 255), outline=(120, 120, 120))
    draw.text((210, 148), "CRUDE-100", fill=(0,0,0))
    
    draw.text((50, 190), "Plant:", fill=(0,0,0))
    draw.rectangle([200, 185, 300, 205], fill=(255, 255, 255), outline=(120, 120, 120))
    draw.text((210, 188), "1000", fill=(0,0,0))
    
    # Execute button (simulado icono reloj)
    draw.rectangle([50, 110, 150, 130], fill=(220, 220, 220), outline=(100, 100, 100))
    draw.text((65, 114), "⌚ Execute (F8)", fill=(0,0,0))
    
    path = os.path.join(SCREENS_DIR, "02_mb51_input.png")
    img.save(path)
    return path

def generate_screen_3_results():
    img, draw = draw_sap_window("Material Document List - Results")
    
    # Tabla
    draw.rectangle([40, 130, 980, 155], fill=(200, 200, 200), outline=(100, 100, 100))
    draw.text((50, 138), "Mat. Doc.", fill=(0,0,0))
    draw.text((150, 138), "Item", fill=(0,0,0))
    draw.text((220, 138), "Pstng Date", fill=(0,0,0))
    draw.text((320, 138), "Qty in Un.", fill=(0,0,0))
    draw.text((420, 138), "MvT", fill=(0,0,0))
    
    draw.text((50, 170), "4900012345", fill=(0,0,0))
    draw.text((150, 170), "0001", fill=(0,0,0))
    draw.text((220, 170), "2026-04-03", fill=(0,0,0))
    draw.text((320, 170), "12,800 BBL", fill=(0,0,0))
    draw.text((420, 170), "101", fill=(0,0,0))
    
    path = os.path.join(SCREENS_DIR, "03_mb51_results.png")
    img.save(path)
    return path

def get_b64(path):
    with open(path, "rb") as f:
        # Importante: devolver el data uri scheme (se maneja sin él para el dict interior)
        return base64.b64encode(f.read()).decode('utf-8')

# ── Generación de JSONL ────────────────────────────────────────────────────

def create_dataset_entries(paths):
    img1, img2, img3 = [get_b64(p) for p in paths]
    entries = []
    
    # Step 1: En Home, meter el comando y apretar enter
    entries.append({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Open transaction MB51 to check crude oil inventory"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"type":"type","text":"MB51"}'}]
            }
        ],
        "images": [img1]
    })
    
    # Step 2: En la pantalla de MB51 input, enviar Enter
    entries.append({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Transaction loaded. Press Enter to submit."}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"type":"press","key":"enter"}'}]
            }
        ],
        "images": [img1] # Ideal seria la tranny frame
    })

    # Step 3: En input, poner el material
    entries.append({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Enter CRUDE-100 in the material field"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"type":"click","x":250,"y":155}'}]
            }
        ],
        "images": [img2]
    })
    
    # Step 4: Y la cantidad/texto
    entries.append({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Typing material name"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"type":"type","text":"CRUDE-100"}'}]
            }
        ],
        "images": [img2]
    })
    
    # Step 5: Click execute (simulado con coordenadas)
    entries.append({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Click the Execute button or press F8"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"type":"click","x":100,"y":120}'}]
            }
        ],
        "images": [img2]
    })
    
    # Step 6: Task complete
    entries.append({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Extract the inventory results."}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"type":"task_complete","summary":"Found 12,800 BBL posted on 2026-04-03 for CRUDE-100."}'}]
            }
        ],
        "images": [img3]
    })

    # Guardar local
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Dataset guardado en {DATASET_PATH} con {len(entries)} steps.")

# ── Envío via Mothership API ───────────────────────────────────────────────

async def upload_dataset():
    print(f"Subiendo dataset a {MOTHERSHIP_URL} ...")
    async with httpx.AsyncClient() as client:
        with open(DATASET_PATH, "rb") as f:
            files = {"file": ("vl_synthetic.jsonl", f, "application/jsonlines")}
            data = {"tenant_id": TENANT_ID, "tool_name": "computer_use"}
            headers = {"x-api-key": API_KEY}
            
            try:
                # Ignorar timeouts porque el append en el edge es muy rápido
                res = await client.post(MOTHERSHIP_URL, files=files, data=data, headers=headers)
                if res.status_code == 200:
                    print(f"✅ ¡Éxito! Dataset VL enviado a ApiLLMOps. Objeto: {res.json().get('object')}")
                else:
                    print(f"❌ Error en upload: {res.status_code} - {res.text}")
            except Exception as e:
                print(f"❌ Excepción: {e}")

if __name__ == "__main__":
    print("1. Generando SAP GUI Mockup Screens...")
    p1 = generate_screen_1_home()
    p2 = generate_screen_2_input()
    p3 = generate_screen_3_results()
    
    print("2. Creando entradas JSONL para entrenamiento Qwen2.5-VL...")
    create_dataset_entries([p1, p2, p3])
    
    print("3. Subiendo a ApiLLMOps...")
    asyncio.run(upload_dataset())
    
    print("\nTodo listo. Puedes despachar el entrenamiento VL usando:")
    print(f"  curl -X POST http://localhost:8001/api/v1/vl/training/job \\")
    print(f"       -H 'x-api-key: {API_KEY}' -H 'Content-Type: application/json' \\")
    print(f"       -d '{{\"tenant_id\": \"{TENANT_ID}\", \"vl_epochs\": 2}}'")
