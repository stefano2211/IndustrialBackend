import urllib.request
import urllib.error
import json
import time
import sys

BACKEND_HOST = "http://185.216.20.101:8000"
BACKEND_LOGIN_URL = f"{BACKEND_HOST}/auth/login"
BACKEND_REGISTER_URL = f"{BACKEND_HOST}/auth/register"
BACKEND_SOURCES_URL = f"{BACKEND_HOST}/db-collector/sources"

def get_token(email, password):
    print(">> Autenticando...")
    req = urllib.request.Request(
        BACKEND_LOGIN_URL,
        data=json.dumps({"email": email, "password": password}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get("access_token")
    except urllib.error.URLError as e:
        if hasattr(e, 'code') and e.code == 400:
            print(">> Usuario no encontrado. Intentando registrarlo automáticamente...")
            return register_and_get_token(email, password)
        else:
            print(f"Error al autenticar: {e}")
            sys.exit(1)

def register_and_get_token(email, password):
    req = urllib.request.Request(
        BACKEND_REGISTER_URL,
        data=json.dumps({
            "email": email,
            "username": email.split('@')[0],
            "password": password, 
            "full_name": "Admin Nube", 
            "is_superuser": True
        }).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req) as response:
            print(">> ¡Usuario registrado con éxito como Administrador!")
            # Ya que se creó, volvemos a iniciar sesión normal para sacar el token
            return get_token(email, password)
    except urllib.error.URLError as e:
        print(f"❌ Error al registrar: {e}")
        if hasattr(e, 'read'): print(e.read().decode('utf-8'))
        sys.exit(1)

def run_test():
    print("Iniciando prueba End-to-End en la Nube (IndustrialBackend -> ApiLLMOps)")
    print("-" * 60)
    
    email = input("Email de Administrador: ")
    password = input("Password: ")
    
    token = get_token(email, password)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    # 1. Registrar la fuente de datos que apunta a la ApiEjemplo (corriendo en el host 172.17.0.1)
    payload = {
        "name": "Sensores_ServerA_Prueba",
        "description": "Base de datos automatizada local via SSH endpoint",
        "db_type": "postgresql",
        "connection_string": "postgresql://postgres:postgres@172.17.0.1:5433/industrial_db",
        "query": "SELECT tag_name as feature, value, timestamp, quality as label FROM measurements WHERE department = 'Maquinaria'",
        "sector": "manufactura",
        "domain": "telemetria",
        "auto_trigger_enabled": True,
        "rows_threshold": 3,
        "cron_expression": "0 * * * *"  # irrelevante para prueba manual
    }

    try:
        print(">> Paso 1: Registrando fuente de datos en el Backend...")
        req = urllib.request.Request(
            BACKEND_SOURCES_URL, 
            data=json.dumps(payload).encode('utf-8'),
            headers=headers
        )
        with urllib.request.urlopen(req) as response:
            source_data = json.loads(response.read().decode('utf-8'))
            source_id = source_data["id"]
            print(f"Fuente registrada exitosamente con ID: {source_id}")
    except urllib.error.URLError as e:
        print(f"Error al registrar la fuente: {e}")
        if hasattr(e, 'read'):
            print(f"Detalle del error: {e.read().decode('utf-8')}")
        sys.exit(1)

    time.sleep(1)

    # 2. Forzar la recolección y sincronización
    sync_url = f"{BACKEND_SOURCES_URL}/{source_id}/run"
    
    try:
        print("\n>> Paso 2: Ejecutando recolección de datos y envío a Mothership...")
        print(f"Haciendo POST a: {sync_url}")
        
        req = urllib.request.Request(
            sync_url, 
            data=b'', 
            headers=headers
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            sync_result = json.loads(response.read().decode('utf-8'))
            print(f"Sincronizacion lanzada con exito!")
            print(f"Detalle del proceso: {sync_result}")
            
    except urllib.error.URLError as e:
        print(f"\nError al sincronizar la fuente: {e}")
        if hasattr(e, 'read'):
             print(f"Detalle del error: {e.read().decode('utf-8')}")
        sys.exit(1)

    # Damos tiempo a que se suba a MinIO en el background
    time.sleep(2)

    # 3. Disparar el Entrenamiento
    BACKEND_TRAINING_URL = f"{BACKEND_HOST}/mlops/training/launch"
    training_payload = {
        "tenant_id": "aura_tenant_01",
        "epochs": 3
    }

    try:
        print("\n>> Paso 3: Disparando el Entrenamiento en ApiLLMOps...")
        req = urllib.request.Request(
            BACKEND_TRAINING_URL,
            data=json.dumps(training_payload).encode('utf-8'),
            headers=headers
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            train_result = json.loads(response.read().decode('utf-8'))
            print("✅ ¡Entrenamiento disparado en la Mothership exitosamente!")
            
            print("\n" + "=" * 60)
            print("PRUEBA ENVIADA CORRECTAMENTE")
            print("=" * 60)
            print("\nSiguientes pasos para revisar en los servidores:")
            print("1. ¡Revisa los logs del SERVER B (ApiLLMOps)! Debería estar entrenando a Unsloth en este momento.")
            print("2. Cuando Server B termine, verás un POST de vuelta al Server A avisando que un nuevo LoRA está listo.")
    except urllib.error.URLError as e:
        print(f"\n❌ Error al disparar entrenamiento: {e}")
        if hasattr(e, 'read'): print(f"Detalle del error: {e.read().decode('utf-8')}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
