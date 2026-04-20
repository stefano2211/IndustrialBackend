# Whitepaper Técnico: Inyección de Contexto Corporativo mediante Fine-Tuning

**Fecha:** Abril de 2026  
**Asunto:** Transición de la "Sobrecarga de la Ventana de Contexto" hacia la "Inyección de Conocimiento Paramétrico" para entornos empresariales estables.

## 1. Resumen Ejecutivo

En los despliegues de Inteligencia Artificial empresarial (específicamente en la IA Industrial de Borde o *Edge AI*), los modelos requieren de un profundo conocimiento de fondo para operar de forma segura y efectiva. Históricamente, este problema se ha resuelto inyectando el Contexto Corporativo (diccionarios técnicos, esquemas de maquinaria, reglas de operación, jerga del sector) directamente en el Prompt del Sistema o mediante la constante recuperación de datos (RAG).

Este documento expone los problemas arquitectónicos asociados con ese enfoque tradicional y establece la justificación técnica para migrar el "Sistema 1" hacia la **Inyección Paramétrica de Contexto mediante Fine-Tuning Supervisado (SFT / QLoRA)**.

---

## 2. Los Problemas de "Sobrecargar" la Ventana de Contexto (Context Window Stuffing)

Intentar alimentar a un modelo de lenguaje (LLM) con la estructura corporativa completa en tiempo real en cada consulta introduce varios puntos críticos en el entorno empresarial:

### 2.1 El Fenómeno de "Perdido en el Medio" (Lost in the Middle)
Diversas investigaciones confirman que a medida que las ventanas de contexto superan los 32.000 tokens, los LLMs sufren una degradación en su capacidad de recordar la información ubicada en el medio del texto enviado. Si el Procedimiento de Seguridad #4 queda sepultado en la línea 1.200 del prompt, es altamente probable que el modelo lo ignore durante tareas de razonamiento complejo.

### 2.2 Latencia Exponencial (Disparo del TTFT)
Antes de que un LLM pueda generar la primera palabra de una respuesta, su mecanismo de atención debe computar la memoria caché Clave-Valor (KV Cache) para todo el prompt de entrada.
- Un prompt de 500 tokens se procesa casi instantáneamente en el hardware local.
- Un prompt que contiene 15 páginas de jerga corporativa y esquemas de bases de datos (aprox. 20.000 tokens) provoca un pico severo en el **Tiempo Hasta el Primer Token (TTFT - Time To First Token)**, aumentando la latencia en varios segundos y paralizando la experiencia de usuario (UX) en tiempo real.

### 2.3 Limitaciones Financieras y de Cómputo (Quema de Tokens)
Ya sea utilizando proveedores en la nube o hardware local (vLLM con VRAM restringida de 24GB/48GB), pasar el mismo conocimiento de fondo estático en *cada consulta única* provoca un desperdicio de cómputo inaceptable. Procesar el mismo diccionario corporativo de 10.000 tokens mil veces al día limita drásticamente la concurrencia (capacidad de atención a múltiples usuarios) y desperdicia los ciclos de la GPU que deberían dedicarse al razonamiento puro de los agentes.

### 2.4 Mantenimiento de Prompts ("Prompts Espagueti")
A medida que diferentes departamentos (ej. Calidad, Logística, Mantenimiento) exigen sus propias reglas, los desarrolladores acaban con meta-prompts inmanejables que contienen instrucciones contradictorias ("Usa el sistema métrico", "Saca metros pero usa imperial para el Área 5", "Siempre alerta sobre la presión"). Esto conduce a un formateo inestable y regresiones sistémicas frecuentes que son una pesadilla de depurar.

---

## 3. La Solución: Ingeniería de Contexto y RAFT

**RAFT (Retrieval-Augmented Fine-Tuning)** propone una clara separación de responsabilidades dentro de la arquitectura de IA Industrial:

| Tipo de Conocimiento | Características | Solución Arquitectónica |
| :--- | :--- | :--- |
| **Contexto Corporativo** | Estable, Jerga, Procedimientos (SOPs), Estructura, Cultura. | **Fine-Tuning (Sistema 1)** |
| **Datos Operativos / Hechos** | Dinámicos, Datos de Sensores en Vivo, Registros Diarios, Alertas. | **RAG / Webhooks MCP (Sistema 2)**|

### 3.1 Inyección de Conocimiento Paramétrico (El Pipeline de Fine-Tuning)
Utilizando tu canalización MLOps ya existente en la *Mothership* (`unsloth_trainer`, QLoRA, bfloat16), se puede "hornear" o incrustar este contexto corporativo directamente en los pesos neuronales de Qwen 3.5.

**Cambio en la Preparación de Datos:**
En lugar de extraer registros numéricos en bruto directamente de la Base de Datos Histórica para afinar el modelo, el `DbCollector` debe compilar:
1. **Diccionarios de Datos:** "Cuando un usuario pregunte por el 'Enfriador 1', debes mapearlo lógicamente al `sensor_id_892`."
2. **Reglas de Comportamiento:** "Siempre formatea los informes de mantenimiento con los indicadores de severidad en la parte superior del mensaje."
3. **Vocabulario de Dominio:** "MB51 es una transacción estandarizada de SAP utilizada para revisar documentos de material."

**Impacto Técnico Inmediato:**
- El modelo entiende *de forma nativa* el entorno industrial de tu fábrica sin que el usuario ni el desarrollador tengan que enviarle el manual entero de fondo.
- Los Prompts de Sistema (System Prompts) se reducen de más de 5.000 tokens a menos de 500 tokens.
- Memoria de Cache KV es liberada radicalmente, permitiendo una concurrencia astronómicamente mayor y velocidad de respuesta al instante.

---

## 4. Arquitectura de Migración para el `Sistema 1 Histórico`

Para transicionar el sistema actual a este nuevo estándar arquitectónico, se deben implementar los siguientes ajustes:

> [!IMPORTANT]
> Detén el fine-tuning del modelo basado en los datos numéricos o métricas de historial efímero. Haz la transición para que los datasets de fine-tuning contengan estrictamente **Reglas Semánticas, Esquemas Arquitectónicos y Procedimientos Operativos Estandarizados (SOPs)** corporativos.

### Implementación Paso a Paso:
1. **Curación del Dataset de "Contexto":** Selecciona y crea un dataset masivo en formato JSONL con instrucciones-respuestas que le enseñen a la IA *cómo* interpretar los términos industriales y los formatos de tu fábrica, en vez un dataset dictándole los valores de telemetría de una máquina en un día en especifico del mes pasado.
2. **Afinamiento del Modelo Base SFT:** Usa Celery y Unsloth para entrenar los parámetros del `aura_tenant_01-v2` con este manual de contexto.
3. **Mantén el Pipeline OTA:** Manten tu sistema de inyección sin tiempo de inactividad de las actualizaciones (`vLLM /load_lora_adapter`) funcionando exactamente igual. Esto sigue siendo una bestia de ingeniería excepcional.
4. **Actualiza la Orquestación de los Agentes:** Dentro de `orchestrator.py`, actualiza el `Sistema 1` actuando formalmente ahora como el Experto Procedimental e interaccional Offline. Dale de vuelta el permiso de **MCP Tools**, para que una vez que haya comprendido fluidamente y haya hecho sus conversiones mentales perfectas sobre un acrónimo o un área, active un request hiper-enfocado para pedirle al `Industrial Expert` esos *hechos transaccionales dinámicos/numéricos y voluminosos*.

### 5. Conclusión General
Trasladar el **Contexto Corporativo** del "Prompt Inyectado con pinzas" hacia los "Pesos Internos del Modelo" logra erradicar uno de los peores problemas de costo, de cuellos de botella de computo y de la degradación neuronal de memoria experimentados en el ecosistema Edge Industrial 2026. Elimina barreras de razonamiento limitante nativa de los modelos ajustados bajo 9B, reduce el gasto de cómputo marginal global de inferencia y entrega una infraestructura asombrosamente resiliente a las fallas generacionales del futuro.
