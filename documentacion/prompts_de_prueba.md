# Batería de Pruebas: Sistema de Agentes Industriales (End-to-End)

Este documento contiene un conjunto de **Prompts de Pruebas** diseñados para evaluar exhaustivamente cada capacidad del sistema (RAG, MCP, Computer Use, Histórico y Ruteo Dinámico). 

Están redactados desde la perspectiva de un *usuario final realista* (un operador de planta, supervisor o ingeniero de confiabilidad) que desconoce cómo funciona la IA por dentro, mezclando preguntas coloquiales con solicitudes operativas.

---

## 1. Pruebas de MCP (Lecturas Dinámicas / SCADA en tiempo real)
*Estas pruebas evalúan si el agente es capaz de encontrar la herramienta correcta (`call_dynamic_mcp`) y usar los filtros inteligentes (`key_values`, `key_figures`) basándose en los datos simulados de la API.*

### Escenario 1.1: Consulta básica de estado
> **Prompt:** "Oye, ¿me puedes dar un resumen rápido de cómo está operando la maquinaria crítica en este momento? Especialmente revisa el Motor 1 y la Bomba A."
* **Objetivo:** Disparar el endpoint de Maquinaria sin necesidad de filtrar mucho.
* **Comportamiento Esperado:** El `mcp-orchestrator` entra en acción, llama al recurso de maquinaria y lista temperatura y vibración del Motor1, y la presión de la Bomba A.

### Escenario 1.2: Filtrado Inteligente por Valores (KVs)
> **Prompt:** "Necesito saber si hay algo en el sistema que esté marcando el estado de Revisión o Warning. Solo muéstrame esos reportes, no quiero ver alarmas sanas."
* **Objetivo:** Probar el `key_values_filter`.
* **Comportamiento Esperado:** Llamada a Manufactura filtrando automáticamente por `Status: Review` o `Quality: Warning`. Debería reportar únicamente los 5 defectos en la Línea A.

### Escenario 1.3: Lectura Medioambiental Continua
> **Prompt:** "Voy a mandar a una cuadrilla a trabajar en la Zona 1 del galpón. ¿Cómo están las condiciones climáticas ahí respecto a temperatura y humedad ahora mismo?"
* **Objetivo:** Evaluar si sabe enrutar correctamente a la colección/endpoint de medio ambiente en lugar de maquinaria o manufactura.

---

## 2. Pruebas de RAG (Investigador de Conocimiento Institucional)
*Estas pruebas asumen que los PDFs creados por `generate_samples.py` ya fueron ingeridos en una base de conocimientos. Evalúa el RAG Híbrido + Reranker.*

### Escenario 2.1: Procedimientos de Seguridad
> **Prompt:** "Necesito hacerle un mantenimiento rápido al Motor 1. Antes de ir para allá, ¿me recuerdas qué equipo de protección necesito usar y qué es lo primero que debo hacer según el manual antes de intervenirlo?"
* **Objetivo:** Extraer datos exactos de un manual de usuario (`knowledge-researcher`).
* **Comportamiento Esperado:** Buscará semánticamente "mantenimiento Motor 1 equipos de protección" y devolverá la sección 2 del PDF: uso de Casco, Guantes, Protección auditiva y verificar que el equipo no esté "Running".

### Escenario 2.2: Consideraciones Técnicas Complejas
> **Prompt:** "La última vez que paramos la bomba tuvimos problemas físicos en el arranque. Según los manuales, ¿qué tengo que tener en cuenta sobre la condensación o puntos de rocío cuando arranca en frío?"
* **Objetivo:** Búsqueda y comprensión contextual de términos técnicos de las "Consideraciones Críticas".

---

## 3. La Prueba de Fuego: Cross-Tooling (RAG + MCP)
*El nivel más alto de evaluación. Pone a prueba la capacidad del Orquestador de invocar de manera paralela o secuencial la inteligencia en vivo y la documental.*

### Escenario 3.1: Validación de Operación vs Límite Teórico
> **Prompt:** "Por favor, revisa la presión que está marcando la Bomba A ahora mismo en la línea de maquinaria. Luego, busca en su manual de operación y compárame si ese valor de presión actual se encuentra en un estado nominal o si ya entró en algún umbral de alerta o crítico."
* **Objetivo:**
  1. Que el analista MCP encuentre que la presión viva es `120.2`.
  2. Que el analista RAG busque las tablas de límites operacionales para la presión en el manual.
  3. Que el orquestador general cruce los datos: si el nominal en el manual era `108` y la alerta `138` (basado en la lógica del script Python del usuario), debe concluir que `120.2` está por encima del nominal pero aún no en alerta.

---

## 4. Pruebas del Sistema 1 (S1-Histórico)
*Evaluación pura de los pesos neuronales estáticos de los modelos LoRA (sin uso de RAG ni SCADA).*

### Escenario 4.1: Interrogatorio Directo a la Memoria
> **Prompt:** "Viene el gerente de arriba y quiere un reporte general. Del último año completo que tengamos registro (como 2023), ¿cuáles recuerdas que fueron las tres fallas que más nos causaron de cabeza en toda la planta?"
* **Objetivo:** Disparar al `sistema1-historico`. Debe ser lo suficientemente general para que el orquestador no intente ni RAG ni MCP.
* **Comportamiento Esperado:** Responde directamente desde su entrenamiento que las fallas fueron (1) Sellos mecánicos, (2) Incrustaciones en válvulas, (3) Disparos por temperatura.

### Escenario 4.2: Prueba de Limitación Negativa
> **Prompt:** "Haciendo memoria de los datos históricos, ¿cuál fue exactamente el consumo eléctrico de la línea de embotellado el 14 de marzo de 2023 a las 14:00?"
* **Objetivo:** Probar que el modelo histórico sabe decir "No sé" (según sus reglas en `prompts/system1_historico.py`) en lugar de alucinar cifras hiper específicas ausentes en su LoRA de entrenamiento.

---

## 5. Pruebas del Sistema 1 VL (Digital Operator / Computer Use)
*Estas pruebas deben ejecutarse en un entorno donde Ghost/Xvfb (Ubuntu) o Docker Display esté activo, evaluando la automatización de la interfaz gráfica local del servidor.*

### Escenario 5.1: Navegación simple e Interacción
> **Prompt:** "Necesito buscar un repuesto rápido. Abre un navegador, busca en Google 'sellos mecánicos industriales SKF' y dime qué dice el primer resultado que encuentres."
* **Objetivo:** Ejecutar la secuencia: abrir chromium, buscar texto (con `navigate_action`), capturar la pantalla post-render y leer visualmente la información, decidiéndo si completar la tarea o explorar.

### Escenario 5.2: Simulación de Transacción SAP Externa
> **Prompt:** "Ve al portal web de SAP de la empresa (o entra a www.sap.com si no lo tenemos), fíjate en el menú de la esquina superior derecha e intenta hacer clic donde dice iniciar sesión. Sólo dime de qué color está ese botón y la forma que tiene."
* **Objetivo:** Probar el parser multimodal (`OmniParser V2`). Debe tomar captura de pantalla, reconocer coordenadas visualmente y mapearlas a `click [X, Y]` correctamente.

---

**Nota para QA:** Ejecuta estos prompts uno a la vez en una ventana de chat en limpio. Especialmente en la Parte 3, observa en la consola si el orquestador logra llamar con éxito las *schemas* en tiempo real antes de contestar.
