# Guía de Integración Industrial: MCP + RAG

Este documento detalla las formas de interactuar con el ecosistema de herramientas del Backend, permitiendo cruzar información estática (documentos, manuales) con información viva (telemetría, APIs externas).

---

## 1. Llamadas al MCP (Model Context Protocol)

El MCP utiliza la herramienta `call_dynamic_mcp` para conectarse a cualquier recurso registrado. Soporta filtrado inteligente para optimizar el uso de tokens.

### A. Llamada Simple (Full Scan)
Se solicita toda la información disponible de una fuente.
```json
{
  "tool": "call_dynamic_mcp",
  "parameters": {
    "tool_config_name": "get_maquinaria",
    "arguments": {}
  }
}
```

### B. Filtrado Categórico (`key_values`)
Ideal para filtrar por estados, categorías, nombres o departamentos.
```json
{
  "tool": "call_dynamic_mcp",
  "parameters": {
    "tool_config_name": "get_maquinaria",
    "arguments": {
      "key_values": {
        "Status": ["Running", "Warning"],
        "Category": ["Thermal"]
      }
    }
  }
}
```

### C. Filtrado Numérico (`key_figures`)
Permite establecer rangos técnicos para identificar anomalías.
```json
{
  "tool": "call_dynamic_mcp",
  "parameters": {
    "tool_config_name": "get_medio_ambiente",
    "arguments": {
      "key_figures": [
        { "field": "Value", "min": 25.0, "max": 40.0 }
      ]
    }
  }
}
```

---

## 2. Llamadas al RAG (Knowledge Base)

El RAG utiliza la herramienta `ask_knowledge_agent` para buscar en la base de conocimientos vectorial (manuales, PDFs, reportes).

```json
{
  "tool": "ask_knowledge_agent",
  "parameters": {
    "query": "¿Cuál es el procedimiento de mantenimiento para el Motor1 cuando supera los 80°C?"
  }
}
```

---

## 3. Ejemplos de Consultas Cruzadas (Power User)

El verdadero potencial surge cuando el agente utiliza ambas herramientas para resolver problemas complejos.

### Escenario I: Diagnóstico de Alerta en Tiempo Real
**Pregunta:** "El sensor de temperatura marca 85°C. ¿Es peligroso según el manual y qué debo hacer?"
1.  **MCP**: Obtiene el valor real y metadatos.
    `call_dynamic_mcp(tool_config_name="get_maquinaria", arguments={"key_values": {"TagName": ["Motor1"]}})`
2.  **RAG**: Busca los límites en el manual técnico.
    `ask_knowledge_agent(query="Límites de temperatura y protocolos de emergencia para Motor1")`
3.  **Resultado**: El agente cruza el valor vivo (85°C) con el límite del manual (80°C) y emite una recomendación de apagado.

### Escenario II: Planificación de Mantenimiento Preventivo
**Pregunta:** "¿Qué motores necesitan mantenimiento esta semana basándome en su última revisión y su vibración actual?"
1.  **RAG**: Busca los reportes de mantenimiento del mes pasado.
    `ask_knowledge_agent(query="Últimas fechas de mantenimiento de motores linea A")`
2.  **MCP**: Consulta la vibración actual de esos motores.
    `call_dynamic_mcp(tool_config_name="get_maquinaria", arguments={"key_values": {"Category": ["Mechanical"]}, "key_figures": [{"field": "Value", "min": 0.08}]})`
3.  **Resultado**: El agente identifica motores que no han sido revisados y que presentan vibraciones fuera de rango.

### Escenario III: Auditoría de Repuestos
**Pregunta:** "Tengo una lectura de presión baja en la Bomba A. Revisa el manual de despiece y dime qué repuesto necesito."
1.  **MCP**: Confirma la presión actual.
    `call_dynamic_mcp(tool_config_name="get_maquinaria", ...)`
2.  **RAG**: Busca en el manual de partes.
    `ask_knowledge_agent(query="Manual de despiece y códigos de repuestos para Bomba A")`
3.  **Resultado**: El agente vincula la falla de presión con el posible sello mecánico dañado y proporciona el código de parte del manual.

---

## Reglas de Oro para el Agente
- **Filtrar Primero**: Siempre usa `key_values` o `key_figures` en MCP para evitar el despliegue de datos masivos.
- **Contexto RAG**: Si una consulta de MCP devuelve un error o un estado desconocido, busca en el RAG para entender el significado de ese estado.
- **Sin Alucinaciones**: Si el MCP no devuelve datos, no los inventes; informa que el sensor podría estar offline y sugiere revisar el RAG para ver dónde está ubicado físicamente el equipo.
