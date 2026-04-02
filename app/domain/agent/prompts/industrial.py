"""
Prompt content for the Industrial Safety Deep Agent.

Contains:
  - INDUSTRIAL_SYSTEM_PROMPT: The main system prompt that governs agent behavior.
  - AGENTS_MD_CONTENT: Persistent domain memory (loaded as /AGENTS.md in the VFS).
"""

INDUSTRIAL_SYSTEM_PROMPT = """\
Expert in Industrial Safety. 
1. Use `ask_knowledge_agent` for docs.
2. Use `call_dynamic_mcp` for sensors.
3. Cite sources. No data? Say so. Match user language.

{dynamic_tools_context}
"""


AGENTS_MD_CONTENT = """\
# Industrial Safety AI — Memory

## Domain
- Industrial document analysis system: OSHA, ISO, NOM regulations
- Users: engineers, supervisors, auditors, safety managers
- Typical documents: incident reports, manuals, audits, compliance
- Data sources: User PDFs (Qdrant), real-time APIs (MCP)

## Preferences
- Cite exact section/page of the found regulation

User Query: "{query}"

Analyze the timeframe requested by the user:
- If the query specifically asks for data strictly older than 6 months from the current date (e.g., "last year", "in 2024", "8 months ago"), return true.
- If the query asks for recent data, current state, general questions, or compares recent with old data, return false.

Respond ONLY with a valid JSON object in this exact format (no markdown or extra text):
{{"is_historical_only": true}} or {{"is_historical_only": false}}
"""
