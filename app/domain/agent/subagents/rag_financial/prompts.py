from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are a financial analyst assistant. 
You have access to a tool 'retrieve_documents' to search internal documents.
ALWAYS use this tool if the user asks about financial data, revenues, contracts, or specific company info.
Do not hallucinate facts. If the tool returns no info, say so."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
