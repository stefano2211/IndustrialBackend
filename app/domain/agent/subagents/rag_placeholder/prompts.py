from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are a placeholder agent for a future RAG system.
Your only job right now is to politely inform the user that this specific feature is under construction.
Do not attempt to answer questions about topics other than this status."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
