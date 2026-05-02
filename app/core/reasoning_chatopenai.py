"""
ReasoningChatOpenAI — ChatOpenAI subclass that preserves reasoning_content.

vLLM with --reasoning-parser qwen3 separates model output into:
  - reasoning_content: the <think>...</think> block (reasoning steps)
  - content: the final answer

LangChain's ChatOpenAI silently drops reasoning_content (known bugs:
langchain#31326, #35059, #35901). This subclass patches both directions:

  1. _create_chat_result(): captures reasoning_content from the raw API
     response and stores it in AIMessage.additional_kwargs["reasoning_content"]
  2. _convert_message_to_dict(): re-serializes reasoning_content when
     building outgoing assistant messages, which is required for correct
     multi-turn tool-calling with vLLM reasoning models.

Additionally, for streaming, the subclass overrides _stream() / _astream()
to capture reasoning tokens from delta.reasoning and yield them as
AIMessageChunks with additional_kwargs["reasoning_content"].
"""

from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from loguru import logger


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI that preserves vLLM reasoning_content as a first-class field."""

    # ── Non-streaming: capture reasoning_content from response ────────────

    def _create_chat_result(self, response: Any, **kwargs: Any) -> Any:
        """Override to capture reasoning_content from the raw API response."""
        chat_result = super()._create_chat_result(response, **kwargs)

        # The raw response from vLLM includes reasoning_content on each choice
        # when --reasoning-parser is active.
        choices = []
        if hasattr(response, "choices"):
            choices = response.choices
        elif isinstance(response, dict):
            choices = response.get("choices", [])

        for i, generation in enumerate(chat_result.generations):
            if i < len(choices):
                choice = choices[i]
                msg = choice.message if hasattr(choice, "message") else (
                    choice.get("message", {}) if isinstance(choice, dict) else None
                )
                if msg is None:
                    continue

                reasoning = (
                    getattr(msg, "reasoning_content", None)
                    or getattr(msg, "reasoning", None)
                    or (msg.get("reasoning_content") if isinstance(msg, dict) else None)
                    or (msg.get("reasoning") if isinstance(msg, dict) else None)
                )
                if reasoning:
                    generation.message.additional_kwargs["reasoning_content"] = reasoning

        return chat_result

    # ── Outgoing: re-serialize reasoning_content in assistant messages ─────

    @staticmethod
    def _convert_message_to_dict(message: BaseMessage) -> dict:
        """Override to include reasoning_content when serializing prior
        assistant messages for the next API request."""
        message_dict = ChatOpenAI._convert_message_to_dict(message)

        if isinstance(message, AIMessage):
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning:
                message_dict["reasoning_content"] = reasoning

        return message_dict

    # ── Streaming: capture reasoning from delta chunks ─────────────────────

    def _convert_chunk_to_generation(
        self, chunk: Any, **kwargs: Any
    ) -> Optional[AIMessageChunk]:
        """Hook into streaming to capture reasoning from delta.

        vLLM sends reasoning tokens via delta.reasoning (or delta.reasoning_content).
        We capture these and store them in additional_kwargs so downstream
        consumers can separate reasoning from content.
        """
        # Let the parent do its normal work first
        result = super()._convert_chunk_to_generation(chunk, **kwargs)

        if result is None:
            return result

        # result is a ChatGenerationChunk; access its .message (AIMessageChunk)
        ai_chunk = result.message if hasattr(result, "message") else result

        # Extract reasoning from the raw delta
        delta = None
        if hasattr(chunk, "choices") and chunk.choices:
            delta = getattr(chunk.choices[0], "delta", None)
        elif isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})

        if delta is not None:
            reasoning = (
                getattr(delta, "reasoning", None)
                or getattr(delta, "reasoning_content", None)
                or (delta.get("reasoning") if isinstance(delta, dict) else None)
                or (delta.get("reasoning_content") if isinstance(delta, dict) else None)
            )
            if reasoning and isinstance(ai_chunk, AIMessageChunk):
                ai_chunk.additional_kwargs["reasoning_content"] = reasoning

        return result
