"""
Centralized reasoning/thinking parser for LLM outputs.

Provides a single, reusable interface to separate reasoning (thinking)
from content in LLM responses. Works with two data sources:

  1. STRUCTURED (preferred): ReasoningChatOpenAI stores reasoning_content
     in AIMessage.additional_kwargs["reasoning_content"]. This is clean and
     reliable when vLLM --reasoning-parser qwen3 works correctly.

  2. FALLBACK (regex): When reasoning leaks into content (vLLM bug with
     certain model versions), strip <think>...</think> tags via regex.
     This is the backward-compatible path.

Usage:
    from app.core.reasoning import extract_reasoning

    # From an AIMessage:
    content, reasoning = extract_reasoning(ai_message)

    # From raw text (non-streaming):
    content, reasoning = extract_reasoning_from_text(raw_text)

    # Streaming filter:
    streamer = StreamingReasoningFilter()
    for token in llm_stream:
        visible, reasoning_chunk = streamer.feed(token)
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Tuple

from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk

# Compiled regex patterns for think tag extraction
_THINK_BLOCK_RE = re.compile(r'<think>(.*?)</think>\s*', flags=re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r'^<think>.*', flags=re.DOTALL)
_ORPHAN_CLOSE_RE = re.compile(r'^.*?</think>\s*', flags=re.DOTALL)


def extract_reasoning(message: BaseMessage) -> Tuple[str, Optional[str]]:
    """
    Extract (content, reasoning) from an AIMessage.

    Priority:
      1. additional_kwargs["reasoning_content"] — set by ReasoningChatOpenAI
      2. Regex fallback — strip <think>...</think> from content
    """
    content = getattr(message, "content", "") or ""
    reasoning: Optional[str] = None

    # 1. Structured path: reasoning already separated by vLLM parser
    if isinstance(message, (AIMessage, AIMessageChunk)):
        reasoning = message.additional_kwargs.get("reasoning_content")

    if reasoning:
        # Content should already be clean, but double-check for leaked tags
        clean_content = _strip_think_tags(content)
        return clean_content, reasoning

    # 2. Fallback: extract from content via regex
    return extract_reasoning_from_text(content)


def extract_reasoning_from_text(text: str) -> Tuple[str, Optional[str]]:
    """
    Extract (content, reasoning) from raw text using regex.

    Handles:
      - Complete <think>...</think> blocks
      - Unclosed <think> at the start (model cut off mid-thinking)
      - Orphan </think> without opening tag
    """
    if not text:
        return "", None

    # Collect all thinking blocks
    thinking_blocks = _THINK_BLOCK_RE.findall(text)
    reasoning = "\n".join(thinking_blocks).strip() if thinking_blocks else None

    # Strip all think blocks from content
    content = _strip_think_tags(text)

    return content, reasoning


def _strip_think_tags(text: str) -> str:
    """Remove all <think>...</think> blocks and edge cases from text."""
    if not text:
        return ""
    # Remove complete <think>...</think> blocks
    result = _THINK_BLOCK_RE.sub('', text)
    # Handle unclosed <think> at the start
    result = _UNCLOSED_THINK_RE.sub('', result)
    # Handle orphan </think> without opening tag
    result = _ORPHAN_CLOSE_RE.sub('', result)
    return result.strip()


# ---------------------------------------------------------------------------
# Streaming reasoning filter — replaces the 150-line state machine
# ---------------------------------------------------------------------------

@dataclass
class StreamingReasoningFilter:
    """
    Streaming filter that separates <think>...</think> from visible content.

    Feed tokens one at a time via .feed(token). Returns (visible, reasoning)
    for each token. Accumulates reasoning internally and exposes it via
    .reasoning property.

    Replaces the complex think_buffer/inside_think_block state machine
    that was previously inline in proactiva/agent/service.py.
    """
    _inside_think: bool = field(default=False, init=False)
    _buffer: str = field(default="", init=False)
    _reasoning_parts: list = field(default_factory=list, init=False)
    _TAG_LEN: int = field(default=8, init=False)  # len("</think>")

    @property
    def reasoning(self) -> Optional[str]:
        """Return accumulated reasoning text, or None if empty."""
        text = "".join(self._reasoning_parts).strip()
        return text if text else None

    def feed(self, token: str) -> Tuple[str, Optional[str]]:
        """
        Process a streaming token.

        Returns:
            (visible_text, reasoning_chunk)
            - visible_text: text to show to the user (may be empty string)
            - reasoning_chunk: reasoning text extracted this step (or None)
        """
        self._buffer += token
        visible = ""
        reasoning_chunk = None

        while self._buffer:
            if self._inside_think:
                end_idx = self._buffer.find("</think>")
                if end_idx != -1:
                    # Found end of think block — capture reasoning
                    chunk = self._buffer[:end_idx]
                    if chunk:
                        self._reasoning_parts.append(chunk)
                        reasoning_chunk = (reasoning_chunk or "") + chunk
                    self._buffer = self._buffer[end_idx + self._TAG_LEN:]
                    self._inside_think = False
                else:
                    # Still inside think block — keep last 8 chars for partial tag
                    if len(self._buffer) > self._TAG_LEN:
                        chunk = self._buffer[:-self._TAG_LEN]
                        self._reasoning_parts.append(chunk)
                        reasoning_chunk = (reasoning_chunk or "") + chunk
                        self._buffer = self._buffer[-self._TAG_LEN:]
                    break
            else:
                start_idx = self._buffer.find("<think>")
                if start_idx != -1:
                    # Emit text before the <think> tag
                    visible += self._buffer[:start_idx]
                    self._buffer = self._buffer[start_idx + 7:]  # len("<think>")
                    self._inside_think = True
                else:
                    # Check for orphan </think>
                    close_idx = self._buffer.find("</think>")
                    if close_idx != -1:
                        # Everything before </think> was leaked reasoning
                        leaked = self._buffer[:close_idx]
                        if leaked:
                            self._reasoning_parts.append(leaked)
                            reasoning_chunk = (reasoning_chunk or "") + leaked
                        self._buffer = self._buffer[close_idx + self._TAG_LEN:]
                        # Don't add leaked text to visible
                    else:
                        # No think tags — keep last 8 chars for partial tags
                        if len(self._buffer) > self._TAG_LEN:
                            visible += self._buffer[:-self._TAG_LEN]
                            self._buffer = self._buffer[-self._TAG_LEN:]
                        break

        return visible, reasoning_chunk

    def flush(self) -> Tuple[str, Optional[str]]:
        """
        Flush remaining buffer at end of stream.

        Returns (visible_text, reasoning_chunk).
        """
        if not self._buffer:
            return "", None

        if self._inside_think:
            # Unclosed think block — treat as reasoning
            self._reasoning_parts.append(self._buffer)
            chunk = self._buffer
            self._buffer = ""
            self._inside_think = False
            return "", chunk

        # Remaining buffer is visible content
        visible = self._buffer
        self._buffer = ""
        return visible, None

    def reset(self) -> None:
        """Reset state for a new model turn (keeps accumulated reasoning)."""
        self._inside_think = False
        self._buffer = ""

    def full_reset(self) -> None:
        """Fully reset including accumulated reasoning."""
        self._inside_think = False
        self._buffer = ""
        self._reasoning_parts = []
