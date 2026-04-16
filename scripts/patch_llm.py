"""
Patch script: Fix llm.py — Bug 4 (stop tokens) + corrupted line.
Run from the IndustrialBackend root:
    uv run python scripts/patch_llm.py
"""
import pathlib

llm_path = pathlib.Path("app/core/llm.py")
content = llm_path.read_text(encoding="utf-8")

# Build the literal token strings at runtime to avoid shell/tool escaping issues
im_end   = "<" + "|im_end|" + ">"
end_text = "<" + "|endoftext|" + ">"

# ── 1. Remove the corrupted bare-token line left by the failed edit ──────────
bad_line = end_text + '"]'
# This line appears on its own line with a trailing \r\n in the file.
if bad_line in content:
    content = content.replace(bad_line + "\r\n", "")
    content = content.replace(bad_line + "\n", "")
    print("[patch_llm] Removed corrupted bare-token line.")
else:
    print("[patch_llm] Corrupted line not found — may already be fixed.")

# ── 2. Insert correct stop-token block after kwargs.pop("top_k", None) ───────
anchor = '    kwargs.pop("top_k", None)\n'
if anchor not in content and anchor.replace("\n", "\r\n") in content:
    anchor = anchor.replace("\n", "\r\n")

correct_block = (
    "\n"
    "    # Qwen3.5 stop tokens — applied only when the caller has NOT specified 'stop'.\n"
    "    # Computer-use VL model passes stop=[] explicitly to disable stop tokens,\n"
    "    # preventing EOS tokens from cutting XML tool calls mid-generation.\n"
    "    if 'stop' not in kwargs:\n"
    f"        kwargs['stop'] = ['{im_end}', '{end_text}']\n"
)

# Only insert once
if "'stop' not in kwargs" not in content:
    content = content.replace(anchor, anchor + correct_block, 1)
    print("[patch_llm] Inserted correct stop-token block.")
else:
    print("[patch_llm] Stop-token block already present — skipping insert.")

llm_path.write_text(content, encoding="utf-8")
print("[patch_llm] Done. llm.py saved.")
