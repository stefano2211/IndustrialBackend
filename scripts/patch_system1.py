"""Patch script for computer_use_tool.py (Bug 3) and computer_use_subagent.py (Bug 2, Mejora B, Mejora C)"""
import pathlib
import hashlib
import json

# ==========================================
# 1. Patch computer_use_tool.py
# ==========================================
tool_path = pathlib.Path("app/domain/agent/tools/computer_use_tool.py")
tool_content = tool_path.read_text(encoding="utf-8")

# We mistakenly removed `elif action_type == "press":` in the last edit.
# Let's cleanly inject the `type` block right before the `press` block.
if 'key = action.get("key", "")' in tool_content and 'elif action_type == "type":' not in tool_content:
    insert_str = '''
            elif action_type == "type":
                text = action.get("text", "")
                import subprocess
                safe_text = text.replace("'", "'\\''")
                try:
                    subprocess.run(
                        f"DISPLAY=:99 xdotool type --clearmodifiers --delay 50 '{safe_text}'",
                        shell=True, check=True, timeout=15
                    )
                except Exception as e:
                    pyautogui.typewrite(text, interval=0.04)
                return f"Texto escrito: '{text[:50]}{'...' if len(text) > 50 else ''}'"

            elif action_type == "press":'''
    tool_content = tool_content.replace('                key = action.get("key", "")\n                pyautogui.press(key)', insert_str + '\n                key = action.get("key", "")\n                pyautogui.press(key)')
    tool_path.write_text(tool_content, encoding="utf-8")
    print("[patch] computer_use_tool.py Bug 3 applied")

# ==========================================
# 2. Patch computer_use_subagent.py
# ==========================================
subagent_path = pathlib.Path("app/domain/agent/subagents/computer_use_subagent.py")
sub_content = subagent_path.read_text(encoding="utf-8")

# Mejora B: Detect loading state in `observe`
observe_target = '''        return {"last_screenshot_b64": b64_data}'''
observe_replacement = '''        
        # Mejora B: Loading state detection
        import hashlib
        prev_b64 = state.get("last_screenshot_b64")
        if prev_b64 and b64_data:
            # Quick hash comparison
            prev_hash = hashlib.md5(prev_b64[:1000].encode()).hexdigest()
            new_hash = hashlib.md5(b64_data[:1000].encode()).hexdigest()
            if prev_hash == new_hash:
                logger.info("[ComputerUse] Pantalla idéntica a la anterior. Esperando 1.5s (Loading state)...")
                import asyncio
                await asyncio.sleep(1.5)
                # Re-capture after wait
                screenshot_result = await take_screenshot.ainvoke({}, config=config)
                b64_data = screenshot_result.split(",", 1)[-1] if "," in screenshot_result else screenshot_result

        return {"last_screenshot_b64": b64_data}'''
if observe_target in sub_content:
    sub_content = sub_content.replace(observe_target, observe_replacement)
    print("[patch] Mejora B (Loading State) applied")

# Bug 2: Clear last_screenshot_b64 and execute real screenshot in think_act
bug2_target_take = '''            if tool_name == "take_screenshot":
                # El modelo quiere ver la pantalla de nuevo (ya fue capturada en observe)
                pass'''
bug2_repl_take = '''            if tool_name == "take_screenshot":
                # Bug 2 Fix: El modelo explícitamente pidió una foto. La tomamos ahora.
                logger.info("[ComputerUse] El modelo solicitó explicitamente take_screenshot.")
                sc_res = await take_screenshot.ainvoke({}, config=config)
                screenshot_b64 = sc_res.split(",", 1)[-1] if "," in sc_res else sc_res
                # Update screenshot inline for the buffer
                state["last_screenshot_b64"] = screenshot_b64'''
if bug2_target_take in sub_content:
    sub_content = sub_content.replace(bug2_target_take, bug2_repl_take)
    print("[patch] Bug 2 (take_screenshot inline) applied")

# Bug 2 Return fix: clear screenshot in all ends of think_act
think_act_return_target = '''        return {
            "steps_taken": steps + 1,
            "is_complete": is_complete,
            "result_summary": result_summary,
            "trajectory": new_trajectory,
            "messages": [HumanMessage(content=user_content), response],
        }'''
think_act_return_repl = '''        return {
            "steps_taken": steps + 1,
            "last_screenshot_b64": None,  # Bug 2 Fix: forces recapture in next observe
            "is_complete": is_complete,
            "result_summary": result_summary,
            "trajectory": new_trajectory,
            "messages": [HumanMessage(content=user_content), response],
        }'''
if think_act_return_target in sub_content:
    sub_content = sub_content.replace(think_act_return_target, think_act_return_repl)
    print("[patch] Bug 2 (clear state) applied")

# Mejora C: Truncate history
history_target = '''        messages_for_llm = [
            SystemMessage(content=COMPUTER_USE_SYSTEM_PROMPT),
        ] + state["messages"] + [
            HumanMessage(content=user_content),
        ]'''
history_repl = '''        # Mejora C: Truncate history to avoid context window explosion
        # Only keep last 4 messages (2 full AI-Human interactions)
        recent_messages = state["messages"][-4:] if len(state["messages"]) > 4 else state["messages"]
        
        messages_for_llm = [
            SystemMessage(content=COMPUTER_USE_SYSTEM_PROMPT),
        ] + recent_messages + [
            HumanMessage(content=user_content),
        ]'''
if history_target in sub_content:
    sub_content = sub_content.replace(history_target, history_repl)
    print("[patch] Mejora C (Truncate history) applied")

subagent_path.write_text(sub_content, encoding="utf-8")
print("[patch] Done patching computer_use_subagent.py")
