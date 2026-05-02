"""
Computer Use Subagent  Sistema 1 / Digital Optimus Local
==========================================================
Implementa el loop Observe-Think-Act con Qwen3.5 nativo multimodal.

Rol en la arquitectura Macrohard:
  - Recibe instrucción de alto nivel del Orchestrator (System 2)
  - Ejecuta la instrucción viendo la pantalla en tiempo real
  - Guarda cada step de la trayectoria en el VL Replay Buffer
  - Termina cuando llama a task_complete() o supera max_steps

Flujo por step:
  1. take_screenshot() ? imagen base64
  2. Modelo Qwen3.5 (nativamente VLM) recibe imagen + instrucción
  3. Modelo devuelve JSON de acción: {"type": "click", "x": N, "y": N}
  4. execute_action(action_json) ? ejecuta la acción
  5. VL Replay Buffer guarda (screenshot, instrucción, acción)
  6. Loop hasta task_complete() o max_steps
"""

import asyncio
import hashlib
import json
import re as _re
from typing import Optional

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from typing import TypedDict, List, Annotated
import operator
from loguru import logger

from app.core.config import settings
from app.domain.shared.agent.tools.computer_use_tool import (
    take_screenshot,
    get_page_context,
    execute_action,
    run_shell_command,
    task_complete,
    COMPUTER_USE_TOOLS,
    get_clean_b64,
)
from app.domain.shared.agent.tools.omniparser_service import get_omniparser
from app.persistence.proactiva.vl_replay_buffer import VLReplayBuffer

# -- System Prompt -------------------------------------------------------------

COMPUTER_USE_SYSTEM_PROMPT = """\
<role>Aura Sistema 1  Autonomous Computer Use Agent</role>

<mission>
You are an autonomous computer operator. You receive a high-level instruction,
observe the screen, decide the single best action, and execute it  repeating
until the task is complete or definitively blocked.

You operate the actual computer screen. Actions have real effects.
Reason carefully before every action. One wrong click can break the flow.
</mission>

<environment>
- OS: Ubuntu Linux (Xvfb headless display :99, real resolution 1920×1080)
- Browser: Chromium (persistent profile  cookies/sessions are preserved across launches)
- Screenshots: delivered at native resolution of 1920×1080 pixels
- Coordinates: use SCREEN space (x: 01920, y: 01080) natively.
  The yellow grid overlay on each screenshot shows SCREEN coordinates as labels (200, 400, 600).
  Read the nearest grid labels to determine the exact screen position of your target.
  Do NOT divide or scale the coordinates  use the label values exactly as printed.
- If a numbered grid or element labels are visible on the screenshot, use those to pinpoint targets precisely.
</environment>

<core_loop>
Every turn follows this exact sequence:
  OBSERVE ? THINK ? ACT

1. OBSERVE  Always call take_screenshot first. Never act without a fresh screenshot.
2. THINK  Reason internally inside <thinking> tags:
   a. Current state: What page/app is visible? Has it finished loading?
   b. Progress: What has changed since the previous step? Did my last action work?
   c. Blockers: Is there a popup, cookie banner, captcha, or error overlay? Handle it first.
   d. Target: What is the SINGLE next concrete action to advance toward the goal?
   e. Grounding: If <detected_elements> exist, which element ID is the right target?
      If no elements, read the yellow grid labels on the screenshot to determine exact SCREEN coordinates (01920, 01080).
   f. Failures: Are any candidates in the FAILED CLICKS list? Skip them, try alternatives.
3. ACT  Output exactly ONE tool call. No text before or after.
</core_loop>

<thinking_strategies>
Use these strategies when stuck or uncertain:

LOADING: If the page looks incomplete, take 12 more screenshots before clicking.
  Signs of loading: spinner, progress bar, grayed-out elements, blank sections.

POPUP / MODAL: If ANY overlay, dialog, cookie banner, or permission prompt is visible:
  ? Handle it IMMEDIATELY before anything else  it blocks all other interactions.
  ? For cookie banners: look for "Accept", "Accept all", "Aceptar", "Agree", "I agree", "OK",
     "Reject" or the X close button. Click the most prominent option.
  ? For Google consent screens: the blue "Accept all" button is usually at the bottom center
     of the dialog (~y=490, x=660 for 1920x1080). Always take a fresh screenshot first to confirm.
  ? After dismissing: take_screenshot to verify it's gone before proceeding.
  ? Only then proceed with the original task.

SCROLL TO FIND: If the target element is not visible in the screenshot:
  ? Scroll down (amount: 35) and take a new screenshot. Repeat until found.
  ? Do not click in the wrong area because you assumed element position.

WRONG CLICK: If a click had no effect or opened the wrong thing:
  ? Take a screenshot to assess the new state.
  ? Try clicking a visually adjacent element or a more precise location.
  ? If a text label is visible on/near the button, aim for the CENTER of that text, not the edge.
  ? If the button still doesn't respond, try clicking 10-15px ABOVE or BELOW your previous coordinate.
  ? As a last resort, try pressing Enter or Space after moving the mouse to the target element.

FORM INPUT: When filling a form field:
  1. Click the field first to focus it.
  2. Press ctrl+a to select any existing text.
  3. Then type the new value.

CAPTCHA: If a CAPTCHA appears that requires human interaction:
  ? Immediately call task_complete with: "BLOCKED: CAPTCHA requires human verification.
     Pre-authenticate the browser profile and retry."
</thinking_strategies>

<som_grounding>
If <detected_elements> is populated, prefer using 'element_id'.
However, since we are moving towards Vision-Only pure multimodal behavior,
if there are no elements or you feel confident, directly predict the raw (x, y) coordinates
based purely on your visual spatial awareness out of 1920x1080.
</som_grounding>

<tools>
take_screenshot()
  ? Captures the current screen as a base64 PNG.
  ? ALWAYS call this first at the start of every turn.

execute_action(action_json)
  ? Executes one GUI interaction. x/y are SCREEN coordinates (01920 for x, 01080 for y):
  {"type": "click",             "x": int, "y": int}        ? e.g. {"type":"click","x":960,"y":540}
  {"type": "double_click",      "x": int, "y": int}
  {"type": "type",              "text": "string"}           ? clipboard-paste auto for text >80 chars
  {"type": "press",             "key": "Return|Tab|Escape|ctrl+a|ctrl+c|ctrl+v|ctrl+l|ctrl+t|ctrl+w|ctrl+Tab|alt+Left|Page_Down|Page_Up|Home|End|space|F5"}
  {"type": "scroll",            "x": int, "y": int, "amount": int}  ? positive=down, negative=up
  {"type": "move",              "x": int, "y": int}
  {"type": "navigate",          "url": "https://..."}   ? FASTEST way to go to any URL (Ctrl+L + type + Enter)
  {"type": "new_tab"}                                   ? Ctrl+T  open a new browser tab
  {"type": "close_tab"}                                 ? Ctrl+W  close current tab
  {"type": "focus_address_bar"}                         ? Ctrl+L  ready to type a URL

run_shell_command(command: str)
  ? Runs a shell command (use to launch Chromium or other apps).
  ? Chromium flags and persistent profile are injected automatically.
  ? Example: run_shell_command("chromium https://example.com &")
  ? Always add & at the end for background processes.

task_complete(result: str)
  ? Signals task completion or unrecoverable failure.
  ? Call when: task is fully done, OR 3+ consecutive attempts at same step all fail.
  ? Include: what was accomplished, what data was found, and any failures.

get_page_context()
  ? Returns the Accessibility Tree (semantic list of interactive elements) + URL + title.
  ? Only available in Playwright mode. Call AFTER take_screenshot to get structured DOM data.
  ? When the tree lists an element (e.g., button "Compose"), PREFER pw_click/pw_fill over raw coords.
</tools>

<playwright_actions>
When get_page_context() returns an Accessibility Tree, use these SEMANTIC actions
(far more reliable than guessing pixel coordinates):

  pw_goto:   {"type": "pw_goto", "url": "https://mail.google.com"}
  pw_click:  {"type": "pw_click", "role": "button", "name": "Compose"}
  pw_fill:   {"type": "pw_fill", "role": "textbox", "name": "To", "value": "ops@plant.com"}
  pw_type:   {"type": "pw_type", "text": "Hello world"}
  pw_press:  {"type": "pw_press", "key": "Enter"}
  pw_scroll: {"type": "pw_scroll", "direction": "down", "amount": 3}
  pw_wait:   {"type": "pw_wait", "timeout": 5000}

DECISION RULE:
  - If get_page_context() returns an accessibility tree → USE pw_* actions.
  - If tree is empty or unavailable → FALL BACK to coordinate-based actions.
  - For native desktop apps (SAP GUI, terminals) → always use coordinate-based actions.
</playwright_actions>

<browser_workflow>
OPENING A WEBSITE (browser not yet open):
  run_shell_command("chromium https://example.com &")
  ? Then take 23 screenshots while the browser loads before clicking anything.

NAVIGATING TO A URL (browser already open  FASTEST):
  execute_action({"type": "navigate", "url": "https://example.com"})
  ? Then take_screenshot to confirm the page loaded.

OPENING A SECOND SITE (new tab):
  execute_action({"type": "new_tab"})
  execute_action({"type": "navigate", "url": "https://other-site.com"})

LOGIN FLOW:
  1. Navigate to the login page.
  2. Click the username/email field ? type the email.
  3. Press Tab or click the password field ? type the password.
  4. Press Return or click the "Sign in" / "Log in" button.
  5. Take screenshot  verify successful login (look for user avatar, dashboard, inbox).
  If a verification step appears (SMS code, 2FA): describe it in task_complete as a blocker.

GOOGLE SEARCH:
  execute_action({"type": "navigate", "url": "https://www.google.com/search?q=YOUR+QUERY"})
  This is faster than typing in the search box.

READING PAGE CONTENT:
  After the page loads, take_screenshot and describe what you can see.
  If content is below the fold: scroll down ? take_screenshot ? repeat.

TYPICAL ELEMENT POSITIONS (1920×1080 image  always verify with screenshot):
  Chromium address bar:  y50,  x700900
  Chromium tabs row:     y20,  x varies per tab
  Google search bar:     y540, x960  (homepage) or y70, x800 (results page)
  Gmail compose button:  y270, x150  (left sidebar)
  Gmail compose window:  y6401060 (bottom-right overlay)
  SAP Fiori search:      y110,  x960
  Generic page content:  y200960, x01920
</browser_workflow>

<adaptive_strategies>
SITE WON'T LOAD: Try pressing F5 (refresh). If still blank after 3 tries, note it.
ELEMENT NOT CLICKABLE: Try scrolling to bring it into view, then click.
TEXT NOT TYPED: Click the field again (re-focus), press ctrl+a to clear, then type.
WRONG PAGE OPENED: Use navigate action to go to the correct URL directly.
PAGE IN WRONG LANGUAGE: The content is whatever the site shows  describe it as-is.
DYNAMIC CONTENT (SPA): After clicking a button, take screenshot before assuming navigation completed.
DROPDOWN MENUS: Click the dropdown trigger, wait for options to appear, then click the option.
AUTOCOMPLETE: Type partial text, wait for suggestions (take screenshot), then click the correct suggestion.
FILE DOWNLOAD: Describe the download dialog visible on screen. Note the filename shown.
</adaptive_strategies>

<error_recovery>
If an action has no effect after 1 attempt:
  1. Take a screenshot to verify current state.
  2. Try a different approach: different coordinates, different action type, scroll first.
  3. After 3 failed attempts on the SAME sub-step: call task_complete describing what failed and why.

Never repeat the exact same failed action more than once.
</error_recovery>

<rules>
- ALWAYS call take_screenshot at the start of every turn before any execute_action.
- Output exactly ONE tool call per turn. Never chain calls.
- Use <thinking> tags for all reasoning. Never output reasoning as plain text.
- No conversational filler. Output is tool calls only.
- Wait for pages/apps to load: take 23 screenshots if content is still appearing.
- Coordinates are ALWAYS in native 1920×1080 IMAGE space. Act precise.
- Never guess content from memory  only act on what you SEE in the screenshot.
</rules>

<examples>
<example>
<instruction>Search for "industrial safety regulations 2025" on Google and report the first 3 results.</instruction>
<step_1>take_screenshot() ? desktop visible, no browser open</step_1>
<thinking>Need to open browser and search Google. Will use direct search URL  fastest approach.</thinking>
<step_2>run_shell_command("chromium https://www.google.com/search?q=industrial+safety+regulations+2025 &")</step_2>
<step_3>take_screenshot() ? Chromium loading</step_3>
<thinking>Browser loading. Wait one more screenshot.</thinking>
<step_4>take_screenshot() ? Google results page fully loaded, showing 10 results</step_4>
<thinking>Page loaded. I can see results. Will read them from the screenshot and report.</thinking>
<step_5>task_complete("Google search completed. First 3 results: 1) OSHA 2025 Standards Update  osha.gov. 2) ISO 45001:2025 Industrial Safety Guide  iso.org. 3) EU Safety Directive 2025/37  eur-lex.europa.eu.")</step_5>
</example>

<example>
<instruction>Go to Gmail and compose an email to ops@plant.com with subject 'Equipment Report' and body 'Please find the attached report for April 2026.'</instruction>
<step_1>take_screenshot() ? desktop, browser already open on some page</step_1>
<thinking>Browser is open. Navigate to Gmail directly  fastest method.</thinking>
<step_2>execute_action({"type": "navigate", "url": "https://mail.google.com"})</step_2>
<step_3>take_screenshot() ? Gmail inbox loaded, user already logged in</step_3>
<thinking>Gmail is open and logged in (session cookie active). I need to click Compose  it's the red button on the left sidebar around (75, 135).</thinking>
<step_4>execute_action({"type": "click", "x": 75, "y": 135})</step_4>
<step_5>take_screenshot() ? Compose window appeared at bottom-right of screen</step_5>
<thinking>Compose window is open. I see the To field at the top. Click it and type the recipient.</thinking>
<step_6>execute_action({"type": "click", "x": 760, "y": 420})</step_6>
<step_7>execute_action({"type": "type", "text": "ops@plant.com"})</step_7>
<step_8>execute_action({"type": "press", "key": "Tab"})</step_8>
<step_9>execute_action({"type": "type", "text": "Equipment Report"})</step_9>
<step_10>execute_action({"type": "press", "key": "Tab"})</step_10>
<step_11>execute_action({"type": "type", "text": "Please find the attached report for April 2026."})</step_11>
<step_12>take_screenshot() ? compose window filled correctly</step_12>
<thinking>All fields are filled. Now click Send button.</thinking>
<step_13>execute_action({"type": "click", "x": 700, "y": 510})</step_13>
<step_14>take_screenshot() ? compose window closed, inbox visible with "Message sent" confirmation</step_14>
<step_15>task_complete("Email composed and sent to ops@plant.com. Subject: 'Equipment Report'. Message sent confirmation visible.")</step_15>
</example>

<example>
<instruction>Navigate to SAP Fiori and open transaction MB51 for material CRUDE-100.</instruction>
<step_1>take_screenshot() ? empty desktop</step_1>
<thinking>No browser open. Launch Chromium with SAP Fiori URL.</thinking>
<step_2>run_shell_command("chromium https://sap-fiori.company.local &")</step_2>
<step_3>take_screenshot() ? SAP Fiori login screen visible</step_3>
<thinking>Login page loaded. I see the username field (y200, x480). Click it and enter credentials.</thinking>
<step_4>execute_action({"type": "click", "x": 480, "y": 200})</step_4>
<step_5>execute_action({"type": "type", "text": "AURA_USER"})</step_5>
<step_6>execute_action({"type": "press", "key": "Tab"})</step_6>
<step_7>execute_action({"type": "type", "text": "PASSWORD"})</step_7>
<step_8>execute_action({"type": "press", "key": "Return"})</step_8>
<step_9>take_screenshot() ? SAP Fiori Launchpad loaded</step_9>
<thinking>Logged in. I need to search for MB51. I'll use the search bar at top.</thinking>
<step_10>execute_action({"type": "click", "x": 480, "y": 55})</step_10>
<step_11>execute_action({"type": "type", "text": "MB51"})</step_11>
<step_12>execute_action({"type": "press", "key": "Return"})</step_12>
<step_13>take_screenshot() ? MB51 transaction open, material field visible</step_13>
<step_14>execute_action({"type": "click", "x": 480, "y": 220})</step_14>
<step_15>execute_action({"type": "type", "text": "CRUDE-100"})</step_15>
<step_16>execute_action({"type": "press", "key": "Return"})</step_16>
<step_17>take_screenshot() ? MB51 results showing movements for CRUDE-100</step_17>
<step_18>task_complete("MB51 opened and executed for CRUDE-100. Results visible: [describe what is shown in the screenshot].")</step_18>
</example>
</examples>
"""


# -- LangGraph State -----------------------------------------------------------

class ComputerUseState(TypedDict):
    instruction: str
    messages: Annotated[List, operator.add]
    steps_taken: int
    last_screenshot_b64: Optional[str]
    trajectory: Annotated[List, operator.add]  # lista de steps para VL Buffer
    is_complete: bool
    result_summary: str
    # UI-TARS style: rolling window of last N (screenshot_thumb, action) pairs for VLM context
    action_history: List[dict]
    # Elements that were clicked but produced no visible screen change
    failed_elements: List[str]
    # OmniParser V2: numbered element list text for SoM grounding
    parsed_elements: Optional[str]
    # OmniParser V2: annotated screenshot with numbered bounding boxes
    annotated_screenshot_b64: Optional[str]
    # Playwright Accessibility Tree (semantic UI elements)
    a11y_tree: Optional[str]


# -- Graph Nodes ---------------------------------------------------------------

def _build_init_node():
    """
    Entry node: extracts the instruction from the messages list injected by
    deepagents and initialises all required ComputerUseState fields.

    deepagents passes subagent inputs as {"messages": [HumanMessage(...)]}.
    The raw StateGraph expects ComputerUseState fields (instruction, steps_taken,
    etc.)  this node bridges that impedance mismatch.
    """

    async def init(state: ComputerUseState) -> dict:
        instruction = state.get("instruction", "")

        if not instruction:
            for msg in reversed(state.get("messages", [])):
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content.strip():
                    instruction = content.strip()
                    break
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "").strip()
                            if text:
                                instruction = text
                                break
                    if instruction:
                        break

        return {
            "instruction": instruction,
            "steps_taken": state.get("steps_taken", 0),
            "last_screenshot_b64": state.get("last_screenshot_b64"),
            "trajectory": state.get("trajectory", []),
            "is_complete": state.get("is_complete", False),
            "result_summary": state.get("result_summary", ""),
            "action_history": state.get("action_history", []),
            "failed_elements": state.get("failed_elements", []),
            "parsed_elements": state.get("parsed_elements", None),
            "annotated_screenshot_b64": state.get("annotated_screenshot_b64", None),
        }

    return init


def _get_b64_dimensions(b64_data: str) -> tuple:
    """Return (width, height) of a base64 PNG. Falls back to (683, 384) if PIL fails."""
    try:
        import base64 as _b64
        from PIL import Image
        import io as _io
        img = Image.open(_io.BytesIO(_b64.b64decode(b64_data)))
        return img.size  # (width, height)
    except Exception:
        return (683, 384)  # half of 1366x768 default virtual display


def _make_history_thumbnail(b64_data: str) -> str:
    """
    Reduce el screenshot a ~480×270 para guardarlo en action_history sin saturar el estado.
    Retorna el b64 reducido, o el original si PIL no está disponible.
    """
    try:
        import base64 as _b64
        from PIL import Image
        import io as _io
        img_bytes = _b64.b64decode(b64_data)
        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((480, 270), Image.LANCZOS)
        buf = _io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return _b64.b64encode(buf.getvalue()).decode()
    except Exception:
        return b64_data[:5000]  # fallback: recortar si PIL no disponible


def _build_observe_node(llm: BaseChatModel):
    """
    Nodo: captura screenshot, corre OmniParser si está activo,
    detecta clicks fallidos y actualiza el estado.
    """

    async def observe(state: ComputerUseState, config: RunnableConfig) -> dict:
        logger.info(f"[ComputerUse] Step {state['steps_taken'] + 1}  Observando pantalla...")

        # Captura pantalla
        screenshot_result = await take_screenshot.ainvoke({}, config=config)
        b64_data = screenshot_result.split(",", 1)[-1] if "," in screenshot_result else screenshot_result

        # -- Loading state detection ---------------------------------------------
        prev_b64 = state.get("last_screenshot_b64")
        screen_changed = True
        if prev_b64 and b64_data:
            prev_hash = hashlib.md5(prev_b64[:10000].encode()).hexdigest()
            new_hash = hashlib.md5(b64_data[:10000].encode()).hexdigest()
            if prev_hash == new_hash:
                screen_changed = False
                # Determine wait time based on last action type (clicks may need longer)
                last_was_click = False
                if state.get("action_history"):
                    try:
                        last_act = json.loads(state["action_history"][-1].get("action_json", "{}"))
                        last_was_click = last_act.get("type") in ("click", "double_click", "pw_click")
                    except Exception:
                        pass
                wait_sec = 3.0 if last_was_click else 1.5
                logger.info(f"[ComputerUse] Pantalla idéntica a la anterior. Esperando {wait_sec}s...")
                await asyncio.sleep(wait_sec)
                screenshot_result = await take_screenshot.ainvoke({}, config=config)
                b64_data = screenshot_result.split(",", 1)[-1] if "," in screenshot_result else screenshot_result
                # Check again after wait
                new_hash2 = hashlib.md5(b64_data[:10000].encode()).hexdigest()
                screen_changed = new_hash2 != prev_hash

        # -- Failed click detection ----------------------------------------------
        # If the screen didn't change after a click action, that element is non-interactive
        failed_elements = list(state.get("failed_elements", []))
        history = state.get("action_history", [])
        if not screen_changed and history:
            last = history[-1]
            last_action = last.get("action_json", "")
            if last_action:
                try:
                    act = json.loads(last_action)
                    if act.get("type") in ("click", "double_click", "pw_click"):
                        fail_desc = f"click at ({act.get('x')},{act.get('y')}) step {last.get('step','?')}"
                        if fail_desc not in failed_elements:
                            failed_elements.append(fail_desc)
                            logger.warning(f"[ComputerUse] Failed click detected: {fail_desc}")
                except Exception as e:
                    logger.debug(f"[ComputerUse] Failed click parse error: {e}")

        # Prune failed_elements to avoid unbounded state growth
        if len(failed_elements) > 15:
            failed_elements = failed_elements[-15:]

        # -- OmniParser V2  Set-of-Marks grounding -----------------------------
        parsed_elements: Optional[str] = None
        annotated_b64: Optional[str] = None

        if settings.omniparser_enabled:
            omniparser = get_omniparser()
            if omniparser.is_available(settings.omniparser_model_dir):
                try:
                    result = await omniparser.parse(b64_data)
                    if result.elements:
                        annotated_b64 = result.annotated_b64
                        parsed_elements = result.element_list_text
                        logger.info(
                            f"[ComputerUse] OmniParser: {len(result.elements)} elements detected."
                        )
                except Exception as e:
                    logger.warning(f"[ComputerUse] OmniParser error: {e}")

        # -- Playwright Accessibility Tree (when available) -------------------------
        a11y_tree = None
        if settings.playwright_enabled:
            try:
                from app.domain.shared.agent.tools.browser_manager import get_browser_manager
                mgr = get_browser_manager()
                if mgr.is_ready:
                    a11y_tree = await mgr.accessibility_snapshot()
                    if a11y_tree:
                        logger.info(
                            f"[ComputerUse] Accessibility Tree captured "
                            f"({len(a11y_tree)} chars, "
                            f"{a11y_tree.count(chr(10))} elements)."
                        )
            except Exception as e:
                logger.debug(f"[ComputerUse] Accessibility tree unavailable: {e}")

        # -- Live Screen Viewer  stream screenshot to frontend via SSE --------------
        thread_id = config.get("configurable", {}).get("thread_id", "default") if config else "default"
        clean_b64 = get_clean_b64(thread_id)
        display_b64 = annotated_b64 if annotated_b64 else (clean_b64 or b64_data)

        # Extract last action info for the viewer (action label + click ripple)
        last_action_str = None
        click_pct = None
        if state.get("action_history"):
            last_entry = state["action_history"][-1]
            if isinstance(last_entry, dict):
                last_action_str = last_entry.get("action_summary")
                last_action_json = last_entry.get("action_json")
                if last_action_json:
                    try:
                        act = json.loads(last_action_json)
                        if act.get("type") in ("click", "double_click", "pw_click") and "x" in act and "y" in act:
                            img_w, img_h = _get_b64_dimensions(display_b64)
                            click_pct = {
                                "x": round(act["x"] / img_w * 100, 2),
                                "y": round(act["y"] / img_h * 100, 2),
                                "type": act["type"],
                            }
                    except Exception:
                        pass

        await adispatch_custom_event(
            "screenshot",
            {
                "b64": display_b64,
                "step": state["steps_taken"] + 1,
                "has_omniparser": annotated_b64 is not None,
                "has_a11y_tree": a11y_tree is not None,
                "action": last_action_str,
                "click": click_pct,
            },
            config=config,
        )

        return {
            "last_screenshot_b64": b64_data,
            "annotated_screenshot_b64": annotated_b64,
            "parsed_elements": parsed_elements,
            "failed_elements": failed_elements,
            "a11y_tree": a11y_tree,
        }

    return observe


def _build_think_act_node(llm: BaseChatModel, vl_replay_buffer: Optional[VLReplayBuffer]):
    """Nodo: razona sobre el screenshot y ejecuta UNA acción."""

    async def think_act(state: ComputerUseState, config: RunnableConfig) -> dict:
        steps = state["steps_taken"]
        instruction = state["instruction"]
        screenshot_b64 = state["last_screenshot_b64"]
        max_steps = settings.computer_use_max_steps

        if steps >= max_steps:
            logger.warning(f"[ComputerUse] Límite de {max_steps} steps alcanzado. Finalizando.")
            return {
                "is_complete": True,
                "result_summary": f"Límite de {max_steps} pasos alcanzado. Tarea parcialmente completada.",
                "steps_taken": steps + 1,
            }

        # Stall detection: if the last 4 actions are clicks within 120px of each other,
        # the agent is stuck in a loop  abort gracefully.
        # Radius is 120px (not 50px) to avoid false positives on small buttons/cookie banners.
        recent_traj = state.get("trajectory", [])
        if len(recent_traj) >= 4:
            try:
                last4 = [json.loads(t["action_json"]) for t in recent_traj[-4:]]
                clicks = [a for a in last4 if a.get("type") in ("click", "double_click")]
                if len(clicks) >= 4:
                    xs = [c["x"] for c in clicks]
                    ys = [c["y"] for c in clicks]
                    if max(xs) - min(xs) < 120 and max(ys) - min(ys) < 120:
                        logger.warning(
                            f"[ComputerUse] Stall detectado: 4 clicks consecutivos en región "
                            f"({min(xs)}-{max(xs)}, {min(ys)}-{max(ys)}). Terminando."
                        )
                        return {
                            "is_complete": True,
                            "result_summary": (
                                "Stall detectado: el agente estuvo clickeando en la misma región "
                                "4 veces consecutivas sin cambio de pantalla. "
                                "Posibles causas: popup no dismissable, botón fuera de foco, página en carga lenta. "
                                "Intenta de nuevo con una instrucción más específica o verifica el estado del navegador."
                            ),
                            "steps_taken": steps + 1,
                        }
            except Exception as e:
                logger.debug(f"[ComputerUse] Stall detection error: {e}")

        # -- Decide which screenshot to show (annotated with SoM > raw) --------
        display_screenshot = state.get("annotated_screenshot_b64") or screenshot_b64
        parsed_elements = state.get("parsed_elements")
        failed_elements = state.get("failed_elements", [])
        action_history = state.get("action_history", [])
        max_history = settings.computer_use_context_screenshots

        # -- Build current turn message --------------------------------------
        text_parts = [f"Instruction: {instruction}\nStep {steps + 1}: Decide next action."]

        if failed_elements:
            text_parts.append(
                "\n?? FAILED CLICKS (no visible screen change  skip these, try alternatives):\n"
                + "\n".join(f"  - {e}" for e in failed_elements[-6:])
            )

        if parsed_elements:
            text_parts.append(
                f"\n<detected_elements>\n{parsed_elements}\n</detected_elements>\n"
                "Prefer 'click element #N' over raw coordinates when elements are listed above."
            )

        a11y_tree = state.get("a11y_tree")
        if a11y_tree:
            text_parts.append(
                f"\n<accessibility_tree>\n{a11y_tree}\n</accessibility_tree>\n"
                "Use pw_click, pw_fill, pw_type, pw_press, pw_scroll for elements listed above."
            )

        # -- Conditional grounding mode (prevents VLM confusion between SoM/A11y/coords) ---
        if parsed_elements:
            text_parts.append(
                "\n<grounding_mode>\n"
                "OmniParser visual SoM is ACTIVE. Elements are numbered on the screenshot.\n"
                "Use element_id references from <detected_elements>. DO NOT guess coordinates.\n"
                "</grounding_mode>"
            )
        elif a11y_tree:
            text_parts.append(
                "\n<grounding_mode>\n"
                "Accessibility Tree is ACTIVE. Use pw_* semantic actions with exact role+name.\n"
                "DO NOT use raw coordinates when semantic actions are available.\n"
                "</grounding_mode>"
            )
        else:
            text_parts.append(
                "\n<grounding_mode>\n"
                "No structured elements available. Use raw coordinates (0-1920, 0-1080).\n"
                "Read yellow grid labels on screenshot to determine exact positions.\n"
                "</grounding_mode>"
            )

        # -- Structured previous_actions for temporal reasoning ----------------
        if action_history:
            prev_lines = []
            for hist in action_history[-max_history:]:
                prev_lines.append(
                    f"  Step {hist.get('step', '?')}: {hist.get('action_summary', 'unknown')}"
                )
            text_parts.append(
                f"\n<previous_actions>\n"
                f"Last {len(prev_lines)} actions taken:\n"
                + "\n".join(prev_lines)
                + "\n</previous_actions>\n"
                "Compare Previous vs Current screenshot to verify the last action worked.\n"
                "If the screen did not change, the action failed — try a different approach."
            )

        user_content = [{"type": "text", "text": "\n".join(text_parts)}]

        # -- Frame history: previous screenshot for temporal comparison ----------
        # With 48GB VRAM we can afford 2 images per turn (prev + current) for
        # state-change detection, loading detection, and animation analysis.
        if action_history and len(action_history) >= 1:
            prev_b64 = action_history[-1].get("screenshot_b64")
            if prev_b64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{prev_b64}"},
                })
                user_content.append({
                    "type": "text",
                    "text": "[Previous screenshot — state BEFORE last action]",
                })

        if display_screenshot:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{display_screenshot}"},
            })
            user_content.append({
                "type": "text",
                "text": "[Current screenshot — state AFTER last action]",
            })

        # -- Build UI-TARS style action history context ----------------------
        # Show last N (screenshot, action) pairs so the model can reason about what changed
        history_messages = []
        for hist in action_history[-max_history:]:
            hist_content = []
            thumb = hist.get("screenshot_thumb")
            if thumb:
                hist_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{thumb}"},
                })
            hist_content.append({
                "type": "text",
                "text": f"[Step {hist.get('step', '?')}] ? Action taken: {hist.get('action_summary', 'unknown')}",
            })
            history_messages.append(HumanMessage(content=hist_content))

        messages_for_llm = (
            [SystemMessage(content=COMPUTER_USE_SYSTEM_PROMPT)]
            + history_messages
            + [HumanMessage(content=user_content)]
        )

        # Llamar al VL model con herramientas
        llm_with_tools = llm.bind_tools(COMPUTER_USE_TOOLS)

        try:
            response = await llm_with_tools.ainvoke(messages_for_llm, config=config)
        except Exception as e:
            logger.error(f"[ComputerUse] Error invocando VL model: {e}")
            return {
                "is_complete": True,
                "result_summary": f"Error en el modelo VL: {e}",
                "steps_taken": steps + 1,
                "messages": [HumanMessage(content=user_content)],
            }

        # Parsear tool calls del response
        tool_calls = getattr(response, "tool_calls", [])
        
        # --- Fallback robusto para Qwen3.5-VL ---
        # A veces Qwen emite el XML correcto pero LangChain falla en parsearlo
        if not tool_calls and isinstance(response.content, str) and "<tool_call>" in response.content:
            try:
                import xml.etree.ElementTree as ET
                xml_str = response.content[response.content.find("<tool_call>"):response.content.rfind("</tool_call>") + 12]
                root = ET.fromstring(xml_str)
                tool_name = root.find("name").text if root.find("name") is not None else ""
                args_str = root.find("arguments").text if root.find("arguments") is not None else "{}"
                tool_calls = [{"name": tool_name.strip(), "args": json.loads(args_str)}]
                logger.info(f"[ComputerUse] Fallback XML parser exitoso: extraídos {len(tool_calls)} tool calls.")
            except Exception as xml_e:
                logger.warning(f"[ComputerUse] Error en fallback XML parser: {xml_e}")

        new_trajectory = []
        is_complete = False
        result_summary = ""
        executed_action_json: Optional[str] = None

        # Process tool calls: only the first executable action (execute_action / run_shell_command /
        # task_complete) is honoured per turn. The system prompt enforces 1 action/turn, but the
        # model may occasionally emit multiple; executing them without re-observing the screen
        # would create inconsistent state. take_screenshot is always allowed (it is read-only).
        _action_executed = False
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            if _action_executed and tool_name != "take_screenshot":
                logger.warning(
                    f"[ComputerUse] Extra tool call '{tool_name}' ignored  "
                    "only 1 action per turn is allowed."
                )
                continue

            if tool_name == "take_screenshot":
                logger.info("[ComputerUse] El modelo solicitó explicitamente take_screenshot.")
                sc_res = await take_screenshot.ainvoke({}, config=config)
                screenshot_b64 = sc_res.split(",", 1)[-1] if "," in sc_res else sc_res

            elif tool_name == "execute_action":
                action_json_raw = tool_args.get("action_json", "{}")

                # -- SoM element ID resolution ------------------------------
                # If the model output {"element_id": N}, resolve to {"type":"click","x":cx,"y":cy}
                try:
                    act_parsed = json.loads(action_json_raw)
                    elem_id = act_parsed.get("element_id") or act_parsed.get("id")
                    if elem_id is not None and parsed_elements:
                        resolved = False
                        for line in parsed_elements.splitlines():
                            if line.startswith(f"[{elem_id}]"):
                                m = _re.search(r"center\((\d+),(\d+)\)", line)
                                if m:
                                    cx, cy = int(m.group(1)), int(m.group(2))
                                    action_json_raw = json.dumps({
                                        "type": act_parsed.get("type", "click"),
                                        "x": cx, "y": cy
                                    })
                                    logger.info(f"[ComputerUse] SoM element #{elem_id} ? ({cx},{cy})")
                                    resolved = True
                                    break
                        if not resolved:
                            logger.warning(f"[ComputerUse] SoM element #{elem_id} not found in parsed_elements")
                except Exception:
                    pass

                action_result = await execute_action.ainvoke(
                    {"action_json": action_json_raw}, config=config
                )
                executed_action_json = action_json_raw
                _action_executed = True
                logger.info(f"[ComputerUse] Acción ejecutada: {action_result}")
                # Post-action pause: give the UI time to react (animations, page transitions)
                # before the next observe cycle captures the screen.
                await asyncio.sleep(1.5)

                if vl_replay_buffer and screenshot_b64:
                    try:
                        await vl_replay_buffer.append_trajectory_step(
                            instruction=instruction,
                            screenshot_b64=screenshot_b64,
                            action_json=action_json_raw,
                            tool_name="computer_use",
                        )
                        new_trajectory.append({
                            "instruction": instruction,
                            "screenshot_b64": screenshot_b64,
                            "action_json": action_json_raw,
                        })
                    except Exception as e:
                        logger.warning(f"[ComputerUse] Error guardando en VL buffer: {e}")

            elif tool_name == "run_shell_command":
                command = tool_args.get("command", "")
                action_result = await run_shell_command.ainvoke(
                    {"command": command}, config=config
                )
                executed_action_json = json.dumps({"type": "shell", "command": command})
                _action_executed = True
                logger.info(f"[ComputerUse] Shell command disparado: {action_result}")
                # Post-shell pause: give launched apps (Chromium etc) time to open
                await asyncio.sleep(2.5)

                if vl_replay_buffer and screenshot_b64:
                    try:
                        await vl_replay_buffer.append_trajectory_step(
                            instruction=instruction,
                            screenshot_b64=screenshot_b64,
                            action_json=executed_action_json,
                            tool_name="run_shell_command",
                        )
                        new_trajectory.append({
                            "instruction": instruction,
                            "screenshot_b64": screenshot_b64,
                            "action_json": executed_action_json,
                        })
                    except Exception as e:
                        logger.warning(f"[ComputerUse] Error guardando en VL buffer: {e}")

            elif tool_name == "task_complete":
                summary = tool_args.get("summary", "Tarea completada.")
                await task_complete.ainvoke({"summary": summary}, config=config)
                is_complete = True
                _action_executed = True
                result_summary = summary
                logger.info(f"[ComputerUse] ? Tarea completada: {summary}")

        # -- Update action_history (UI-TARS rolling window) ------------------
        new_action_history = list(action_history)
        if executed_action_json:
            try:
                act = json.loads(executed_action_json)
                action_summary = f"{act.get('type','?')}"
                if act.get('type') in ('click', 'double_click'):
                    action_summary += f" at ({act.get('x')},{act.get('y')})"
                elif act.get('type') == 'type':
                    action_summary += f" '{act.get('text','')[:40]}'"
                elif act.get('type') == 'press':
                    action_summary += f" '{act.get('key','')}'"  
                elif act.get('type') == 'shell':
                    action_summary = f"shell: {act.get('command','')[:60]}"
            except Exception:
                action_summary = executed_action_json[:80]

            thumb = _make_history_thumbnail(screenshot_b64) if screenshot_b64 else None
            new_action_history.append({
                "step": steps + 1,
                "action_summary": action_summary,
                "action_json": executed_action_json,
                "screenshot_thumb": thumb,
                "screenshot_b64": screenshot_b64,
            })
            if len(new_action_history) > max_history + 1:
                new_action_history = new_action_history[-(max_history + 1):]

        return {
            "steps_taken": steps + 1,
            "last_screenshot_b64": screenshot_b64,
            "is_complete": is_complete,
            "result_summary": result_summary,
            "trajectory": new_trajectory,
            "messages": [HumanMessage(content=user_content), response],
            "action_history": new_action_history,
            "failed_elements": failed_elements,
        }

    return think_act


def _should_continue(state: ComputerUseState) -> str:
    """Router: continúa el loop o termina."""
    if state.get("is_complete", False):
        return END
    if state["steps_taken"] >= settings.computer_use_max_steps:
        return END
    return "observe"


# -- Factory -------------------------------------------------------------------

def create_computer_use_agent(
    vision_llm: BaseChatModel,
    vl_replay_buffer: Optional[VLReplayBuffer] = None,
) -> CompiledStateGraph:
    """
    Construye el grafo LangGraph del Computer Use Agent (Sistema 1).

    Args:
        vision_llm: Instancia de ChatOpenAI configurada con el modelo VL
                    (Qwen2.5-VL local cargado en vLLM).
        vl_replay_buffer: Buffer donde se guardan los steps para re-training.
                           Si es None, no se guardan los steps.

    Returns:
        CompiledStateGraph listo para ser invocado con:
        {
          "instruction": "Open MB51 in SAP, enter CRUDE-100, update stock...",
          "messages": [],
          "steps_taken": 0,
          "last_screenshot_b64": None,
          "trajectory": [],
          "is_complete": False,
          "result_summary": "",
        }
    """
    init_node = _build_init_node()
    observe_node = _build_observe_node(vision_llm)
    think_act_node = _build_think_act_node(vision_llm, vl_replay_buffer)

    graph = StateGraph(ComputerUseState)
    graph.add_node("init", init_node)
    graph.add_node("observe", observe_node)
    graph.add_node("think_act", think_act_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "observe")
    graph.add_edge("observe", "think_act")
    graph.add_conditional_edges("think_act", _should_continue, {END: END, "observe": "observe"})

    return graph.compile()


async def run_computer_use_task(
    instruction: str,
    vision_llm: BaseChatModel,
    vl_replay_buffer: Optional[VLReplayBuffer] = None,
    config: Optional[RunnableConfig] = None,
) -> str:
    """
    Helper de alto nivel para ejecutar una tarea de computer use completa.

    Args:
        instruction: Instrucción de alto nivel del Orchestrator.
        vision_llm: Modelo VL inicializado.
        vl_replay_buffer: Buffer para guardar el dataset de training.
        config: RunnableConfig de LangGraph (thread_id, user_id, etc.)

    Returns:
        Resumen del resultado de la tarea.
    """
    agent = create_computer_use_agent(vision_llm, vl_replay_buffer)

    initial_state = {
        "instruction": instruction,
        "messages": [],
        "steps_taken": 0,
        "last_screenshot_b64": None,
        "trajectory": [],
        "is_complete": False,
        "result_summary": "",
        "action_history": [],
        "failed_elements": [],
        "parsed_elements": None,
        "annotated_screenshot_b64": None,
    }

    try:
        final_state = await agent.ainvoke(initial_state, config=config)
        result = final_state.get("result_summary", "Tarea completada sin resumen.")
        steps = final_state.get("steps_taken", 0)
        trajectory_count = len(final_state.get("trajectory", []))
        logger.info(
            f"[ComputerUse] Tarea finalizada en {steps} steps. "
            f"{trajectory_count} steps guardados en VL buffer."
        )
        return result
    except Exception as e:
        logger.error(f"[ComputerUse] Error ejecutando tarea: {e}")
        return f"Error en computer use agent: {e}"

