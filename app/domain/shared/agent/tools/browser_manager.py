"""
Playwright Browser Manager — Hybrid Vision Pipeline
=====================================================
Singleton async manager for a persistent Chromium browser inside Docker.

This enables the Computer Use agent to interact with web applications
using **structured DOM data** (Accessibility Tree) in addition to raw
screenshots, matching the architecture of Anthropic Computer Use,
OpenAI Operator, and the browser-use framework (2026 state-of-the-art).

Why this exists:
  - Raw pixel capture (mss) + coordinate clicking (xdotool) is fragile
    for web tasks. Playwright gives us:
      1. Clean viewport-only screenshots (no Xvfb chrome/taskbar noise)
      2. Accessibility Tree → semantic element list for the VLM
      3. Reliable programmatic actions (click by role, fill by label)
  - Native desktop apps (SAP GUI, terminals) still use mss + xdotool.
  - The agent dynamically selects the right backend per action.

Lifecycle:
  - Created lazily on first use (not at FastAPI startup).
  - Persistent context preserves cookies/sessions across events.
  - Closed gracefully via shutdown() from FastAPI lifespan.

Usage:
  manager = get_browser_manager()
  await manager.ensure_ready()
  page = manager.page
  screenshot_b64 = await manager.screenshot_b64()
  tree_yaml = await manager.accessibility_snapshot()
"""

import asyncio
import base64
from typing import Optional

from loguru import logger

from app.core.config import settings


class PlaywrightBrowserManager:
    """
    Singleton async manager for a persistent Playwright Chromium browser.

    Thread-safety: all public methods are async and guarded by _lock.
    Only one browser instance exists at a time (persistent context).
    """

    _instance: Optional["PlaywrightBrowserManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._playwright = None
        self._context = None       # BrowserContext (persistent)
        self._page = None          # Primary Page
        self._lock = asyncio.Lock()
        self._ready = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def page(self):
        """The current active page. None if browser not started."""
        return self._page

    @property
    def is_ready(self) -> bool:
        return self._ready and self._page is not None and not self._page.is_closed()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def ensure_ready(self) -> "PlaywrightBrowserManager":
        """
        Lazily launch the browser if not already running.
        Safe to call multiple times — idempotent.
        """
        if self.is_ready:
            return self

        async with self._lock:
            # Double-check after acquiring lock
            if self.is_ready:
                return self

            try:
                from playwright.async_api import async_playwright

                logger.info("[BrowserManager] Launching Playwright Chromium...")

                self._playwright = await async_playwright().start()

                # Persistent context preserves cookies/sessions across events.
                # user_data_dir matches the Chromium profile used by xdotool/shell.
                user_data_dir = "/tmp/chromium-pw-profile"

                launch_args = [
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-extensions",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                ]

                viewport = {"width": 1920, "height": 1080}

                self._context = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    headless=settings.playwright_headless,
                    args=launch_args,
                    viewport=viewport,
                    locale="es-ES",
                    timezone_id="America/Caracas",
                    ignore_https_errors=True,
                    # Accessibility tree needs a real display, not a void
                    no_viewport=False,
                )

                # Use the first page if one exists, else create one
                if self._context.pages:
                    self._page = self._context.pages[0]
                else:
                    self._page = await self._context.new_page()

                self._ready = True
                logger.success(
                    f"[BrowserManager] ✓ Chromium ready — "
                    f"viewport {viewport['width']}×{viewport['height']}, "
                    f"headless={settings.playwright_headless}"
                )

            except ImportError:
                logger.error(
                    "[BrowserManager] playwright not installed. "
                    "Run: pip install playwright && playwright install chromium"
                )
                self._ready = False
            except Exception as e:
                logger.error(f"[BrowserManager] Failed to launch browser: {e}")
                self._ready = False

        return self

    async def shutdown(self):
        """Gracefully close the browser. Called from FastAPI lifespan shutdown."""
        async with self._lock:
            try:
                if self._context:
                    await self._context.close()
                if self._playwright:
                    await self._playwright.stop()
            except Exception as e:
                logger.warning(f"[BrowserManager] Error during shutdown: {e}")
            finally:
                self._context = None
                self._page = None
                self._playwright = None
                self._ready = False
                logger.info("[BrowserManager] Shutdown complete.")

    # ── Observation: Screenshots ──────────────────────────────────────────────

    async def screenshot_b64(self, full_page: bool = False) -> str:
        """
        Capture the current viewport as a base64 PNG string.

        Unlike mss, this captures ONLY the browser viewport — no Xvfb
        window decorations, taskbar, or other noise. The result is a
        cleaner image for the VLM to analyze.

        Args:
            full_page: If True, captures the entire scrollable page.
                       If False (default), captures only the visible viewport.

        Returns:
            Base64-encoded PNG string (no data:image/png prefix).
        """
        if not self.is_ready:
            raise RuntimeError("[BrowserManager] Browser not ready. Call ensure_ready() first.")

        try:
            screenshot_bytes = await self._page.screenshot(
                full_page=full_page,
                type="png",
            )
            return base64.b64encode(screenshot_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"[BrowserManager] Screenshot failed: {e}")
            raise

    # ── Observation: Accessibility Tree ────────────────────────────────────────

    async def accessibility_snapshot(self) -> str:
        """
        Capture the accessibility tree of the current page as a YAML string.

        This is the key differentiator from pure pixel-based agents:
        the VLM receives a structured, semantic list of interactive elements
        (buttons, inputs, links with their names and states) instead of
        having to guess from raw pixels.

        Format (Playwright aria_snapshot YAML):
          - heading "Gmail" [level=1]
          - button "Compose"
          - textbox "Search mail" [value=""]
          - link "Inbox (3)"
          - link "Sent"

        Returns:
            YAML string of the accessibility tree, or a fallback message.
        """
        if not self.is_ready:
            return "(Browser not ready)"

        try:
            # page.accessibility.snapshot() returns a full JSON tree.
            # We convert it to a compact text format for the VLM prompt.
            snapshot = await self._page.accessibility.snapshot(interesting_only=True)
            if not snapshot:
                return "(Empty accessibility tree — page may still be loading)"

            return self._format_a11y_tree(snapshot, depth=0)

        except Exception as e:
            logger.warning(f"[BrowserManager] Accessibility snapshot failed: {e}")
            return f"(Accessibility tree unavailable: {e})"

    def _format_a11y_tree(self, node: dict, depth: int = 0, max_depth: int = 5) -> str:
        """
        Recursively format an accessibility tree node into compact text.

        Produces output like:
          [1] button "Compose"
          [2] textbox "Search mail" [value=""]
          [3] link "Inbox (3)"
          [4] heading "Primary" [level=2]
            [5] link "Email from John" 
            [6] link "Meeting tomorrow"
        """
        if depth > max_depth:
            return ""

        lines = []
        role = node.get("role", "")
        name = node.get("name", "")
        children = node.get("children", [])

        # Skip generic/structural nodes that add noise
        skip_roles = {"generic", "none", "presentation", "group", "main", 
                      "navigation", "banner", "contentinfo", "complementary",
                      "region", "section", "article"}

        # Include this node if it has a meaningful role
        include = role and role not in skip_roles
        # Always include if it has a name (even if role is generic)
        if name and name.strip():
            include = True

        indent = "  " * depth

        if include and role:
            attrs = []
            for attr in ["value", "checked", "selected", "disabled", "expanded", 
                         "level", "pressed", "readonly"]:
                if attr in node:
                    val = node[attr]
                    if isinstance(val, bool):
                        if val:
                            attrs.append(attr)
                    else:
                        attrs.append(f'{attr}="{val}"')

            attr_str = f" [{', '.join(attrs)}]" if attrs else ""
            name_str = f' "{name}"' if name else ""
            lines.append(f'{indent}- {role}{name_str}{attr_str}')

        for child in children:
            child_text = self._format_a11y_tree(child, depth + (1 if include else 0), max_depth)
            if child_text:
                lines.append(child_text)

        return "\n".join(lines)

    # ── Observation: Page Info ────────────────────────────────────────────────

    async def page_info(self) -> dict:
        """Returns current URL, title, and viewport size."""
        if not self.is_ready:
            return {"url": "", "title": "", "viewport": ""}
        return {
            "url": self._page.url,
            "title": await self._page.title(),
            "viewport": f"{self._page.viewport_size['width']}×{self._page.viewport_size['height']}",
        }

    # ── Actions: Semantic (Playwright) ────────────────────────────────────────

    async def goto(self, url: str, wait_until: str = "domcontentloaded") -> str:
        """Navigate to a URL. Returns the final URL after redirects."""
        if not self.is_ready:
            raise RuntimeError("[BrowserManager] Browser not ready.")
        try:
            response = await self._page.goto(url, wait_until=wait_until, timeout=30000)
            status = response.status if response else "unknown"
            final_url = self._page.url
            logger.info(f"[BrowserManager] Navigated to {final_url} (HTTP {status})")
            return f"Navigated to {final_url} (HTTP {status})"
        except Exception as e:
            logger.error(f"[BrowserManager] Navigation failed: {e}")
            return f"ERROR navigating to {url}: {e}"

    async def click_by_role(self, role: str, name: str, exact: bool = False) -> str:
        """
        Click an element by its ARIA role and accessible name.
        
        Examples:
            click_by_role("button", "Compose")
            click_by_role("link", "Inbox")
            click_by_role("textbox", "Search mail")
        """
        if not self.is_ready:
            raise RuntimeError("[BrowserManager] Browser not ready.")
        try:
            locator = self._page.get_by_role(role, name=name, exact=exact)
            await locator.click(timeout=10000)
            logger.info(f"[BrowserManager] Clicked {role} '{name}'")
            return f"Clicked {role} '{name}'"
        except Exception as e:
            logger.warning(f"[BrowserManager] click_by_role failed: {e}")
            return f"ERROR clicking {role} '{name}': {e}"

    async def click_by_text(self, text: str, exact: bool = False) -> str:
        """Click an element by its visible text content."""
        if not self.is_ready:
            raise RuntimeError("[BrowserManager] Browser not ready.")
        try:
            locator = self._page.get_by_text(text, exact=exact)
            await locator.first.click(timeout=10000)
            logger.info(f"[BrowserManager] Clicked text '{text}'")
            return f"Clicked element with text '{text}'"
        except Exception as e:
            logger.warning(f"[BrowserManager] click_by_text failed: {e}")
            return f"ERROR clicking text '{text}': {e}"

    async def fill_field(self, role: str, name: str, value: str) -> str:
        """
        Fill a form field by its ARIA role and accessible name.

        Examples:
            fill_field("textbox", "To", "ops@plant.com")
            fill_field("textbox", "Subject", "Equipment Report")
        """
        if not self.is_ready:
            raise RuntimeError("[BrowserManager] Browser not ready.")
        try:
            locator = self._page.get_by_role(role, name=name)
            await locator.fill(value, timeout=10000)
            logger.info(f"[BrowserManager] Filled {role} '{name}' with '{value[:40]}'")
            return f"Filled {role} '{name}' with '{value[:60]}'"
        except Exception as e:
            logger.warning(f"[BrowserManager] fill_field failed: {e}")
            return f"ERROR filling {role} '{name}': {e}"

    async def press_key(self, key: str) -> str:
        """Press a keyboard key. Supports: Enter, Tab, Escape, etc."""
        if not self.is_ready:
            raise RuntimeError("[BrowserManager] Browser not ready.")
        try:
            await self._page.keyboard.press(key)
            logger.info(f"[BrowserManager] Pressed key '{key}'")
            return f"Pressed key '{key}'"
        except Exception as e:
            return f"ERROR pressing key '{key}': {e}"

    async def type_text(self, text: str, delay: float = 30) -> str:
        """Type text character by character (simulates real typing)."""
        if not self.is_ready:
            raise RuntimeError("[BrowserManager] Browser not ready.")
        try:
            await self._page.keyboard.type(text, delay=delay)
            logger.info(f"[BrowserManager] Typed '{text[:40]}'")
            return f"Typed '{text[:60]}'"
        except Exception as e:
            return f"ERROR typing text: {e}"

    async def scroll(self, direction: str = "down", amount: int = 3) -> str:
        """Scroll the page. direction: 'up' or 'down'. amount: number of wheel ticks."""
        if not self.is_ready:
            raise RuntimeError("[BrowserManager] Browser not ready.")
        try:
            delta = amount * 100 * (1 if direction == "down" else -1)
            await self._page.mouse.wheel(0, delta)
            logger.info(f"[BrowserManager] Scrolled {direction} ×{amount}")
            return f"Scrolled {direction} ×{amount}"
        except Exception as e:
            return f"ERROR scrolling: {e}"

    async def click_coordinates(self, x: int, y: int) -> str:
        """Click at exact pixel coordinates (fallback for elements without good ARIA labels)."""
        if not self.is_ready:
            raise RuntimeError("[BrowserManager] Browser not ready.")
        try:
            await self._page.mouse.click(x, y)
            logger.info(f"[BrowserManager] Clicked at ({x}, {y})")
            return f"Clicked at ({x}, {y})"
        except Exception as e:
            return f"ERROR clicking at ({x}, {y}): {e}"

    async def wait_for_load(self, timeout: int = 5000) -> str:
        """Wait for network idle (page fully loaded)."""
        if not self.is_ready:
            return "Browser not ready"
        try:
            await self._page.wait_for_load_state("networkidle", timeout=timeout)
            return "Page loaded (network idle)"
        except Exception:
            return "Page load timeout — may still be loading"

    async def get_page_text(self, max_chars: int = 3000) -> str:
        """Extract visible text content from the page (for non-visual analysis)."""
        if not self.is_ready:
            return ""
        try:
            text = await self._page.inner_text("body")
            return text[:max_chars] if text else ""
        except Exception:
            return ""


# ── Module-level singleton ────────────────────────────────────────────────────

_browser_manager = PlaywrightBrowserManager()


def get_browser_manager() -> PlaywrightBrowserManager:
    """Returns the singleton PlaywrightBrowserManager instance."""
    return _browser_manager
