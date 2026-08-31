import asyncio

from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
)

from agent.state import Step


class BrowserExecutor:

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        # Playwright's Page API is not safe to call concurrently from two
        # coroutines. One BrowserExecutor is shared across all tasks in
        # main.py, so every action against the page is serialized here.
        self.lock = asyncio.Lock()

    # =========================================================
    # START BROWSER
    # =========================================================

    async def start(self):
        """Start Playwright and create browser/page."""
        await self.close()
        print("🚀 Starting Playwright...")

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        self.context = await self.browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            ),
        )
        # Playwright's default context exposes navigator.webdriver = true,
        # which is the #1 signal sites like Amazon use to fingerprint and
        # kill automated sessions. Strip it before any page script runs.
        await self.context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )
        self.page = await self.context.new_page()
        self.page.on("crash", lambda: print("💥 Page CRASHED (renderer died — often anti-bot JS on heavy sites)."))
        self.page.on("close", lambda: print("🚪 Page closed."))

        print("✅ Browser started successfully.")

    # =========================================================
    # ENSURE BROWSER
    # =========================================================

    async def ensure_browser(self):
        if (
            self.playwright is None
            or self.browser is None
            or self.context is None
            or self.page is None
        ):
            await self.start()
            return

        try:
            await self.page.title()
        except Exception as exc:
            print(f"⚠️ Browser/page is no longer available: {type(exc).__name__}: {exc}")
            print("🔄 Starting fresh browser...")
            await self.start()

    # =========================================================
    # EXECUTE STEP OBJECT
    # =========================================================

    async def execute(self, step: Step):
        await self.ensure_browser()

        try:
            print("\n====================================")
            print(f"🤖 STEP: {step.id}")
            print(f"⚙️ ACTION: {step.action}")
            print(f"🎯 TARGET: {step.target}")
            print(f"📝 VALUE: {step.value}")
            print(f"🌐 CURRENT URL: {self.page.url}")
            print("====================================")

            result = await self.execute_step(
                action=step.action, target=step.target, value=step.value
            )
            observation = await self._observation()
            result["observation"] = observation
            return result

        except PlaywrightTimeoutError as exc:
            observation = await self._observation()
            raise RuntimeError(
                f"""
Playwright timeout for step {step.id}

Action:
{step.action}

Target:
{step.target}

Value:
{step.value}

Current URL:
{self.page.url}

Browser observation:
{observation}
"""
            ) from exc

        except Exception as exc:
            observation = await self._observation()
            raise RuntimeError(
                f"""
Playwright execution failed.

Step:
{step.id}

Action:
{step.action}

Target:
{step.target}

Value:
{step.value}

Current URL:
{self.page.url}

Error:
{exc}

Browser observation:
{observation}
"""
            ) from exc

    # =========================================================
    # EXECUTE ACTION
    # =========================================================

    async def execute_step(self, action: str, target: str = "", value: str = ""):
        await self.ensure_browser()

        action = (action or "").lower().strip()
        target = target or ""
        value = value or ""

        async with self.lock:

            # =================================================
            # NAVIGATE
            # =================================================
            if action == "navigate":
                url = value.strip() or target.strip()
                if not url:
                    raise ValueError("Navigate action requires a URL.")
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url

                print(f"🌐 Navigating to: {url}")
                await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await self.page.wait_for_timeout(1000)

                return {"status": "success", "action": "navigate", "url": self.page.url}

            # =================================================
            # CLICK
            # =================================================
            elif action == "click":
                if not target:
                    raise ValueError("Click action requires target.")

                print(f"🖱️ Clicking: {target}")
                locator = await self._get_visible_locator(target)

                tag_name = await locator.evaluate("(el) => el.tagName.toLowerCase()")
                if tag_name in ("input", "textarea"):

                    print("⚠️ AI requested CLICK on input/textarea.")

                    if value:
                        print("🔄 Converting CLICK → FILL")
                        await locator.fill(value, timeout=10000)
                        return {"status": "success", "action": "fill", "converted_from": "click"}

                    # No new value given. If the field already has text in it,
                    # a bare click here is almost always a failed attempt to
                    # submit — clicking never submits a form. Press Enter
                    # instead, which is what the model actually wants.
                    try:
                        current_value = (await locator.input_value()).strip()
                    except Exception:
                        current_value = ""

                    if current_value:
                        print(f"🔄 Field already contains '{current_value}' — treating CLICK as submit (pressing Enter).")
                        await locator.press("Enter", timeout=10000)
                        await self.page.wait_for_timeout(1000)
                        return {"status": "success", "action": "press", "converted_from": "click", "key": "Enter"}

                await self._assert_reachable(locator, target)
                await locator.click(timeout=10000)
                await self.page.wait_for_timeout(500)

                return {"status": "success", "action": "click"}

            # =================================================
            # FILL
            # =================================================
            elif action == "fill":
                if not target:
                    raise ValueError("Fill action requires target.")

                print(f"⌨️ Filling: {target}")
                locator = await self._get_visible_locator(target)
                await locator.fill(value, timeout=10000)

                return {"status": "success", "action": "fill"}

            # =================================================
            # PRESS
            # =================================================
            elif action == "press":
                if not target:
                    raise ValueError("Press action requires target.")

                print(f"⌨️ Pressing {value} on {target}")
                locator = await self._get_visible_locator(target)
                await locator.press(value, timeout=10000)
                await self.page.wait_for_timeout(1000)

                return {"status": "success", "action": "press"}

            # =================================================
            # BACK  (declared as supported in agent.py but missing here before)
            # =================================================
            elif action == "back":
                print("⬅️ Navigating back")
                await self.page.go_back(wait_until="domcontentloaded", timeout=15000)
                await self.page.wait_for_timeout(500)

                return {"status": "success", "action": "back", "url": self.page.url}

            # =================================================
            # SCROLL  (declared as supported in agent.py but missing here before)
            # target: optional selector to scroll into view.
            # value: optional pixel amount ("600") or "up" / "down" / "bottom" / "top".
            # =================================================
            elif action == "scroll":
                if target:
                    print(f"↕️ Scrolling element into view: {target}")
                    locator = await self._get_visible_locator(target)
                    await locator.scroll_into_view_if_needed(timeout=10000)
                else:
                    direction = (value or "down").strip().lower()
                    print(f"↕️ Scrolling page: {direction}")
                    if direction == "top":
                        await self.page.evaluate("window.scrollTo(0, 0)")
                    elif direction == "bottom":
                        await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    elif direction == "up":
                        await self.page.evaluate("window.scrollBy(0, -600)")
                    elif direction.lstrip("-").isdigit():
                        await self.page.evaluate(f"window.scrollBy(0, {int(direction)})")
                    else:
                        await self.page.evaluate("window.scrollBy(0, 600)")
                await self.page.wait_for_timeout(300)

                return {"status": "success", "action": "scroll"}

            # =================================================
            # SELECT  (declared as supported in agent.py but missing here before)
            # target: selector of the <select>. value: option value/label/index.
            # =================================================
            elif action == "select":
                if not target:
                    raise ValueError("Select action requires target.")

                print(f"🔽 Selecting '{value}' on {target}")
                locator = await self._get_visible_locator(target)
                try:
                    await locator.select_option(value, timeout=10000)
                except Exception:
                    # Fall back to matching by visible label.
                    await locator.select_option(label=value, timeout=10000)

                return {"status": "success", "action": "select"}

            # =================================================
            # WAIT
            # =================================================
            elif action == "wait":
                milliseconds = int(value or 1000)
                print(f"⏳ Waiting {milliseconds} ms")
                await self.page.wait_for_timeout(milliseconds)

                return {"status": "success", "action": "wait"}

            # =================================================
            # EXTRACT TEXT
            # =================================================
            elif action == "extract_text":
                if not target:
                    raise ValueError("extract_text requires target.")

                locator = await self._get_visible_locator(target)
                text = await locator.inner_text(timeout=10000)

                return {"status": "success", "action": "extract_text", "text": text}

            # =================================================
            # DONE
            # =================================================
            elif action == "done":
                print("✅ Agent completed task.")
                return {"status": "completed", "action": "done"}

            # =================================================
            # UNSUPPORTED
            # =================================================
            else:
                raise ValueError(f"Unsupported action: {action}")

    # =========================================================
    # SMART LOCATOR
    # =========================================================

    async def _assert_reachable(self, locator, target: str):
        """Fail fast (well under a second) if an element is positioned
        somewhere it can never actually be scrolled into view — instead of
        letting Playwright's click retry loop burn its full ~10s timeout
        discovering the same thing. This mainly catches off-screen
        accessibility-only elements that pass is_visible() but aren't truly
        reachable (get_page_snapshot filters most of these out already;
        this is a defense-in-depth check for anything that slips through
        or changes between snapshot and execution)."""
        try:
            viewport = self.page.viewport_size or {"width": 1440, "height": 900}
            box = await locator.bounding_box(timeout=2000)
        except Exception:
            return  # let the real click attempt surface its own error
        if box is None:
            return
        if box["x"] < -200 or box["y"] < -200 or box["x"] > viewport["width"] + 2000:
            raise RuntimeError(
                f"Element for target '{target}' is positioned off-screen "
                f"(x={box['x']}, y={box['y']}) and cannot actually be clicked "
                f"— it is likely a hidden accessibility-only element."
            )

    async def _get_visible_locator(self, target: str):
        if not target:
            raise ValueError("Target selector is required.")

        print(f"🔎 Finding visible element: {target}")
        locator = self.page.locator(target)
        count = await locator.count()

        if count > 0:
            for index in range(count):
                candidate = locator.nth(index)
                try:
                    if await candidate.is_visible():
                        print(f"✅ Found visible element at index {index}")
                        return candidate
                except Exception:
                    continue

        print(f"🔄 CSS selector not usable. Trying text: {target}")
        try:
            text_locator = self.page.get_by_text(target, exact=True).first
            if await text_locator.is_visible():
                return text_locator
        except Exception:
            pass

        for role in ["button", "link"]:
            try:
                role_locator = self.page.get_by_role(role, name=target, exact=True).first
                if await role_locator.is_visible():
                    return role_locator
            except Exception:
                continue

        raise RuntimeError(
            f"""
Element not found or not visible.

Selector:
{target}

Current URL:
{self.page.url}

Page title:
{await self.page.title()}
"""
        )

    # =========================================================
    # OBSERVATION
    # =========================================================

    async def _observation(self):
        if not self.page:
            return "No browser page available."

        try:
            title = await self.page.title()
            url = self.page.url
            body_text = await self.page.locator("body").inner_text(timeout=5000)

            return f"URL: {url}\nTitle: {title}\nVisible text:\n{body_text[:5000]}"
        except Exception as exc:
            return f"Unable to capture browser observation: {exc}"

    # =========================================================
    # PAGE SNAPSHOT FOR GEMMA
    # =========================================================

    async def get_page_snapshot(self):
        await self.ensure_browser()

        url = self.page.url
        title = await self.page.title()
        viewport = self.page.viewport_size or {"width": 1440, "height": 900}
        elements = await self.page.locator("input, button, a, select, textarea").all()

        snapshot = []
        for index, element in enumerate(elements[:100]):
            try:
                if not await element.is_visible():
                    continue

                # tabindex="-1" marks an element as not meant for direct
                # user interaction (often a hidden accessibility helper,
                # like Amazon's off-screen keyboard-shortcut menu items).
                # is_visible() alone doesn't catch these — they pass CSS
                # visibility checks but are positioned off-canvas and can
                # never actually be scrolled into view or clicked.
                tabindex = await element.get_attribute("tabindex")
                if tabindex == "-1":
                    continue

                box = await element.bounding_box()
                if box is None:
                    continue
                # A large negative/oversized offset is the standard
                # "visually hidden but present for screen readers" trick.
                # Real, clickable page content stays within a reasonable
                # margin of the actual viewport.
                if (
                    box["width"] <= 0 or box["height"] <= 0
                    or box["x"] < -200 or box["y"] < -200
                    or box["x"] > viewport["width"] + 2000
                ):
                    continue

                tag = await element.evaluate("(el) => el.tagName.toLowerCase()")
                text = (await element.inner_text()).strip()
                aria_label = await element.get_attribute("aria-label")
                placeholder = await element.get_attribute("placeholder")
                element_id = await element.get_attribute("id")
                name = await element.get_attribute("name")
                element_type = await element.get_attribute("type")
                href = await element.get_attribute("href") if tag == "a" else None

                snapshot.append(
                    {
                        "index": index,
                        "tag": tag,
                        "text": text[:200],
                        "aria_label": aria_label,
                        "placeholder": placeholder,
                        "id": element_id,
                        "name": name,
                        "type": element_type,
                        "href": href,
                    }
                )
            except Exception:
                continue

        return {"url": url, "title": title, "elements": snapshot}

    # =========================================================
    # HUMAN INTERVENTION DETECTION
    # =========================================================

    async def detect_intervention(self):
        """Detect common browser states that require a real human.

        This is intentionally detection-only. The agent never attempts to
        solve or bypass CAPTCHA/security verification.
        """
        await self.ensure_browser()

        try:
            body = (await self.page.locator("body").inner_text(timeout=3000)).lower()
        except Exception:
            body = ""

        signals = [
            ("CAPTCHA", ["captcha", "recaptcha", "hcaptcha"]),
            ("HUMAN_VERIFICATION", ["are you human", "verify you are human", "verify you're human", "i'm not a robot", "im not a robot", "human verification"]),
            ("SECURITY_CHECK", ["security check", "checking your browser", "bot detection", "unusual traffic"]),
            ("OTP", ["one-time password", "one time password", "verification code", "enter otp"]),
        ]

        for kind, patterns in signals:
            for pattern in patterns:
                if pattern in body:
                    return {"required": True, "type": kind, "reason": pattern}

        return {"required": False}

    # =========================================================
    # CLOSE
    # =========================================================

    async def close(self):
        try:
            if self.page:
                await self.page.close()
        except Exception as exc:
            print(f"⚠️ Page close warning: {exc}")

        try:
            if self.context:
                await self.context.close()
        except Exception as exc:
            print(f"⚠️ Context close warning: {exc}")

        try:
            if self.browser:
                await self.browser.close()
        except Exception as exc:
            print(f"⚠️ Browser close warning: {exc}")

        try:
            if self.playwright:
                await self.playwright.stop()
        except Exception as exc:
            print(f"⚠️ Playwright stop warning: {exc}")

        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None

        print("🛑 Browser executor closed.")
