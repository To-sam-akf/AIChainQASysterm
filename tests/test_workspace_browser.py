"""Opt-in browser checks against a separately started local Vite server.

AIKA_WEB_TEST_URL=http://127.0.0.1:5179 pytest -q tests/test_workspace_browser.py
All application API calls are intercepted; no user conversations are written.
"""
import base64
import json
import os
import shutil

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def page():
    url = os.getenv("AIKA_WEB_TEST_URL")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not url or not chrome:
        pytest.skip("set AIKA_WEB_TEST_URL and install Chrome/Chromium to run browser checks")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome, headless=True, args=["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader"])
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        node_id = base64.urlsafe_b64encode(json.dumps(["IndustryConcept", "液冷"]).encode()).decode().rstrip("=")
        node = {"id": node_id, "name": "液冷", "type": "IndustryConcept"}
        conversation = {"id": "fixture-session", "title": "测试对话", "created_at": "2026-11-27T10:00:00", "updated_at": "2026-11-27T10:00:00", "turn_count": 0, "preview": "", "turns": [], "pinned": False}
        doc = {"document_id": "fixture", "title": "测试资料", "source_type": "text", "source_url": "", "filename": "", "status": "ready", "semantic_status": "not_configured", "error": "", "active_version": 1, "tags": [], "topic": "", "company": "", "raw_text": "测试资料原文", "created_at": "2026-11-27"}

        def api(route):
            path = route.request.url.split("/api/", 1)[-1]
            if path.startswith("knowledge/documents?"):
                data = {"documents": [doc]}
            elif path.startswith("knowledge/documents/fixture"):
                data = doc
            elif path.startswith("graph/entities?"):
                data = {"items": [node]}
            elif path.startswith("graph/entities/"):
                data = {"node": node, "items": [], "total": 0, "offset": 0}
            elif path.startswith("graph/summary"):
                data = {"relation_options": {"全部关系": ""}}
            elif path == "status":
                data = {"settings": {"thinking_enabled": False, "reasoning_effort": "low", "reasoning_efforts": ["low", "medium", "high"]}}
            elif path == "examples":
                data = {"examples": []}
            elif path == "conversations":
                data = {"conversations": [conversation]}
            elif path == "conversations/fixture-session/pin":
                conversation["pinned"] = json.loads(route.request.post_data)["pinned"]
                data = conversation
            elif path == "conversations/fixture-session":
                data = conversation
            elif path == "agent/roles":
                data = {"roles": []}
            else:
                route.fulfill(status=404, content_type="application/json", body="{}")
                return
            route.fulfill(status=200, content_type="application/json", body=json.dumps(data))

        page.route("**/api/**", api)
        page.goto(url, wait_until="networkidle")
        yield page
        browser.close()
        assert not errors, "browser errors: " + "; ".join(errors)


def test_inspector_layout_and_graph_survive_navigation(page):
    page.get_by_role("button", name="知识库", exact=True).click()
    page.locator(".knowledge-item .context-host > button").first.click()
    panel = page.get_by_role("complementary", name="资料详情")
    panel.wait_for()
    original = panel.bounding_box()["width"]
    panel.get_by_role("separator").focus()
    page.keyboard.press("ArrowLeft")
    assert panel.bounding_box()["width"] > original
    page.get_by_role("button", name="紧凑视图").click()
    page.reload(wait_until="networkidle")
    page.get_by_role("button", name="知识库", exact=True).click()
    page.get_by_role("button", name="详细视图").wait_for()
    assert page.get_by_role("button", name="详细视图").is_visible()
    page.get_by_role("button", name="产业链图谱", exact=True).click()
    page.get_by_role("button", name="液冷", exact=True).first.click()
    page.locator(".space-graph canvas").wait_for()
    page.evaluate("window.testCanvas = document.querySelector('.space-graph canvas')")
    page.get_by_role("button", name="列表视图").click()
    page.get_by_role("button", name="图谱视图").click()
    page.get_by_role("button", name="数据概览", exact=True).click()
    page.get_by_role("button", name="产业链图谱", exact=True).click()
    assert page.evaluate("window.testCanvas === document.querySelector('.space-graph canvas')")
    assert "液冷" in page.get_by_role("complementary", name="详情").inner_text()
    page.set_viewport_size({"width": 390, "height": 844})
    assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
    page.get_by_role("button", name="切换侧栏").click()
    page.wait_for_timeout(300)
    assert page.locator(".sidebar").bounding_box()["x"] == 0


def test_context_pin_and_command_keyboard(page):
    item = page.locator(".history-item").filter(has_text="测试对话")
    item.wait_for()
    item.click(button="right")
    page.get_by_role("menuitem", name="置顶", exact=True).click()
    page.get_by_role("button", name="对话操作：测试对话").click()
    page.get_by_role("menuitem", name="取消置顶", exact=True).wait_for()
    page.keyboard.press("Escape")
    assert not page.get_by_role("menu").count()
    page.keyboard.press("Control+k")
    page.locator(".command-dialog input").fill("数据概览")
    page.keyboard.press("Enter")
    assert page.locator(".page-header h1").inner_text() == "数据概览"
    page.keyboard.press("Control+b")
    page.reload(wait_until="networkidle")
    assert "sidebar-hidden" in page.locator(".app-shell").get_attribute("class")
