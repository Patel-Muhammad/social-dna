"""Capture real screenshots of the running app (needs: pip install playwright and Google Chrome).
Usage: start the app (streamlit run app.py --server.port 8599) then run this script."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8599"
PAGES = [("Overview", "01_overview"), ("Community Explorer", "02_explorer"), ("Community Profiles", "03_profiles"),
         ("Community Comparison", "04_comparison"), ("Influence Analysis", "05_influence"),
         ("AI Community Analyst", "06_ai_analyst"), ("Methodology & Dataset", "07_methodology")]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("text=Overview", timeout=120000)
        for label, name in PAGES:
            page.get_by_role("link", name=label).first.click()
            page.wait_for_timeout(6000)
            if name == "06_ai_analyst":
                page.get_by_role("button", name="Compare community 2 and community 4.").click()
                page.wait_for_timeout(9000)
            page.screenshot(path=str(OUT / f"{name}.png"), full_page=(name != "02_explorer"))
            print("saved", name)
        browser.close()


if __name__ == "__main__":
    main()
