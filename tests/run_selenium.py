#!/usr/bin/env python3
"""Run the browser test pages in Firefox via Selenium, report each verdict."""
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

# cheap pure-math page first, then the engine/app page
TEST_PAGES = ["limb_array_math_test.html", "fractaldive_test.html"]
TIMEOUT_S = 600  # GPU frame sections can be slow


def run_page(driver, page_name: str) -> bool:
    """Open one test page, print its log, return True on pass."""
    url = Path(__file__).with_name(page_name).resolve().as_uri()
    driver.get(url)
    wait = WebDriverWait(driver, TIMEOUT_S)
    wait.until(lambda d: (cls := d.find_element(By.ID, "test-summary").get_attribute("class"))
               in ("pass", "fail"))
    summary = driver.find_element(By.ID, "test-summary")
    print(driver.find_element(By.ID, "test-log").text)
    print("=" * 60)
    print(f"SUMMARY [{page_name}]: {summary.text} [{summary.get_attribute('class')}]")
    return summary.get_attribute("class") == "pass"


def main() -> int:
    opts = webdriver.FirefoxOptions()
    # WebGPU is off by default in Firefox release on Linux
    opts.set_preference("dom.webgpu.enabled", True)
    opts.set_preference("gfx.webrender.all", True)
    opts.set_preference("dom.file.createInChild", True)
    driver = webdriver.Firefox(options=opts)
    try:
        # list() runs every page even after a failure — full report, then fail
        return 0 if all([run_page(driver, page) for page in TEST_PAGES]) else 1
    finally:
        driver.quit()


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"elapsed: {time.time() - t0:.1f}s")
    sys.exit(rc)
