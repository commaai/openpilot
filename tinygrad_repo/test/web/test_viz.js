const { spawn } = require("child_process");
const puppeteer = require("puppeteer");

async function main() {
  // ** start viz server
  const proc = spawn("python", ["-u", "-c", "from tinygrad import Tensor; Tensor.arange(4).realize()"], { env: { ...process.env, VIZ:"1" },
                      stdio: ["inherit", "pipe", "inherit"]});
  await new Promise(resolve => proc.stdout.on("data", r => {
    if (r.includes("ready")) resolve();
  }));

  // ** run browser tests
  let browser, page;
  try {
    browser = await puppeteer.launch({ headless: true });
    page = await browser.newPage();
    const res = await page.goto("http://localhost:8000", { waitUntil:"domcontentloaded" });
    if (res.status() !== 200) throw new Error("Failed to load page");
    const scheduleSelector = await page.waitForSelector("ul:nth-of-type(2)");
    scheduleSelector.click();
    await page.waitForSelector("rect");
    await page.waitForFunction(() => {
      const nodes = document.querySelectorAll("#nodes > g").length;
      const edges = document.querySelectorAll("#edges > path").length;
      return nodes > 0 && edges > 0;
    });
  } finally {
    // ** cleanups
    if (page != null) await page.close();
    if (browser != null) await browser.close();
    proc.kill();
  }
}

main();
