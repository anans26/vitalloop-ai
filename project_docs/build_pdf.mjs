import { readFileSync, writeFileSync } from "node:fs";
import { marked } from "marked";

const SRC = "F:/MLOPS PROJECT/project_docs/PROJECT_DESIGN.md";
const OUT_HTML = new URL("./design.html", import.meta.url).pathname.replace(/^\//, "");

const md = readFileSync(SRC, "utf-8");

marked.setOptions({ gfm: true, breaks: false });
let body = marked.parse(md);

// Convert fenced mermaid blocks into <pre class="mermaid"> for client-side rendering
body = body.replace(
  /<pre><code class="language-mermaid">([\s\S]*?)<\/code><\/pre>/g,
  (_, code) => {
    const decoded = code
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'");
    return `<pre class="mermaid">${decoded
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")}</pre>`;
  }
);

const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VitalLoop 2.0 — Technical Design Specification</title>
<style>
  @page {
    size: A4;
    margin: 22mm 18mm 22mm 18mm;
    @bottom-center { content: "VitalLoop 2.0 · Technical Design Specification — Page " counter(page); font-size: 8.5pt; color: #6b7280; font-family: "Segoe UI", Arial, sans-serif; }
  }
  * { box-sizing: border-box; }
  html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body {
    font-family: "Segoe UI", "Calibri", Arial, sans-serif;
    font-size: 10.5pt; line-height: 1.55; color: #1f2937;
    max-width: 100%; margin: 0;
  }
  h1, h2, h3, h4 { font-family: "Segoe UI Semibold", "Segoe UI", Arial, sans-serif; color: #0f3057; line-height: 1.25; }
  h1 { font-size: 24pt; margin: 0.4em 0; }
  h2 { font-size: 16pt; border-bottom: 2.5px solid #0f3057; padding-bottom: 5px; margin-top: 1.6em; break-after: avoid; }
  h3 { font-size: 12.5pt; margin-top: 1.3em; break-after: avoid; }
  h4 { font-size: 11pt; break-after: avoid; }
  /* Cover page: the first centered div */
  body > div[align="center"]:first-of-type {
    min-height: 240mm; display: flex; flex-direction: column; justify-content: center;
    page-break-after: always; text-align: center;
  }
  body > div[align="center"]:first-of-type h1 { font-size: 34pt; color: #0f3057; margin-bottom: 0; }
  body > div[align="center"]:first-of-type h2 { font-size: 17pt; border: none; color: #2c5f8a; font-weight: 500; }
  body > div[align="center"]:first-of-type h3 { font-size: 12pt; color: #6b7280; font-weight: 400; }
  body > div[align="center"]:first-of-type table { margin: 2em auto 0 auto; text-align: left; }
  table { border-collapse: collapse; width: 100%; margin: 0.9em 0; font-size: 9.3pt; }
  th { background: #0f3057; color: #fff; text-align: left; padding: 6px 9px; }
  td { border: 1px solid #d1d5db; padding: 5px 9px; vertical-align: top; }
  tr:nth-child(even) td { background: #f3f6fa; }
  table, pre { break-inside: auto; }
  tr { break-inside: avoid; }
  code { font-family: "Consolas", "Courier New", monospace; font-size: 9pt; background: #eef2f7; padding: 1px 4px; border-radius: 3px; }
  pre { background: #f6f8fa; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px 12px; overflow-x: hidden; white-space: pre-wrap; word-wrap: break-word; }
  pre code { background: none; padding: 0; font-size: 8.6pt; line-height: 1.45; }
  pre.mermaid { background: #ffffff; border: 1px solid #e5e7eb; text-align: center; break-inside: avoid; padding: 8px; }
  pre.mermaid svg { max-width: 100% !important; height: auto !important; max-height: 225mm; }
  blockquote { border-left: 4px solid #2c5f8a; background: #f0f6fc; margin: 0.9em 0; padding: 8px 14px; color: #334155; border-radius: 0 6px 6px 0; }
  blockquote p { margin: 0.3em 0; }
  a { color: #1d4ed8; text-decoration: none; }
  hr { border: none; border-top: 1px solid #d1d5db; margin: 1.6em 0; }
  li { margin: 0.22em 0; }
  strong { color: #111827; }
</style>
</head>
<body>
${body}
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs";
  mermaid.initialize({ startOnLoad: true, theme: "neutral", flowchart: { useMaxWidth: true }, securityLevel: "loose" });
  window.__mermaid_loaded = true;
</script>
</body>
</html>`;

writeFileSync("design.html", html, "utf-8");
console.log("HTML written:", "design.html", html.length, "chars");
