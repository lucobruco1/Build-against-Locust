#!/usr/bin/env node
/**
 * Is this deployment actually able to run the game?
 *
 *   node tools/check-static.mjs                       # local static serve of the repo
 *   node tools/check-static.mjs https://xxx.netlify.app
 *   PORT=3000 node tools/check-static.mjs http://localhost:3000
 *
 * It walks the page the way a browser does: fetch the document, read its
 * `<link>` / `<script src>` tags and its import map, then follow every module
 * specifier in every file it can reach — and for each response it checks the
 * thing a status line hides, the content type. A host with a catch-all rewrite
 * answers `shared/rules.js` with `index.html` at HTTP 200, the module loader
 * refuses it, and the page dies while the Network tab looks spotless; that is the
 * single most common way to deploy this repo wrong, and it is invisible to a
 * status-code check.
 *
 * No dependencies, no build: same rules as the rest of the project.
 */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const target = process.argv[2];
const MIME_OK = /(javascript|ecmascript|css|html)/;

async function serveRepo(port) {
  const server = createServer(async (req, res) => {
    const rel = decodeURIComponent((req.url || '/').split('?')[0]);
    const file = path.join(ROOT, rel === '/' ? 'index.html' : rel);
    if (!file.startsWith(ROOT)) { res.writeHead(403).end(); return; }
    try {
      const buf = await readFile(file);
      const ext = path.extname(file);
      res.writeHead(200, {
        'content-type': ext === '.js' ? 'text/javascript' : ext === '.css' ? 'text/css' : 'text/html; charset=utf-8',
      }).end(buf);
    } catch { res.writeHead(404, { 'content-type': 'text/plain' }).end('404'); }
  });
  await new Promise((r) => server.listen(port, '127.0.0.1', r));
  return server;
}

export async function checkStatic(base) {
  const seen = new Set();
  const problems = [];
  const rows = [];
  const fetchOne = async (url) => {
    try {
      const r = await fetch(url, { headers: { accept: '*/*' } });
      const body = await r.text();
      return { status: r.status, type: r.headers.get('content-type') || '(none)', body, ok: r.ok };
    } catch (err) {
      return { status: 0, type: '(unreachable)', body: '', ok: false, error: String(err.message || err) };
    }
  };

  const entry = new URL('index.html', base.endsWith('/') ? base : base + '/').href;
  const doc = await fetchOne(entry);
  if (!doc.ok || !/html/i.test(doc.type)) {
    problems.push(`the document itself: HTTP ${doc.status} ${doc.type} at ${entry}`);
    return { problems, rows, count: 0 };
  }
  rows.push({ url: entry.replace(/^https?:/, ''), status: doc.status, type: doc.type, ok: true });

  const mapMatch = /<script type="importmap">([\s\S]*?)<\/script>/.exec(doc.body);
  let imports = {};
  if (mapMatch) {
    try { imports = JSON.parse(mapMatch[1]).imports || {}; } catch (e) { problems.push(`import map is not valid JSON: ${e.message}`); }
  } else problems.push('no <script type="importmap"> in the document — `three` will not resolve');

  const refs = [...doc.body.matchAll(/(?:href|src)="([^"#]+)"/g)].map((m) => m[1])
    .filter((u) => !u.startsWith('data:') && !/^https?:/.test(u))
    .concat(Object.values(imports));

  const queue = refs.map((r) => ({ from: entry, spec: r }));
  while (queue.length) {
    const { from, spec } = queue.shift();
    const url = new URL(spec, from).href;
    const key = url;
    if (seen.has(key)) continue;
    seen.add(key);
    const res = await fetchOne(url);
    const isJs = /\.(m?js)($|\?)/.test(url);
    const isCss = /\.css($|\?)/.test(url);
    const looksHtml = /^\s*<(!DOCTYPE|html)/i.test(res.body);
    const good = res.ok && !looksHtml && (!isJs || /javascript|ecmascript/.test(res.type)) && (!isCss || /css/.test(res.type));
    rows.push({ url: url.replace(/^https?:/, ''), status: res.status, type: res.type, ok: good });
    if (!good) {
      problems.push(`${url.replace(/^https?:\/\/[^/]+/, '') || '/'} → HTTP ${res.status}, ${res.type}` +
        (looksHtml ? ' — that is HTML, not the file the page asked for (a catch-all rewrite). Remove the `/* /index.html` rule or fix the publish directory.' : '') +
        (res.error ? ` (${res.error})` : ''));
      continue;
    }
    if (isJs) {
      for (const m of res.body.matchAll(/(?:from\s+|import\s*\(\s*)['"]([^'"]+)['"]/g)) {
        const sp = m[1];
        if (sp.startsWith('.') || sp.startsWith('/')) queue.push({ from: url, spec: sp });
        else if (!(sp in imports)) problems.push(`module imports "${sp}" but the import map has no entry for it`);
      }
    }
  }
  return { problems, rows, count: seen.size + 1, MIME_OK };
}

let server = null;
let base = target;
if (!base) {
  const port = Number(process.env.PORT) || 4173;
  server = await serveRepo(port);
  base = `http://127.0.0.1:${port}`;
  console.log(`(serving ${ROOT} at ${base} — pass a URL to check a real deployment)\n`);
}

const { problems, rows, count } = await checkStatic(base);
if (server) server.close();

const width = Math.max(...rows.map((r) => r.url.length), 12);
for (const r of rows.sort((a, b) => Number(a.ok) - Number(b.ok) || a.url.localeCompare(b.url))) {
  console.log(`${r.ok ? '✓' : '✗'} ${r.url.padEnd(width)}  ${String(r.status).padStart(3)}  ${r.type.split(';')[0]}`);
}
console.log(`\n${count} files reachable from index.html; ${problems.length} problem(s).`);
for (const p of problems) console.log('  ✗ ' + p);
if (!problems.length) {
  console.log('  every module the page needs is served as JavaScript ✓');
  console.log('  (this checks paths and content types, not WebGL — if it passes but the');
  console.log('   menu still says "loading the client modules…", read the on-page error)');
}
process.exit(problems.length ? 1 : 0);
