#!/usr/bin/env node

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import playwright from '../tests/ui/node_modules/playwright/index.js';
const { chromium } = playwright;

const repo = process.cwd();
const source = fs.readFileSync(path.join(repo, 'assets/brand/token-ingot.svg'), 'utf8');
const staticBrand = path.join(repo, 'static/brand');
const exportsRoot = path.join(repo, 'assets/brand/exports');
fs.mkdirSync(staticBrand, { recursive: true });
fs.mkdirSync(exportsRoot, { recursive: true });

function write(file, data) {
    const target = path.join(repo, file);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, data);
}

async function render(page, file, width, height, padding = 0) {
    const encoded = Buffer.from(source).toString('base64');
    const scale = padding ? 1 - padding * 2 : 1;
    await page.setViewportSize({ width, height });
    // Flex-centers the (square) mark within an arbitrary WxH canvas. The
    // previous width%/height% + margin-top approach only worked by
    // coincidence for square outputs (16x16, etc) — on a non-square canvas
    // like the 1200x630 social export, percentage margin resolves against
    // the containing block's *width*, not its height, which pushed the mark
    // far outside the visible frame instead of centering it.
    await page.setContent(`<!doctype html><style>html,body{margin:0;width:100%;height:100%;background:transparent;overflow:hidden;display:flex;align-items:center;justify-content:center}img{width:${scale * 100}%;height:${scale * 100}%;object-fit:contain}</style><img src="data:image/svg+xml;base64,${encoded}" />`);
    await page.screenshot({ path: path.join(repo, file), omitBackground: true });
}

function copy(sourceFile, targets) {
    for (const target of targets) {
        fs.mkdirSync(path.dirname(path.join(repo, target)), { recursive: true });
        fs.copyFileSync(path.join(repo, sourceFile), path.join(repo, target));
    }
}

function digest(file) {
    return crypto.createHash('sha256').update(fs.readFileSync(path.join(repo, file))).digest('hex');
}

function allFiles(directory, result = []) {
    for (const entry of fs.readdirSync(path.join(repo, directory), { withFileTypes: true })) {
        const relative = path.join(directory, entry.name).split(path.sep).join('/');
        if (entry.isDirectory()) allFiles(relative, result);
        else result.push(relative);
    }
    return result;
}

function createProofSheet() {
    const output = 'docs/plans/evidence/20260811-local-llm-foundry/phase-01/small-size-contact-sheet.png';
    const files = [16, 20, 22, 24, 32, 64, 128, 256, 512].map((size) => `static/brand/token-ingot-${size}.png`);
    const script = [
        'from PIL import Image, ImageDraw',
        'import sys',
        'files = sys.argv[2:]',
        'cell = 180',
        'sheet = Image.new("RGBA", (cell * 3, cell * 3), (247, 248, 250, 255))',
        'draw = ImageDraw.Draw(sheet)',
        'for index, filename in enumerate(files):',
        '    image = Image.open(filename).convert("RGBA")',
        '    image.thumbnail((128, 128), Image.Resampling.LANCZOS)',
        '    x = (index % 3) * cell + (cell - image.width) // 2',
        '    y = (index // 3) * cell + 12',
        '    sheet.alpha_composite(image, (x, y))',
        '    draw.text(((index % 3) * cell + 12, (index // 3) * cell + 150), filename.rsplit("-", 1)[-1].replace(".png", " px"), fill=(31, 35, 42, 255))',
        'sheet.save(sys.argv[1], format="PNG")',
    ].join('\n');
    const result = spawnSync('python3', ['-c', script, path.join(repo, output), ...files.map((file) => path.join(repo, file))], { encoding: 'utf8' });
    if (result.status !== 0) throw new Error(result.stderr || 'contact sheet generation failed');
}

function createRasterReports() {
    const evidenceDir = path.join(repo, 'docs/plans/evidence/20260811-local-llm-foundry/phase-01');
    const files = [16, 20, 22, 24, 32, 64, 128, 180, 192, 256, 512, 1024].map((size) => `static/brand/token-ingot-${size}.png`)
        .concat(['static/brand/token-ingot-maskable-192.png', 'static/brand/token-ingot-maskable-512.png']);
    const script = [
        'from PIL import Image, ImageDraw',
        'import json, sys',
        'files = sys.argv[2:]',
        'report = []',
        'for filename in files:',
        '    image = Image.open(filename).convert("RGBA")',
        '    alpha = image.getchannel("A")',
        '    report.append({"file": filename, "width": image.width, "height": image.height, "has_transparency": alpha.getextrema()[0] < 255, "transparent_pixels": sum(1 for value in alpha.getdata() if value == 0)})',
        'with open(sys.argv[1], "w") as output: json.dump(report, output, indent=2); output.write("\\n")',
        'cell = 180',
        'sheet = Image.new("RGBA", (cell * 3, cell * 2), (31, 35, 42, 255))',
        'draw = ImageDraw.Draw(sheet)',
        'for index, platform in enumerate(["macos", "windows", "linux"]):',
        '    for row, background in enumerate([(31,35,42,255), (247,248,250,255)]):',
        '        x = index * cell',
        '        y = row * cell',
        '        draw.rectangle((x, y, x + cell, y + cell), fill=background)',
        '        filename = f"assets/brand/exports/tray/{platform}/token-ingot-64.png"',
        '        image = Image.open(filename).convert("RGBA")',
        '        image.thumbnail((112, 112), Image.Resampling.LANCZOS)',
        '        sheet.alpha_composite(image, (x + (cell-image.width)//2, y + 22))',
        '        label = "dark" if row == 0 else "light"',
        '        draw.text((x + 12, y + 148), f"{platform} / {label}", fill=(255,255,255,255) if row == 0 else (31,35,42,255))',
        'sheet.save(sys.argv[1].replace("raster-alpha-report.json", "native-platform-contact-sheet.png"), format="PNG")',
    ].join('\n');
    const report = path.join(evidenceDir, 'raster-alpha-report.json');
    const result = spawnSync('python3', ['-c', script, report, ...files.map((file) => path.join(repo, file))], { cwd: repo, encoding: 'utf8' });
    if (result.status !== 0) throw new Error(result.stderr || 'raster report generation failed');
}

function writePhaseReceipt() {
    const staticFiles = allFiles('static/brand').sort();
    const exportFiles = allFiles('assets/brand/exports').sort();
    const svgFiles = ['assets/brand/token-ingot.svg', 'assets/brand/token-ingot-dark.svg', 'assets/brand/token-ingot-light.svg', 'assets/brand/token-ingot-mono.svg', 'assets/brand/token-ingot-one-color-black.svg', 'assets/brand/token-ingot-one-color-white.svg', 'assets/brand/token-ingot-macos-template.svg', 'static/icon.svg'];
    const manifest = {
        schema_version: 1,
        production_master: 'assets/brand/token-ingot.svg',
        svg_safety: svgFiles.map((file) => {
            const content = fs.readFileSync(path.join(repo, file), 'utf8');
            const lower = content.toLowerCase();
            return { file, sha256: digest(file), xml_like: content.trimStart().startsWith('<svg'), forbidden_tokens: ['<script', '<image', '<filter', 'foreignobject', 'href="http'].filter((token) => lower.includes(token)) };
        }),
        web_pwa: staticFiles.map((file) => ({ file, sha256: digest(`static/brand/${path.relative(path.join(repo, 'static/brand'), path.join(repo, file))}`) })),
        exports: exportFiles.map((file) => ({ file, sha256: digest(file), bytes: fs.statSync(path.join(repo, file)).size })),
        required_sizes: { tray_and_package_px: [16, 20, 22, 24, 32, 64, 128, 256, 512, 1024], apple_touch_px: 180, regular_pwa_px: [192, 512], maskable_pwa_px: [192, 512], social_px: [1200, 630], windows: 'token-ingot.ico', macos: 'token-ingot.icns', linux: 'hicolor/*/apps/local-llm-foundry.png' },
    };
    const evidenceDir = path.join(repo, 'docs/plans/evidence/20260811-local-llm-foundry/phase-01');
    fs.mkdirSync(evidenceDir, { recursive: true });
    fs.writeFileSync(path.join(evidenceDir, 'export-matrix.json'), `${JSON.stringify(manifest, null, 2)}\n`);
    fs.writeFileSync(path.join(evidenceDir, 'svg-safety-report.json'), `${JSON.stringify({ production_master: manifest.production_master, results: manifest.svg_safety, pass: manifest.svg_safety.every((item) => item.xml_like && item.forbidden_tokens.length === 0) }, null, 2)}\n`);
}

function createIco() {
    const script = [
        'from PIL import Image',
        'import sys',
        'image = Image.open(sys.argv[1]).convert("RGBA")',
        'image.save(sys.argv[2], format="ICO", sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])',
    ].join('; ');
    const result = spawnSync('python3', ['-c', script, path.join(repo, 'static/brand/token-ingot-512.png'), path.join(repo, 'static/brand/token-ingot.ico')], { encoding: 'utf8' });
    if (result.status !== 0) throw new Error(result.stderr || 'Pillow ICO export failed');
    fs.copyFileSync(path.join(staticBrand, 'token-ingot.ico'), path.join(exportsRoot, 'token-ingot.ico'));
}

function createIcns() {
    const iconset = `${fs.mkdtempSync(path.join(os.tmpdir(), 'token-ingot-'))}.iconset`;
    fs.mkdirSync(iconset);
    const names = [[16, 'icon_16x16.png'], [32, 'icon_16x16@2x.png'], [32, 'icon_32x32.png'], [64, 'icon_32x32@2x.png'], [128, 'icon_128x128.png'], [256, 'icon_128x128@2x.png'], [256, 'icon_256x256.png'], [512, 'icon_256x256@2x.png'], [512, 'icon_512x512.png'], [1024, 'icon_512x512@2x.png']];
    for (const [size, name] of names) fs.copyFileSync(path.join(staticBrand, `token-ingot-${size}.png`), path.join(iconset, name));
    const target = path.join(exportsRoot, 'token-ingot.icns');
    const result = spawnSync('iconutil', ['-c', 'icns', iconset, '-o', target], { encoding: 'utf8' });
    fs.rmSync(iconset, { recursive: true, force: true });
    if (result.status !== 0) throw new Error(result.stderr || 'iconutil export failed');
}

async function main() {
    for (const [file, fill] of [['assets/brand/token-ingot-one-color-black.svg', '#000'], ['assets/brand/token-ingot-one-color-white.svg', '#fff'], ['assets/brand/token-ingot-macos-template.svg', '#000']]) {
        write(file, source.replaceAll(/fill="#[0-9a-f]+"/gi, `fill="${fill}"`));
    }
    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({ deviceScaleFactor: 1 });
    for (const size of [16, 20, 22, 24, 32, 64, 128, 180, 192, 256, 512, 1024]) {
        await render(page, `static/brand/token-ingot-${size}.png`, size, size);
        copy(`static/brand/token-ingot-${size}.png`, [`assets/brand/exports/tray/macos/token-ingot-${size}.png`, `assets/brand/exports/tray/windows/token-ingot-${size}.png`, `assets/brand/exports/tray/linux/token-ingot-${size}.png`]);
    }
    for (const size of [192, 512]) await render(page, `static/brand/token-ingot-maskable-${size}.png`, size, size, 0.18);
    await render(page, 'assets/brand/exports/social/token-ingot-1200x630.png', 1200, 630, 0.25);
    await browser.close();
    createIco();
    createIcns();
    for (const size of [16, 32, 64, 128, 256, 512]) copy(`static/brand/token-ingot-${size}.png`, [`assets/brand/exports/linux/hicolor/${size}x${size}/apps/local-llm-foundry.png`]);
    createProofSheet();
    createRasterReports();
    writePhaseReceipt();
    console.log('Generated Token Ingot raster, ICO, ICNS, tray, package, maskable, and social derivatives.');
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
