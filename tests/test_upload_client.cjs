"use strict";

// Run with: node --test tests/test_upload_client.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const {File} = require("node:buffer");
const source = fs.readFileSync(path.join(__dirname, "../static/app.js"), "utf8");

function fixture({requestBytes = 4000000, encode} = {}) {
  const elements = new Map();
  const uploads = [];
  const canvases = [];
  const drawing = [];
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      hidden: true, disabled: true, value: "", textContent: "", style: {},
      naturalWidth: 4000, naturalHeight: 2000,
      classList: {add() {}, remove() {}, toggle() {}},
      handlers: {},
      addEventListener(name, callback) { this.handlers[name] = callback; },
      setAttribute(name, value) { this[name] = value; },
      removeAttribute(name) { delete this[name]; },
      replaceChildren() {}, appendChild() {}, append() {}, focus() {}, click() {},
    });
    return elements.get(id);
  }
  const document = {
    body: {dataset: {uploadMaxBytes: String(25 * 1024 * 1024), requestUploadBytes: String(requestBytes)}},
    getElementById: element,
    createElement(tag) {
      if (tag !== "canvas") return element(Symbol(tag));
      const context = {
        fillRect(...args) { drawing.push({action: "fill", color: this.fillStyle, args}); },
        drawImage(...args) { drawing.push({action: "draw", args}); },
      };
      const canvas = {
        getContext: () => context,
        toBlob(callback, type, quality) {
          if (encode) encode(callback, type, quality);
          else callback(new Blob([new Uint8Array(300000)], {type}));
        },
      };
      canvases.push(canvas);
      return canvas;
    },
  };
  class FormData {
    append(name, file) { this[name] = file; }
  }
  vm.runInNewContext(source, {
    document, window: {addEventListener() {}}, File, FormData, AbortController,
    DOMException, Blob, Uint8Array, DataView, setTimeout: () => 1, clearTimeout() {},
    URL: {createObjectURL: () => "blob:original-preview", revokeObjectURL() {}},
    async fetch(url, options) {
      if (url === "/api/health") return {ok: true, json: async () => ({status: "ok"})};
      uploads.push(options.body.image);
      return {ok: true, json: async () => ({
        predictions: [{label: "cat", confidence: 95}],
        image: {width: 1600, height: 800}, elapsed_ms: 10,
      })};
    },
  });
  return {
    element, uploads, canvases, drawing,
    select(file) {
      element("image-input").files = [file];
      element("image-input").handlers.change();
      if (element("image-preview").onload) element("image-preview").onload();
    },
    analyze: () => element("analyze-button").handlers.click(),
    reset: () => element("reset-button").handlers.click(),
  };
}

function jpeg(size = 5000000) {
  return new File([new Uint8Array([255, 216, 255]), new Uint8Array(size - 3)], "original.jpg", {type: "image/jpeg"});
}

test("small uploads keep the original bytes and preview", async () => {
  const app = fixture();
  const file = jpeg(1000);
  app.select(file);
  await app.analyze();
  assert.equal(app.uploads[0], file);
  assert.equal(app.canvases.length, 0);
  assert.equal(app.element("image-preview").src, "blob:original-preview");
  assert.equal(app.element("upload-note").hidden, true);
});

test("oversized uploads become bounded JPEGs with white transparency background", async () => {
  const app = fixture();
  const file = jpeg();
  app.select(file);
  await app.analyze();
  assert.equal(app.uploads[0].type, "image/jpeg");
  assert.ok(app.uploads[0].size <= 4000000);
  assert.equal(app.drawing[0].action, "fill");
  assert.equal(app.drawing[0].color, "#ffffff");
  assert.deepEqual(app.drawing[0].args, [0, 0, 1600, 800]);
  assert.equal(app.drawing[1].action, "draw");
  assert.equal(file.size, 5000000);
  assert.equal(app.element("image-preview").src, "blob:original-preview");
  assert.match(app.element("file-size").textContent, /4000 × 2000/);
  assert.match(app.element("upload-note").textContent, /1600 × 800 px JPEG/);
  assert.equal(app.canvases[0].width, 0);
});

test("local deployments preserve the original 25 MiB upload allowance", async () => {
  const app = fixture({requestBytes: 25 * 1024 * 1024});
  const file = jpeg();
  app.select(file);
  await app.analyze();
  assert.equal(app.uploads[0], file);
  assert.equal(app.canvases.length, 0);
});

test("files exceeding the original choice limit are rejected before upload", async () => {
  const app = fixture();
  app.select(jpeg(26 * 1024 * 1024));
  await app.analyze();
  assert.equal(app.uploads.length, 0);
  assert.match(app.element("error-message").textContent, /25 MB/);
});

test("changing files while encoding prevents stale requests and notes", async () => {
  let finish;
  const app = fixture({encode: (callback) => { finish = callback; }});
  app.select(jpeg());
  const pending = app.analyze();
  await new Promise(setImmediate);
  assert.equal(typeof finish, "function");
  const replacement = jpeg(1000);
  app.select(replacement);
  finish(new Blob([new Uint8Array(100)], {type: "image/jpeg"}));
  await pending;
  assert.equal(app.uploads.length, 0);
  assert.equal(app.element("upload-note").hidden, true);
  assert.equal(app.element("analyze-button").disabled, false);
  await app.analyze();
  assert.equal(app.uploads[0], replacement);
  assert.equal(app.canvases[0].width, 0);
});

test("encoding retries lower quality and never sends an oversized result", async () => {
  const qualities = [];
  const app = fixture({encode(callback, type, quality) {
    qualities.push(quality);
    callback(new Blob([new Uint8Array(quality === 0.7 ? 300000 : 5000000)], {type}));
  }});
  app.select(jpeg());
  await app.analyze();
  assert.deepEqual(qualities, [0.9, 0.8, 0.7]);
  assert.equal(app.uploads[0].size, 300000);
});

test("encoding failure is reported without uploading", async () => {
  const app = fixture({encode: (callback) => callback(null)});
  app.select(jpeg());
  await app.analyze();
  assert.equal(app.uploads.length, 0);
  assert.match(app.element("error-message").textContent, /choose a smaller image/);
  assert.equal(app.element("analyze-button").disabled, false);
});

for (const type of ["png", "webp"]) {
  test(`oversized animated ${type} is not silently converted into a still image`, async () => {
    const bytes = Buffer.alloc(5000000);
    if (type === "png") {
      Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(bytes);
      // Place acTL beyond a large ancillary chunk to exercise bounded header reads.
      bytes.writeUInt32BE(100000, 8);
      bytes.write("tEXt", 12);
      bytes.writeUInt32BE(8, 100020);
      bytes.write("acTL", 100024);
    } else {
      bytes.write("RIFF");
      bytes.writeUInt32LE(bytes.length - 8, 4);
      bytes.write("WEBPVP8X", 8);
      bytes.writeUInt32LE(10, 16);
      bytes[20] = 2;
    }
    const app = fixture();
    app.select(new File([bytes], `animation.${type}`, {type: `image/${type}`}));
    await app.analyze();
    assert.equal(app.uploads.length, 0);
    assert.equal(app.canvases.length, 0);
    assert.match(app.element("error-message").textContent, /Animated images are not supported/);
  });
}
