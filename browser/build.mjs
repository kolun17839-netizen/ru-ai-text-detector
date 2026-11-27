import { copyFile, cp, mkdir, rm } from "node:fs/promises";
import { build } from "esbuild";

const dist = new URL("./dist/", import.meta.url);
const local = (path) => new URL(path, import.meta.url);
const copy = (source, target) => copyFile(local(source), new URL(target, dist));

await rm(dist, { recursive: true, force: true });
await mkdir(new URL("licenses/", dist), { recursive: true });
await cp(local("./public/"), dist, { recursive: true });
await cp(local("./assets/"), dist, { recursive: true });
for (const module of ["app.js", "text.js"]) await copy("./src/" + module, module);

await build({
  entryPoints: [local("./src/worker.js").pathname],
  outfile: new URL("worker.js", dist).pathname,
  bundle: true,
  minify: true,
  format: "esm",
  target: "es2022",
  conditions: ["onnxruntime-web-use-extern-wasm"],
  legalComments: "linked",
});

for (const extension of ["mjs", "wasm"]) {
  const filename = "ort-wasm-simd-threaded." + extension;
  await copy("./node_modules/onnxruntime-web/dist/" + filename, filename);
}
for (const [font, subset, style] of [
  ["lora", "cyrillic", "normal"],
  ["lora", "cyrillic", "italic"],
  ["manrope", "cyrillic", "normal"],
  ["manrope", "latin", "normal"],
]) {
  const source = `./node_modules/@fontsource-variable/${font}/files/${font}-${subset}-wght-${style}.woff2`;
  const target = `${font}-${subset}${style === "italic" ? "-italic" : ""}.woff2`;
  await copy(source, target);
}
for (const [source, target] of [
  ["../LICENSE", "original-project.txt"],
  ["./node_modules/@fontsource-variable/lora/LICENSE", "lora.txt"],
  ["./node_modules/@fontsource-variable/manrope/LICENSE", "manrope.txt"],
  ["./node_modules/@huggingface/tokenizers/LICENSE", "tokenizers.txt"],
]) {
  await copy(source, "licenses/" + target);
}
