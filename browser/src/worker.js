import * as ort from "onnxruntime-web/wasm";
import { Tokenizer } from "@huggingface/tokenizers";
import { countWords, partition, subdivide } from "./text.js";

const BASE = new URL(".", self.location.href);
const TOKEN_LIMIT = 256;

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = BASE.href;

async function readJson(name) {
  const response = await fetch(new URL(name, BASE));
  if (!response.ok) throw new Error("Failed to load " + name);
  return response.json();
}

async function loadModel() {
  const [definition, settings, calibration, session] = await Promise.all([
    readJson("tokenizer.json"),
    readJson("tokenizer_config.json"),
    readJson("calibration.json"),
    ort.InferenceSession.create(new URL("model.onnx", BASE).href, {
      executionProviders: ["wasm"],
    }),
  ]);
  // Split in the worker before inference, so the tokenizer must not trim the input.
  delete definition.truncation;
  delete definition.padding;
  return { tokenizer: new Tokenizer(definition, settings), calibration, session };
}

function encodeWindows(tokenizer, text) {
  const inputs = [];
  function append(part) {
    const { ids } = tokenizer.encode(part, { add_special_tokens: true });
    if (ids.length <= TOKEN_LIMIT) {
      inputs.push({ text: part, ids });
      return;
    }
    for (const piece of subdivide(part)) append(piece);
  }
  for (const part of partition(text)) append(part);
  return inputs;
}

function interpolate(value, { x, y }) {
  if (value <= x[0]) return y[0];
  for (let i = 1; i < x.length; i += 1) {
    if (value <= x[i]) {
      const fraction = (value - x[i - 1]) / (x[i] - x[i - 1]);
      return y[i - 1] + fraction * (y[i] - y[i - 1]);
    }
  }
  return y.at(-1);
}

let model;
self.onmessage = async ({ data: text }) => {
  try {
    if (!model) {
      self.postMessage({ kind: "loading" });
      model = await loadModel();
    }
    const { tokenizer, calibration, session } = model;
    const windows = encodeWindows(tokenizer, text);
    const results = [];
    for (const { text: part, ids } of windows) {
      self.postMessage({ kind: "progress", current: results.length + 1, total: windows.length });
      const shape = [1, ids.length];
      const { logits } = await session.run({
        input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), shape),
        attention_mask: new ort.Tensor("int64", new BigInt64Array(ids.length).fill(1n), shape),
        token_type_ids: new ort.Tensor("int64", new BigInt64Array(ids.length), shape),
      });
      const raw = 1 / (1 + Math.exp(logits.data[0] - logits.data[1]));
      const score = interpolate(raw, calibration);
      const short = countWords(part) < 40;
      results.push({ text: part, score, short, high: !short && score >= calibration.threshold });
    }
    self.postMessage({ kind: "done", results });
  } catch (error) {
    console.error(error);
    self.postMessage({ kind: "error" });
  }
};
