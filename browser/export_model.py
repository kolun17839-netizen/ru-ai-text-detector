"""Export the upstream v2 classifier and its calibration for in-browser inference."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from onnx import TensorProto, helper, numpy_helper
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_REPO = "k1y0mi/ru-ai-text-detector"
MODEL_REVISION = "175e75ec79e9471f3f5b4ab0da9b859188446807"
CHECKSUMS = {
    "config.json": "49eb1deb4b1fdb081a58571b22c5eafb8beb71aaa1132508d51d78593e65b319",
    "tokenizer.json": "93738ba3891ee2502e03e89efc2277b4bab33c7b44655a417e6632aba58bab3b",
    "tokenizer_config.json": "18ca8313f412bc34fe4015aef1d5817ec2f810e9b42f441d6840db767fbc9b36",
    "special_tokens_map.json": "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
    "model.safetensors": "c960c584fb4e2086d81cff7516f8620ba559f5f05ea6572b6b68243e06f04a1a",
}
INPUT_NAMES = ["input_ids", "attention_mask", "token_type_ids"]


def prepare_weights(weights: Path, download: bool):
    weights.mkdir(parents=True, exist_ok=True)
    for filename, expected in CHECKSUMS.items():
        target = weights / filename
        if download and not target.exists():
            source = hf_hub_download(
                MODEL_REPO, filename, subfolder="v2_model", revision=MODEL_REVISION,
            )
            shutil.copyfile(source, target)
        with target.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {target}")


def compare(model, tokenizer, path):
    """Catch export errors on the original repository's examples."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    max_difference = 0.0
    with torch.inference_mode():
        for sample in sorted((ROOT / "samples").glob("sample_*.txt")):
            text = sample.read_text(encoding="utf-8")
            inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
            expected = model(**inputs).logits.softmax(dim=-1).numpy()[0, 1]
            (logits,) = session.run(None, {name: inputs[name].numpy() for name in INPUT_NAMES})
            actual = 1 / (1 + np.exp(logits[0, 0] - logits[0, 1]))
            max_difference = max(max_difference, abs(float(expected - actual)))
    if max_difference > 0.025:
        raise ValueError(f"ONNX model raw score drift: {max_difference:.4f}")
    print(f"Maximum raw score change across six upstream examples: {max_difference:.4f}")


def compress_embeddings(source, target):
    """Store each vocabulary row with its own int8 scale, leaving the encoder in FP32."""
    model = onnx.load(source)
    graph = model.graph
    name = "bert.embeddings.word_embeddings.weight"
    original = next(value for value in graph.initializer if value.name == name)
    weights = numpy_helper.to_array(original)
    scales = np.maximum(np.max(np.abs(weights), axis=1, keepdims=True) / 127, 1e-12)
    quantized = np.rint(weights / scales).astype(np.int8)
    graph.initializer.remove(original)
    graph.initializer.extend([
        numpy_helper.from_array(quantized, name),
        numpy_helper.from_array(scales, "word_embedding_scales"),
    ])
    index, gather = next(
        (index, node) for index, node in enumerate(graph.node)
        if node.op_type == "Gather" and node.input[0] == name
    )
    ids = gather.input[1]
    output = gather.output[0]
    del graph.node[index]
    nodes = [
        helper.make_node("Gather", [name, ids], ["embedding_integer"]),
        helper.make_node("Gather", ["word_embedding_scales", ids], ["embedding_scale"]),
        helper.make_node("Cast", ["embedding_integer"], ["embedding_float"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["embedding_float", "embedding_scale"], [output]),
    ]
    for offset, node in enumerate(nodes):
        graph.node.insert(index + offset, node)
    onnx.checker.check_model(model)
    onnx.save(model, target)


def export(weights: Path, download: bool):
    torch.set_num_threads(2)
    prepare_weights(weights, download)
    output = ROOT / "browser" / "assets"
    output.mkdir(exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        weights, use_safetensors=True, local_files_only=True, attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    sample = tokenizer("Пример текста для экспорта модели.", return_tensors="pt")
    axes = {name: {1: "sequence"} for name in INPUT_NAMES}
    with tempfile.TemporaryDirectory() as work:
        raw = Path(work) / "model-float.onnx"
        torch.onnx.export(
            model, tuple(sample[name] for name in INPUT_NAMES), str(raw),
            input_names=INPUT_NAMES, output_names=["logits"], dynamic_axes=axes,
            opset_version=17,
        )
        quantized = Path(work) / "model.onnx"
        compress_embeddings(raw, quantized)
        compare(model, tokenizer, quantized)
        shutil.copyfile(quantized, output / "model.onnx")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(weights / name, output / name)
    licenses = output / "licenses"
    licenses.mkdir(exist_ok=True)
    for source, target in [
        ("LICENSE", "onnxruntime.txt"),
        ("ThirdPartyNotices.txt", "onnxruntime-third-party.txt"),
    ]:
        shutil.copyfile(Path(ort.__file__).parent / source, licenses / target)
    # Convert the original isotonic calibration to data, without Python on the site.
    calibrator = joblib.load(ROOT / "models" / "v2_calibrator.joblib")
    threshold = json.loads((ROOT / "models" / "threshold_v2.json").read_text())["threshold"]
    calibration = {
        "x": calibrator.X_thresholds_.tolist(),
        "y": calibrator.y_thresholds_.tolist(),
        "threshold": threshold,
    }
    (output / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="Use previously downloaded v2 weights")
    args = parser.parse_args()
    if args.weights:
        export(args.weights, download=False)
    else:
        with tempfile.TemporaryDirectory() as work:
            export(Path(work), download=True)
