"""Writes toy.onnx: a small ReLU network 2 -> 16 -> 16 -> 2 with fixed random
weights, and prints a sound L-infinity Lipschitz bound for it."""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

rng = np.random.default_rng(2026)
sizes = [2, 16, 16, 2]
layers = [(rng.normal(size=(m, n)) / np.sqrt(n), rng.normal(size=m) * 0.1)
          for n, m in zip(sizes, sizes[1:])]

nodes, inits, cur = [], [], "input"
for i, (w, b) in enumerate(layers):
    inits += [numpy_helper.from_array(w.T.astype(np.float32), f"w{i}"),
              numpy_helper.from_array(b.astype(np.float32), f"b{i}")]
    nodes.append(helper.make_node("MatMul", [cur, f"w{i}"], [f"mm{i}"]))
    out = "output" if i == len(layers) - 1 else f"add{i}"
    nodes.append(helper.make_node("Add", [f"mm{i}", f"b{i}"], [out]))
    if i < len(layers) - 1:
        nodes.append(helper.make_node("Relu", [out], [f"relu{i}"]))
        cur = f"relu{i}"
graph = helper.make_graph(
    nodes, "toy", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, os.path.join(os.path.dirname(os.path.abspath(__file__)), "toy.onnx"))

m = np.abs(layers[0][0])
for w, _ in layers[1:]:
    m = np.abs(w) @ m
print(f"L-infinity Lipschitz bound: {m.sum(axis=1).max():.4f}")
