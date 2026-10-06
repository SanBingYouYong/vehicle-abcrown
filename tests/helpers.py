"""Small networks and queries for the tests."""

import os
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
MNIST_DIR = os.path.join(PROJECT, "examples", "mnist")
MNIST_ONNX = os.path.join(MNIST_DIR, "mnist-classifier.onnx")


def mlp_weights(sizes, seed=0, scale=0.5):
    rng = np.random.default_rng(seed)
    return [(rng.normal(size=(m, n)) * scale, rng.normal(size=m) * 0.1)
            for n, m in zip(sizes, sizes[1:])]


def make_mlp(path, weights):
    """A ReLU MLP with input [1, n] and output [1, m]."""
    nodes, inits = [], []
    cur = "input"
    for i, (w, b) in enumerate(weights):
        inits.append(numpy_helper.from_array(w.T.astype(np.float32), f"w{i}"))
        inits.append(numpy_helper.from_array(b.astype(np.float32), f"b{i}"))
        nodes.append(helper.make_node("MatMul", [cur, f"w{i}"], [f"mm{i}"]))
        out = "output" if i == len(weights) - 1 else f"add{i}"
        nodes.append(helper.make_node("Add", [f"mm{i}", f"b{i}"], [out]))
        if i < len(weights) - 1:
            nodes.append(helper.make_node("Relu", [out], [f"relu{i}"]))
            cur = f"relu{i}"
    graph = helper.make_graph(
        nodes, "mlp",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, weights[0][0].shape[1]])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, weights[-1][0].shape[0]])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def mlp_forward(weights, x):
    x = np.asarray(x, dtype=np.float64)
    for i, (w, b) in enumerate(weights):
        x = w @ x + b
        if i < len(weights) - 1:
            x = np.maximum(x, 0)
    return x


def linf_lipschitz_bound(weights):
    """A sound bound on max_i |f(x)_i - f(y)_i| / ||x - y||_inf."""
    m = np.abs(weights[0][0])
    for w, _ in weights[1:]:
        m = np.abs(w) @ m
    return float(m.sum(axis=1).max())


def global_query(n_in, n_out, eps, delta, output_index, upper):
    """The VNN-LIB 2.0 query Vehicle generates for one disjunct of
    `forall x y . valid x, valid y, |x - y| <= eps => |f x ! i - f y ! i| <= delta`."""
    lines = [
        "(vnnlib-version <2.0>)",
        "(declare-network f",
        f"  (declare-input f_X Real [{n_in}])",
        f"  (declare-output f_Y Real [{n_out}])",
        ")",
        "(declare-network f_2",
        "  (equal-to f)",
        f"  (declare-input f_2_X Real [{n_in}])",
        f"  (declare-output f_2_Y Real [{n_out}])",
        ")",
    ]
    for i in range(n_in):
        lines.append(f"(assert (<= (+ f_2_X[{i}] (- f_X[{i}])) {eps}))")
        lines.append(f"(assert (<= (+ (- f_2_X[{i}]) f_X[{i}]) {eps}))")
    if upper:
        lines.append(f"(assert (> (+ f_2_Y[{output_index}] (- f_Y[{output_index}])) {delta}))")
    else:
        lines.append(f"(assert (< (+ f_2_Y[{output_index}] (- f_Y[{output_index}])) -{delta}))")
    for i in range(n_in):
        lines.append(f"(assert (and (<= 0.0 f_X[{i}]) (<= f_X[{i}] 1.0)))")
        lines.append(f"(assert (and (<= 0.0 f_2_X[{i}]) (<= f_2_X[{i}] 1.0)))")
    return "\n".join(lines) + "\n"


def have_abcrown():
    root = os.environ.get("ABCROWN_ROOT", os.path.expanduser("~/alpha-beta-CROWN"))
    return os.path.exists(os.path.join(root, "complete_verifier", "abcrown.py"))


def vehicle_executable():
    return os.environ.get("VEHICLE") or shutil.which("vehicle")
