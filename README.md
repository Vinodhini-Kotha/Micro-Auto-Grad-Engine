# Vectorized Autograd & Deep Learning Engine from Scratch

An open-source, lightweight automatic differentiation library and modular Multi-Layer Perceptron (MLP) framework engineered entirely from the ground up using Python and NumPy.

While popular educational frameworks (like micrograd) are restricted to scalar-by-scalar operations, this engine is designed around **vectorized matrix calculus**. It maps full multi-dimensional tensors into a dynamic computational graph, allowing it to efficiently process entire training batches in parallel via optimized linear algebra routines.

---

## Core Architecture & Mechanics

The framework is built around three foundational pillars:

* **The Graph Engine (`Tensor`)**: Overloads native Python mathematical operators (`+`, `*`, `@`) to implicitly construct a Directed Acyclic Graph (DAG) during the forward pass. Every node tracks its inputs (`_prev`) and operational footprint (`_op`).
* **Topological Sorting (`.backward()`)**: Before computing gradients, the engine performs a recursive topological sort starting at the final loss scalar. This linearizes the DAG dependencies, ensuring that a node's gradient is never evaluated until every forward operation relying on that node has been fully unrolled.
* **Vectorized Chain Rule**: Once ordered, the framework traverses the sorted nodes in reverse, executing matrix calculus derivatives (such as $dL/dX = dL/dY \cdot W^T$) while elegantly handling multi-dimensional array broadcasting.
* **Modular Multi-Layer Perceptrons**: An object-oriented network layer architecture (`Module`, `Linear`, `MLP`) that manages weights, biases, and interchangeable activations (`ReLU`, `Tanh`, `Sigmoid`) to update parameters via Stochastic Gradient Descent.
