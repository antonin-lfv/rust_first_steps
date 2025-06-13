<h1 align="center">
  <img src="https://github.com/user-attachments/assets/b002326b-11ce-4cfc-baab-ba70d28dbd69" width="260">
<br>
</br>
  Getting started with Rust
</h1>

---

### 🛠️ Build

To build the project:

```bash
cargo build
```

To run a specific example:

```bash
cargo run --bin <file_name_without_rs>
```

---

### 🤖 Machine Learning

- [Linear Regression (SVD)](src/bin/linear_regression_svd.rs) – Linear regression implemented from scratch using Singular Value Decomposition.
- [K-means](src/bin/kmeans.rs) – *(in progress)* Unsupervised clustering using centroids and Euclidean distance.

---

### 🧰 Utilities

- [Euclidean Distance](src/bin/euclidean_distance.rs) – Basic implementation of $\ell_2$ norm between two vectors.
- [CSV Reading](src/bin/handle_csv.rs) – Simple CSV parsing with `csv` and `serde`.
- [2D Plotting](src/bin/plot_2D_functions.rs) – Visualizing functions and model outputs using `plotly`.
- [Linear Algebra with `nalgebra`](src/bin/matrices_vectors_nalgebra.rs) – Matrix and vector operations using the `nalgebra` crate.
- [Linear Algebra with `ndarray`](src/bin/matrices_vectors_ndarray.rs) – Alternative linear algebra toolkit closer to NumPy-style arrays.

---

### 📁 Project Structure

This repository is organized with one file per experiment or module inside `src/bin/`. Each file is a standalone binary crate meant for hands-on learning and testing.
