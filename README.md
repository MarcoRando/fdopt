# FDOpt

Code used to perform the experiments of the paper "A Structured Tour of Optimization with Finite Differences"


## :package: Installation

We recommend installing this library in a dedicated [Conda](https://docs.conda.io/en/latest/) environment to manage dependencies cleanly.

### 1. Create a Conda environment

```bash
conda create -n fdopt python=3.10
conda activate fdopt
```

### 2. Install the library

Clone the repository
```bash
git clone https://github.com/MarcoRando/fdopt.git
cd fdopt
```

Install the library from the local source using `pip`:

```bash
pip install .
```

### 3. Install PyCUTEst ( for CUTEst experiments)

To enable optimization experiments with CUTEst problems, install `pycutest` and its dependency `cutest`.

Please follow the detailed installation instructions at the official PyCUTEst GitHub repository:

👉 [https://jfowkes.github.io/pycutest/_build/html/install.html](https://jfowkes.github.io/pycutest/_build/html/install.html)

> **Note:** CUTEst requires a Fortran compiler and additional setup. Follow all the steps from the PyCUTEst documentation to ensure correct installation.

### 4. Install PyTorch (with CUDA support)

Install a version of PyTorch compatible with your system's NVIDIA CUDA version.

Visit the official PyTorch [Get Started](https://pytorch.org/get-started/locally/) page and select the appropriate options for your system.

---

## \:file\_folder: Experiments

All experiments reported in the paper can be found in the [`experiments/`](./experiments) directory.
This includes scripts to run the experiments, as well as the scripts used to generate the figures and tables.

---

