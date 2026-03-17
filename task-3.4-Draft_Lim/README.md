# Pattern Formation in Reaction-Diffusion Systems

## 1. Introduction
[cite_start]This project implements Alan Turing’s theory of morphogenesis (1952) using numerical methods[cite: 1, 34].

## 2. Implemented Models
- [cite_start]**Grey-Scott Model**: Used for standard maze and spot patterns[cite: 18, 37].
- [cite_start]**Giraffe Markings**: Based on Murray's reaction model[cite: 20, 38].
- [cite_start]**Leopard Rosettes**: Implemented via a Two-stage Turing model[cite: 22, 39].

## 3. Numerical Solvers
- [cite_start]**Explicit Euler**: 2nd-order finite difference in space, 1st-order in time[cite: 15].
- [cite_start]**Crank-Nicolson**: Semi-implicit (IMEX) scheme for diffusion, providing unconditional stability[cite: 16].
- [cite_start]**Matrix-Free Solver**: Iterative solution using Conjugate Gradient for memory efficiency[cite: 31].

## 4. Parallelization (Task 3.4)
[cite_start]Leverages **DASK** for distributed computing of both explicit and implicit solvers across multiple CPU cores.

## 5. How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run benchmark: `python main_dask_benchmark.py`
3. Run unit tests: `python -m unittest unittest_solvers.py`
