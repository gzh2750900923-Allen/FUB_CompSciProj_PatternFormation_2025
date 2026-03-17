# Pattern Formation in Reaction-Diffusion Systems

## 1. Introduction
- This project implements Alan Turing’s theory of morphogenesis (1952) using numerical methods.

## 2. Implemented Models
- **Grey-Scott Model**: Used for standard maze and spot patterns.
- **Giraffe Markings**: Based on Murray's reaction model.
- **Leopard Rosettes**: Implemented via a Two-stage Turing model.

## 3. Numerical Solvers
- **Explicit Euler**: 2nd-order finite difference in space, 1st-order in time.
- **Crank-Nicolson**: Semi-implicit (IMEX) scheme for diffusion, providing unconditional stability.
- **Matrix-Free Solver**: Iterative solution using Conjugate Gradient for memory efficiency.

## 4. Parallelization (Task 3.4)
- Leverages **DASK** for distributed computing of both explicit and implicit solvers across multiple CPU cores.

## 5. How to Steup
#### 1. Prerequisites
- **Python 3.9+** (Tested on 3.13)
- **Virtual Environment** (Highly Recommended)
  - 1. Create and activate virtual environment
    - python3 -m venv venv
    - source venv/bin/activate  # Mac/Linux
    - venv\Scripts\activate     # For Windows
  - 2. Install dependencies
    - pip3 install -r requirements.txt
  - 3. Install the package locally
    - pip3 install -e .  
  - 4. Environment Configuration
    - export PYTHONPATH=$PYTHONPATH:.

## 6. How to Run
  - 1. Numerical Validation(Task 3.2)
    - python3 scripts/main_order_validation.py
  - 2. Pattern Generation(Task 3.3)
    - python3 scripts/main_task_3_3_final.py
  - 3. Dask Parallel Benchmark(Task 3.4)
    - python3 scripts/main_dask_benchmark.py
    - Dask Dashboard: Open http://127.0.0.1:8787 in your browser to monitor real-time CPU core utilization.
  - 4. Running Unit Tests
    - python3 -m unittest discover -v -s tests -p "test_*.py"
