### main_order_validation.py
  1. Validate the order of the method:
     - Using a log-log plot, we visually demonstrate that the Explicit Euler method has a slope of $1(O(Δt))$, while the Crank-Nicolson method has a slope of $2(O(Δt^2))$.
     - This serves as the strongest evidence that the algorithm has been implemented correctly.
  2. Benchmark performance
     - It calculates the error at each $Δt$ and indirectly demonstrates which method allows for a larger $Δt$ while achieving the same level of accuracy (efficiency).
