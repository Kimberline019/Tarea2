# Tutorial

Ejemplo de uso del módulo:

```python
from cuadrature import gaussian_quadrature
import numpy as np

f = lambda x: np.sin(x**2)
resultado = gaussian_quadrature(f, 0, np.pi, 10)
print(resultado)

