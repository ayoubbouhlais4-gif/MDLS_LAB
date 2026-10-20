import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# raise NotImplementedError()
print(np.__version__)
print(pd.__version__)
print(plt.matplotlib.__version__)
print(sns.__version__)
import types

assert "np" in globals(), "❌ 'np' is not defined. Did you run 'import numpy as np'?"
assert isinstance(
    globals()["np"], types.ModuleType
), "❌ 'np' exists but is not a valid Python module."
assert (
    globals()["np"].__name__ == "numpy"
), "❌ 'np' exists but does not point to the 'numpy' package."

print("✅ NumPy is correctly imported as 'np'.")
### END TESTS
import numpy as np

# Initialize S, theta and bias
S = np.array([[10, 4], [2, 1], [7, 3], [1, 2]])
theta = np.array([0.3, 0.7])
bias = 0.5

# 1. Compute linear combination z
z = S @ theta + bias
print("z =", z)
print("First value is 6.3:", np.isclose(z[0], 6.3))

# 2. Compute sigmoid probabilities and predict spam (probability >= 0.5)
probabilities = 1 / (1 + np.exp(-z))
print("probabilities =", probabilities)

# Identify predicted spam emails
predicted_spam = probabilities >= 0.5
print("predicted_spam =", predicted_spam)
print("Spam e-mail indices:", np.where(predicted_spam)[0])
# raise NotImplementedError()