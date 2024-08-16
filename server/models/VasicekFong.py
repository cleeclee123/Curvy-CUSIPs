import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define Vasicek-Fong discount function
def vf_discount_function(parameters, knots, t):
    sum_df = 1
    a = parameters[0]
    b = parameters[1:]
    num_b = len(b) - len(knots)
    for i, beta in enumerate(b):
        if i < num_b:
            sum_df += beta * (1 - math.exp(-(i + 1) * a * t))
        else:
            k_t = knots[i - num_b]
            if t >= k_t:
                sum_df += beta * ((1 - math.exp(-a * (t - k_t)))
                                  - (1 - math.exp(-2 * a * (t - k_t)))
                                  + (1 / 3) * (1 - math.exp(-3 * a * (t - k_t))))
    return sum_df

# Define function to fit Vasicek-Fong model
def fit_vasicek_fong(maturities, yields, initial_params, knots):
    def objective(params):
        model_yields = []
        for t in maturities:
            df = vf_discount_function(params, knots, t)
            model_yield = -math.log(df) / t
            model_yields.append(model_yield)
        return np.sum((np.array(model_yields) - np.array(yields))**2)
    
    result = minimize(objective, initial_params, method='Nelder-Mead')
    return result.x

# Define interpolation function
def yield_curve_interpolation(fitted_params, knots, maturities):
    interpolated_yields = []
    for t in maturities:
        df = vf_discount_function(fitted_params, knots, t)
        interpolated_yield = -math.log(df) / t
        interpolated_yields.append(interpolated_yield)
    return interpolated_yields


# Input pre-calculated parameters and knot points
initial_params = [0.0304, -0.5577, -0.8063, 0.5342, -2.5088, 0.6706, 0.7541, 0.3981, 0.1756]
knots = [0.5, 2, 5, 10, 30]

# Define maturities (in years) and yields (in decimal form) for fitting
otr_maturities = [0.5, 1, 2, 3, 5, 7, 10, 30]
otr_yields = [0.02, 0.022, 0.025, 0.027, 0.03, 0.032, 0.035, 0.04]

# Fit the Vasicek-Fong model to the given yields
fitted_params = fit_vasicek_fong(otr_maturities, otr_yields, initial_params, knots)

# Create interpolated yields for plotting
interpolated_maturities = np.linspace(0.5, 30, 100)
interpolated_yields = yield_curve_interpolation(fitted_params, knots, interpolated_maturities)

# Plot the yield curve
plt.plot(interpolated_maturities, interpolated_yields, label='Fitted Yield Curve')
plt.scatter(otr_maturities, otr_yields, color='red', marker='x', label='Original Yields')
plt.xlabel('Maturity (years)')
plt.ylabel('Yield')
plt.title('Vasicek-Fong Fitted Yield Curve')
plt.legend()
plt.grid(True)
plt.show()
