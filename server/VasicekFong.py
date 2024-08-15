from scipy.optimize import minimize
import numpy as np
import math


# TODO
class VasicekFong:

    @staticmethod
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
                    sum_df += beta * (
                        (1 - math.exp(-a * (t - k_t)))
                        - (1 - math.exp(-2 * a * (t - k_t)))
                        + (1 / 3) * (1 - math.exp(-3 * a * (t - k_t)))
                    )
        return sum_df

    @staticmethod
    def fit_vasicek_fong(maturities, yields, initial_params, knots):
        def objective(params):
            model_yields = []
            for t in maturities:
                df = VasicekFong.vf_discount_function(params, knots, t)
                model_yield = -math.log(df) / t
                model_yields.append(model_yield)
            return np.sum((np.array(model_yields) - np.array(yields)) ** 2)

        result = minimize(objective, initial_params, method="Nelder-Mead")
        return result.x

    @staticmethod
    def yield_curve_interpolation(fitted_params, knots, maturities):
        interpolated_yields = []
        for t in maturities:
            df = VasicekFong.vf_discount_function(fitted_params, knots, t)
            interpolated_yield = -math.log(df) / t
            interpolated_yields.append(interpolated_yield)
        return interpolated_yields
