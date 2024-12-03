import numpy as np
from scipy.stats import norm

# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2420757
# https://www.clarusft.com/analytic-implied-basis-point-volatility/
# https://www.clarusft.com/bachelier-model-fast-accurate-implied-volatility/

def bachelier_implied_vol(C, F, K, T, option_type='call'):
    """
    Compute the implied normal volatility (basis point volatility) under the Bachelier model.

    Parameters:
    - C: Option price
    - F: Forward price
    - K: Strike price
    - T: Time to expiry (in years)
    - option_type: 'call' or 'put'

    Returns:
    - sigma: Implied normal volatility
    """
    x = F - K
    if option_type == 'put':
        x = -x  # Adjust for put option

    if x == 0:
        # ATM case
        sigma = C * np.sqrt(2 * np.pi / T)
        return sigma

    # z is the moneyness ratio
    if x > 0:
        z = (C - x) / x
    else:
        z = -C / x

    # For in-the-money options
    if z > 0.15:
        # Near-the-money approximation
        eta = -z / np.log1p(-z)
        # Rational function approximation for h(eta)
        numerator = (
            0.06155371425063157 +
            eta * (2.723711658728403 +
            eta * (10.83806891491789 +
            eta * (301.0827907126612 +
            eta * (1082.864564205999 +
            eta * (790.7079667603721 +
            eta * (109.330638190985 +
            eta * 0.1515726686825187))))))
        )
        denominator = (
            1.0 +
            eta * (1.436062756519326 +
            eta * (118.6674859663193 +
            eta * (441.1914221318738 +
            eta * (313.4771127147156 +
            eta * 40.90187645954703))))
        )
        h = numerator / denominator
        sigma = C * h / np.sqrt(T)
        return sigma

    # For out-of-the-money options
    else:
        # Parameters for rational function approximation
        beta_start = -np.log(0.15)
        beta_end = -np.log(np.finfo(float).tiny)
        u = -(np.log(z) + beta_start) / (beta_end - beta_start)
        # Determine which approximation to use based on u
        if u < 0.0091:
            # Use the first rational function
            numerator = (
                0.6409168551974356 +
                u * (788.5769356915809 +
                u * (445231.8217873989 +
                u * (149904950.4316367 +
                u * (32696572166.83277 +
                u * (4679633190389.852 +
                u * (420159669603232.9 +
                u * 2.053009222143781e16))))))
            )
            denominator = (
                1.0 +
                u * (644.3895239520736 +
                u * (211503.4461395385 +
                u * (42017301.42101825 +
                u * (5311468782.258145 +
                u * (411727826816.0715 +
                u * (17013504968737.03 +
                u * 247411313213747.3))))))
            )
        elif u < 0.088:
            # Use the second rational function
            numerator = (
                0.6421698396894946 +
                u * (639.0799338046976 +
                u * (278070.4504753253 +
                u * (64309618.34521588 +
                u * (8434470508.516712 +
                u * (429163238246.6056 +
                u * (8127970878235.127 +
                u * 53601225394979.81))))))
            )
            denominator = (
                1.0 +
                u * (428.4860093838116 +
                u * (86806.89002606465 +
                u * (8635134.393384729 +
                u * (368872214.1525768 +
                u * (6359299149.626331 +
                u * (39926015967.88848 +
                u * 67434966969.06365))))))
            )
        else:
            # Use the third rational function
            numerator = (
                0.9419766804760195 +
                u * (319.5904313022832 +
                u * (169280.1584005307 +
                u * (7680298.116948191 +
                u * (102052455.1237945 +
                u * (497528976.6077898 +
                u * (930641173.0039455 +
                u * 619268950.1849232))))))
            )
            denominator = (
                1.0 +
                u * (170.3825619167351 +
                u * (6344.159541465554 +
                u * (78484.04408022196 +
                u * (370696.1131305614 +
                u * (682908.5433659635 +
                u * (451067.0450625782 +
                u * 79179.06152239779))))))
            )
        h = numerator / denominator
        sigma = abs(x) / np.sqrt(h * T)
        return sigma
