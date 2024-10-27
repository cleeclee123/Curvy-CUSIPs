"""
Base class for cointegration approach in statistical arbitrage.
"""

from abc import ABC
import pandas as pd


def construct_spread(price_data: pd.DataFrame, hedge_ratios: pd.Series, dependent_variable: str = None) -> pd.Series:
    """
    Construct spread from `price_data` and `hedge_ratios`. If a user sets `dependent_variable` it means that a
    spread will be:

    hedge_ratio_dependent_variable * dependent_variable - sum(hedge_ratios * other variables).
    Otherwise, spread is:  hedge_ratio_0 * variable_0 - sum(hedge ratios * variables[1:]).

    :param price_data: (pd.DataFrame) Asset prices data frame.
    :param hedge_ratios: (pd.Series) Hedge ratios series (index-tickers, values-hedge ratios).
    :param dependent_variable: (str) Dependent variable to use. Set None for dependent variable being equal to 0 column.
    :return: (pd.Series) Spread series.
    """

    weighted_prices = price_data * hedge_ratios  # price * hedge

    if dependent_variable is not None:
        non_dependent_variables = [x for x in weighted_prices.columns if x != dependent_variable]
        return weighted_prices[dependent_variable] - weighted_prices[non_dependent_variables].sum(axis=1)

    return weighted_prices[weighted_prices.columns[0]] - weighted_prices[weighted_prices.columns[1:]].sum(axis=1)


class CointegratedPortfolio(ABC):
    """
    Class for portfolios formed using the cointegration method (Johansen test, Engle-Granger test).
    """

    def construct_mean_reverting_portfolio(self, price_data: pd.DataFrame, cointegration_vector: pd.Series = None) -> pd.Series:
        """
        When cointegration vector was formed, this function is used to multiply asset prices by cointegration vector
        to form mean-reverting portfolio which is analyzed for possible trade signals.

        :param price_data: (pd.DataFrame) Price data with columns containing asset prices.
        :param cointegration_vector: (pd.Series) Cointegration vector used to form a mean-reverting portfolio.
            If None, a cointegration vector with maximum eigenvalue from fit() method is used.
        :return: (pd.Series) Cointegrated portfolio dollar value.
        """

        if cointegration_vector is None:
            cointegration_vector = self.cointegration_vectors.iloc[0]  # Use eigenvector with biggest eigenvalue.

        return (cointegration_vector * price_data).sum(axis=1)

    def get_scaled_cointegration_vector(self, cointegration_vector: pd.Series = None) -> pd.Series:
        """
        This function returns the scaled values of the cointegration vector in terms of how many units of other
        cointegrated assets should be bought if we buy one unit of one asset.

        :param cointegration_vector: (pd.Series) Cointegration vector used to form a mean-reverting portfolio.
            If None, a cointegration vector with maximum eigenvalue from fit() method is used.
        :return: (pd.Series) The scaled cointegration vector values.
        """

        if cointegration_vector is None:
            cointegration_vector = self.cointegration_vectors.iloc[0]  # Use eigenvector with biggest eigenvalue

        scaling_coefficient = 1 / cointegration_vector.iloc[0]  # Calculating the scaling coefficient

        # Calculating the scaled cointegration vector
        scaled_cointegration_vector = cointegration_vector * scaling_coefficient

        return scaled_cointegration_vector


import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class JohansenPortfolio(CointegratedPortfolio):
    """
    The class implements the construction of a mean-reverting portfolio using eigenvectors from
    the Johansen cointegration test. It also checks Johansen (eigenvalue and trace statistic) tests
    for the presence of cointegration for a given set of assets.
    """

    def __init__(self):
        """
        Class constructor.
        """

        self.price_data = None  # pd.DataFrame with price data used to fit the model.
        self.cointegration_vectors = None  # Johansen eigenvectors used to form mean-reverting portfolios.
        self.hedge_ratios = None  # Johansen hedge ratios.
        self.johansen_trace_statistic = None  # Trace statistic data frame for each asset used to test for cointegration.
        self.johansen_eigen_statistic = None  # Eigenvalue statistic data frame for each asset used to test for cointeg.

    def fit(self, price_data: pd.DataFrame, dependent_variable: str = None, det_order: int = 0, n_lags: int = 1):
        """
        Finds cointegration vectors from the Johansen test used to form a mean-reverting portfolio.

        Note: Johansen test yields several linear combinations that may yield mean-reverting portfolios. The function
        stores all of them in decreasing order of eigenvalue meaning that the first linear combination forms the most
        mean-reverting portfolio which is used in trading. However, researchers may use other stored cointegration vectors
        to check other portfolios.

        This function will calculate and set johansen_trace_statistic and johansen_eigen_statistic only if
        the number of variables in the input dataframe is <=12. Otherwise it will generate a warning.

        A more detailed description of this method can be found on p. 54-58 of
        `"Algorithmic Trading: Winning Strategies and Their Rationale" by Ernie Chan
        <https://www.wiley.com/en-us/Algorithmic+Trading%3A+Winning+Strategies+and+Their+Rationale-p-9781118460146>`_.

        This function is a wrapper around the coint_johansen function from the statsmodels.tsa module. Detailed
        descriptions of this function are available in the
        `statsmodels documentation
        <https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html>`_.

        :param price_data: (pd.DataFrame) Price data with columns containing asset prices.
        :param dependent_variable: (str) Column name which represents the dependent variable (y).
            By default, the first column is used as a dependent variable.
        :param det_order: (int) -1 for no deterministic term in Johansen test, 0 - for constant term, 1 - for linear trend.
        :param n_lags: (int) Number of lags used in the Johansen test. The practitioners use 1 as the default base value.
        """

        if not dependent_variable:
            dependent_variable = price_data.columns[0]

        self.price_data = price_data

        test_res = coint_johansen(price_data, det_order=det_order, k_ar_diff=n_lags)

        # Store eigenvectors in decreasing order of eigenvalues
        cointegration_vectors = pd.DataFrame(test_res.evec[:, test_res.ind].T, columns=price_data.columns)

        # Adjusting cointegration vectors to hedge ratios
        data_copy = price_data.copy()
        data_copy.drop(columns=dependent_variable, axis=1, inplace=True)

        # Store cointegration vectors
        self.cointegration_vectors = pd.DataFrame(test_res.evec[:, test_res.ind].T, columns=price_data.columns)

        # Calculating hedge ratios
        all_hedge_ratios = pd.DataFrame()

        # Convert to a format expected by `construct_spread` function and
        # normalize such that dependent has a hedge ratio 1.
        for vector in range(cointegration_vectors.shape[0]):

            hedge_ratios = cointegration_vectors.iloc[vector].to_dict()
            for ticker, ratio in hedge_ratios.items():
                if ticker != dependent_variable:
                    # Set value to be list to make it easier to read into pandas DataFrame
                    hedge_ratios[ticker] = [-ratio / hedge_ratios[dependent_variable]]
            # Set value to be list to make it easier to read into pandas DataFrame
            hedge_ratios[dependent_variable] = [1.0]

            # Concat together in one DataFrame
            all_hedge_ratios = pd.concat([all_hedge_ratios, pd.DataFrame(hedge_ratios)])

        self.hedge_ratios = all_hedge_ratios

        # Test critical values are available only if number of variables <= 12
        if price_data.shape[1] <= 12:
            # Eigenvalue test
            self.johansen_eigen_statistic = pd.DataFrame(test_res.max_eig_stat_crit_vals.T, columns=price_data.columns, index=["90%", "95%", "99%"])
            self.johansen_eigen_statistic.loc["eigen_value"] = test_res.max_eig_stat.T
            self.johansen_eigen_statistic.sort_index(ascending=False)

            # Trace statistic
            self.johansen_trace_statistic = pd.DataFrame(test_res.trace_stat_crit_vals.T, columns=price_data.columns, index=["90%", "95%", "99%"])
            self.johansen_trace_statistic.loc["trace_statistic"] = test_res.trace_stat.T
            self.johansen_trace_statistic.sort_index(ascending=False)


import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from typing import Tuple


class EngleGrangerPortfolio(CointegratedPortfolio):
    """
    The class implements the construction of a mean-reverting portfolio using the two-step Engle-Granger method.
    It also tests model residuals for unit-root (presence of cointegration).
    """

    # pylint: disable=invalid-name
    def __init__(self):
        """
        Class constructor method.
        """

        self.price_data = None  # pd.DataFrame with price data used to fit the model.
        self.residuals = None  # OLS model residuals.
        self.dependent_variable = None  # Column name for dependent variable used in OLS estimation.
        self.cointegration_vectors = None  # Regression coefficients used as hedge-ratios.
        self.hedge_ratios = None  # Engle-Granger hedge ratios.
        self.adf_statistics = None  # ADF statistics.

    def perform_eg_test(self, residuals: pd.Series):
        """
        Perform Engle-Granger test on model residuals and generate test statistics and p values.

        :param residuals: (pd.Series) OLS residuals.
        """
        test_result = adfuller(residuals)
        critical_values = test_result[4]
        self.adf_statistics = pd.DataFrame(index=["99%", "95%", "90%"], data=critical_values.values())
        self.adf_statistics.loc["statistic_value", 0] = test_result[0]

    def fit(self, price_data: pd.DataFrame, add_constant: bool = False):
        """
        Finds hedge-ratios using a two-step Engle-Granger method to form a mean-reverting portfolio.
        By default, the first column of price data is used as a dependent variable in OLS estimation.

        This method was originally described in `"Co-integration and Error Correction: Representation,
        Estimation, and Testing," Econometrica, Econometric Society, vol. 55(2), pages 251-276, March 1987
        <https://www.jstor.org/stable/1913236>`_ by Engle, Robert F and Granger, Clive W J.

        :param price_data: (pd.DataFrame) Price data with columns containing asset prices.
        :param add_constant: (bool) A flag to add a constant term in linear regression.
        """

        self.price_data = price_data
        self.dependent_variable = price_data.columns[0]

        # Fit the regression
        hedge_ratios, _, _, residuals = self.get_ols_hedge_ratio(
            price_data=price_data, dependent_variable=self.dependent_variable, add_constant=add_constant
        )
        self.cointegration_vectors = pd.DataFrame(
            [np.append(1, -1 * np.array([hedge for ticker, hedge in hedge_ratios.items() if ticker != self.dependent_variable]))],
            columns=price_data.columns,
        )

        self.hedge_ratios = pd.DataFrame(
            [np.append(1, np.array([hedge for ticker, hedge in hedge_ratios.items() if ticker != self.dependent_variable]))],
            columns=price_data.columns,
        )

        # Get model residuals
        self.residuals = residuals
        self.perform_eg_test(self.residuals)

    @staticmethod
    def get_ols_hedge_ratio(
        price_data: pd.DataFrame, dependent_variable: str, add_constant: bool = False
    ) -> Tuple[dict, pd.DataFrame, pd.Series, pd.Series]:
        """
        Get OLS hedge ratio: y = beta*X.

        :param price_data: (pd.DataFrame) Data Frame with security prices.
        :param dependent_variable: (str) Column name which represents the dependent variable (y).
        :param add_constant: (bool) Boolean flag to add constant in regression setting.
        :return: (Tuple) Hedge ratios, X, and y and OLS fit residuals.
        """

        ols_model = LinearRegression(fit_intercept=add_constant)

        X = price_data.copy()
        X.drop(columns=dependent_variable, axis=1, inplace=True)
        exogenous_variables = X.columns.tolist()
        if X.shape[1] == 1:
            X = X.values.reshape(-1, 1)

        y = price_data[dependent_variable].copy()

        ols_model.fit(X, y)
        residuals = y - ols_model.predict(X)

        hedge_ratios = ols_model.coef_
        hedge_ratios_dict = dict(zip([dependent_variable] + exogenous_variables, np.insert(hedge_ratios, 0, 1.0)))

        return hedge_ratios_dict, X, y, residuals
