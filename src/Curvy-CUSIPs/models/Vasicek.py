from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ..utils.simulation import Simulation


class Vasicek:

    def __init__(
        self,
        alpha: float,
        beta: float,
        sigma: float,
        rt: float,
        time: float,
        delta_time: float,
    ) -> None:
        self._alpha = alpha
        self._beta = beta
        self._sigma = sigma
        self._rt = rt
        self._dt = delta_time
        self._steps = int(time / delta_time)

    def get_attr(self, attr: str) -> Union[float, int]:
        return self.__getattribute__(attr)

    def _sigma_part(self, n: int) -> float:
        return (
            self.get_attr("_sigma")
            * np.sqrt(self.get_attr("_dt"))
            * np.random.normal(size=n)
        )

    def _mu_dt(self, rt: np.ndarray) -> float:
        return (
            self.get_attr("_alpha")
            * (self.get_attr("_beta") - rt)
            * self.get_attr("_dt")
        )

    def simulate_paths(self, n: int) -> np.array:
        simulation = np.zeros(shape=(self.get_attr("_steps"), n))
        simulation[0, :] = self._rt
        for i in range(1, self.get_attr("_steps"), 1):
            dr = self._mu_dt(simulation[i - 1, :]) + self._sigma_part(n)
            simulation[i, :] = simulation[i - 1, :] + dr
        return Simulation(simulation, self.get_attr("_dt"))

    @staticmethod
    def plot_calibrated(
        simul: Simulation,
        instantaneous_forward: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ) -> None:
        fig = plt.figure(figsize=(12.5, 8))
        fig.suptitle("Model Fitting Curve T=0")
        fig.canvas.set_window_title("Model Fitting Curve T=0")
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel("t, years")
        ax1.set_ylabel("Yield")
        ax1.plot(
            np.linspace(1, simul.get_steps, simul.get_steps) * simul.get_dt,
            simul.get_sim,
            lw=0.5,
        )
        ax1.plot(
            np.linspace(1, simul.get_steps, simul.get_steps) * simul.get_dt,
            simul.yield_curve().get_rate,
            lw=3,
            c="Navy",
            label="Vasicek Term Structure",
        )
        ax1.plot(
            instantaneous_forward.get_time,
            instantaneous_forward.get_rate,
            c="darkred",
            label="Initial Term Structure",
            lw=3,
        )
        plt.legend()
        plt.show()
        return fig
