# adapted from: https://github.com/ahgperrin/PyCurve/blob/master/src/PyCurve/simulation.py

from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class Simulation:
    def __init__(self, simulated_paths: np.ndarray, dt: float) -> None:
        self._sim = self._is_valid_attr(simulated_paths)
        self._dt = dt

    @property
    def get_sim(self) -> np.ndarray:
        return self.__getattribute__("_sim")

    @property
    def get_nb_sim(self) -> int:
        return self.__getattribute__("_sim").shape[1]

    @property
    def get_steps(self) -> int:
        return self.__getattribute__("_sim").shape[0]

    @property
    def get_dt(self) -> float:
        return self.__getattribute__("_dt")

    @staticmethod
    def _is_valid_attr(attr: Any) -> np.ndarray:
        assert isinstance(
            attr, np.ndarray
        ), "Class Constructor takes only numpy arrays or list as arguments"
        return attr

    def yield_curve(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        discount_factor: np.ndarray = self.discount_factor()
        yield_curve = (
            np.mean(discount_factor, axis=1)
            ** (-1 / np.full(self.get_steps, self.get_dt).cumsum())
        ) - 1
        return np.full(self.get_steps, self.get_dt).cumsum(), yield_curve

    def discount_factor(self) -> np.array:
        rieman_sum: np.ndarray = np.zeros(shape=self.get_sim.shape)
        discount_factor: np.ndarray = np.zeros(shape=self.get_sim.shape)
        for sim in range(self.get_nb_sim):
            rieman_sum[:, sim] = np.cumsum(
                np.multiply(self.get_sim[:, sim], self.get_dt)
            )
            discount_factor[:, sim] = np.exp(-rieman_sum[:, sim])
        return discount_factor

    def plot_discount_curve(self, average: bool = False) -> None:
        discount_factor: np.ndarray = self.discount_factor()
        t: np.ndarray = np.full(self.get_steps, self.get_dt).cumsum()
        fig, ax = plt.subplots(1)
        fig.canvas.set_window_title("Discount Factor")
        fig.suptitle("Discount Factor")
        ax.set_xlabel("Time, t")
        ax.set_ylabel("Simulated Discount Factor")
        if average:
            ax.plot(t, np.mean(discount_factor, axis=1), c="navy")
        else:
            ax.plot(t, discount_factor)

    def plot_simulation(self) -> None:
        t: np.ndarray = np.full(self.get_steps, self.get_dt).cumsum()
        fig, ax = plt.subplots(1)
        fig.suptitle("Simulated Paths")
        fig.canvas.set_window_title("Simulated Paths")
        ax.set_xlabel("Time, t")
        ax.set_ylabel("Simulated Yield")
        ax.plot(t, self.get_sim, lw=0.5)
        plt.show()
        return fig
