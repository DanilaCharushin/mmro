import sys
import numpy as np
from typing import List
import scipy.interpolate as interp
import scipy.integrate as integr


class DistributionLawError(Exception):
    pass


class SmoothPointNumberError(Exception):
    pass


class Approximation:
    @staticmethod
    def approximate_least_squares(x: List[float], y: List[float], n=1) -> callable:
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        A = []
        b = []
        for i in range(n + 1):
            A.append([])
            b.append(sum(y[k] * x[k] ** i for k in range(len(x))))
            for j in range(n + 1):
                if i == j == 0:
                    A[i].append(len(x))
                else:
                    A[i].append(sum(x[k] ** (i + j) for k in range(len(x))))
        c = np.linalg.solve(np.array(A, dtype=np.float64), np.array(b, dtype=np.float64))
        print(f"{c=}")
        return lambda x: sum(c[i] * x ** i for i in range(len(c)))


class Interpolation:
    @staticmethod
    def interpolate_demo(x: List[float], y: List[float], kind: str) -> callable:
        if kind == "lagrange":
            return interp.lagrange(x, y)
        else:
            return interp.interp1d(x, y, kind=kind)

    @staticmethod
    def interpolate(x: List[float], y: List[float], n: int) -> callable:
        return interp.UnivariateSpline(x, y, s=0, k=n)


class Smoothing:
    @staticmethod
    def smooth(y: List[float], n=3) -> np.array:
        res = []
        if n == 3:
            if len(y) < 3:
                raise SmoothPointNumberError("Points number must be >= 3")
            else:
                res.append((5 * y[0] + 2 * y[1] - y[2]) / 6)
                for i in range(1, len(y) - 1):
                    res.append((y[i - 1] + y[i] + y[i + 1]) / 3)
                res.append((5 * y[-1] + 2 * y[-2] - y[-3]) / 6)
        elif n == 5:
            if len(y) < 5:
                raise SmoothPointNumberError("Points number must be >= 5")
            else:
                res.append((3 * y[0] + 2 * y[1] + y[2] - y[4]) / 5)
                res.append((4 * y[0] + 3 * y[1] + 2 * y[2] + y[3]) / 10)
                for i in range(2, len(y) - 2):
                    res.append((y[i - 2] + y[i - 1] + y[i] + y[i + 1] + y[i + 2]) / 5)
                res.append((4 * y[-1] + 3 * y[-2] + 2 * y[-3] + y[-4]) / 10)
                res.append((3 * y[-1] + 2 * y[-2] + y[-3] - y[-5]) / 5)
        elif n == 7:
            if len(y) < 7:
                raise SmoothPointNumberError("Points number must be >= 7")
            else:
                res.append((39 * y[0] + 8 * y[1] - 4 * (y[2] + y[3] - y[4]) + y[5] - 2 * y[6]) / 42)
                res.append((8 * y[0] + 19 * y[1] + 16 * y[2] + 6 * y[3] - 4 * y[4] - 7 * y[5] + 4 * y[6]) / 42)
                res.append((-4 * y[0] + 16 * y[1] + 19 * y[2] + 12 * y[3] + 2 * y[4] - 4 * y[5] + y[6]) / 42)
                for i in range(3, len(y) - 3):
                    res.append(
                        (7 * y[i] + 6 * (y[i + 1] + y[i - 1]) + 3 * (y[i + 2] + y[i - 2]) - 2 * (
                                y[i + 3] + y[i - 3])) / 21
                        )
                res.append((-4 * y[-1] + 16 * y[-2] + 19 * y[-3] + 12 * y[-4] + 2 * y[-5] - 4 * y[-6] + y[-7]) / 42)
                res.append(
                    (8 * y[-1] + 19 * y[-2] + 16 * y[-3] + 6 * y[-4] - 4 * y[-5] - 7 * y[-6] + 4 * y[-7]) / 42
                )
                res.append((39 * y[-1] + 8 * y[-2] - 4 * y[-3] - 4 * y[-4] + y[-5] + 4 * y[-6] - 2 * y[-7]) / 42)
        else:
            raise SmoothPointNumberError("Unknown smooth point number. Available: 3, 5, 7")
        return np.array(res, dtype=np.float64)


class Noise:
    @staticmethod
    def make_noise(y, p: float, law="uniform"):
        """
        p - уровень шума, от 0 до 1
        """
        eps = abs(y * p)
        if law == "uniform":
            return np.random.uniform(y - eps, y + eps)
        elif law == "normal":
            return np.random.normal(y, eps / 3)
        else:
            raise DistributionLawError("Unknown distribution type. Available: normal, uniform")