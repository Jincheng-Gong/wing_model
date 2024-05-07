'''
@Time: 2024/4/12 9:10 AM
@Author: Jincheng Gong
@Contact: Jincheng.Gong@hotmail.com
@File: black_scholes.py
@Desc: Black-Scholes Utilities include Black-Scholes Price, Implied Volatility and Greeks
'''

import numpy as np
from scipy.stats import norm

MAX_ITERS = 1000
MAX_ERROR = 10e-7


def bs_d1(f: float, k: float, t: float, v: float) -> float:
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :return: d1
    """
    return (np.log(f / k) + v * v * t / 2) / v / np.sqrt(t)


def bs_d2(f: float, k: float, t: float, v: float) -> float:
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :return: d2
    """
    return bs_d1(f, k, t, v) - v * np.sqrt(t)


def normal_distrib(z: float, mean=0, stdev=1) -> float:
    """
    :param z: Datapoint
    :param mean: Normal Distribution Expectation
    :param stdev: Normal Distribution Standard Deviation
    :return: Datapoint Normal Value
    """
    return norm.pdf(z, loc=mean, scale=stdev)


def snorm(z: float) -> float:
    """
    :param z: Datapoint
    :return: Datapoint Cumulative Normal Value
    """
    return norm.cdf(z)


def bs_price(f: float, k: float, t: float, v: float, r: float, opt_type: str) -> float:
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param r: Risk-free Rate (in %)
    :param opt_type: Either "c", "p", "c+p" or "c-p"
    :return: Black-Scholes Price
    """
    d1 = bs_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    switcher = {
        "c": np.exp(-r * t) * (f * snorm(d1) - k * snorm(d2)),
        "p": np.exp(-r * t) * (-f * snorm(-d1) + k * snorm(-d2)),
        "c+p": np.exp(-r * t) * (f * snorm(d1) - snorm(-d1)) - k * (snorm(d2) - snorm(-d2)),
        "c-p": np.exp(-r * t) * (f * snorm(d1) + snorm(-d1)) - k * (snorm(d2) + snorm(-d2)),
    }
    return switcher.get(opt_type.lower(), 0)


def bs_delta(f: float, k: float, t: float, v: float, r: float, opt_type: str) -> float:
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param r: Risk-free Rate (in %)
    :param opt_type: Either "c", "p", "c+p" or "c-p"
    :return: Black-Scholes Delta
    """
    d1 = bs_d1(f, k, t, v)
    switcher = {
        "c": np.exp(-r * t) * snorm(d1),
        "p": -np.exp(-r * t) * snorm(-d1),
        "c+p": np.exp(-r * t) * (snorm(d1) - snorm(-d1)),
        "c-p": np.exp(-r * t) * (snorm(d1) + snorm(-d1)),
    }
    return switcher.get(opt_type.lower(), 0)


def bs_dual_delta(f: float, k: float, t: float, v: float, r: float, opt_type: str) -> float:
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param r: Risk-free Rate (in %)
    :param opt_type: Either "c", "p", "c+p" or "c-p"
    :return: Black-Scholes Dual Delta
    """
    d2 = bs_d2(f, k, t, v)
    switcher = {
        "c": -np.exp(-r * t) * snorm(d2),
        "p": np.exp(-r * t) * snorm(-d2),
        "c+p": np.exp(-r * t) * (-snorm(d2) + snorm(-d2)),
        "c-p": np.exp(-r * t) * (-snorm(d2) - snorm(-d2)),
    }
    return switcher.get(opt_type.lower(), 0)


def bs_gamma(f: float, k: float, t: float, v: float, r: float, opt_type: str) -> float:
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param r: Risk-free Rate (in %)
    :param opt_type: Either "c", "p", "c+p" or "c-p"
    :return: Black-Scholes Gamma
    """
    d1 = bs_d1(f, k, t, v)
    fd1 = normal_distrib(d1)
    switcher = {
        "c": np.exp(-r * t) * fd1 / (f * v * np.sqrt(t)),
        "p": np.exp(-r * t) * fd1 / (f * v * np.sqrt(t)),
        "c+p": 2 * np.exp(-r * t) * fd1 / (f * v * np.sqrt(t)),
        "c-p": 0,
    }
    return switcher.get(opt_type.lower(), 0)


def bs_theta(f: float, k: float, t: float, v: float, r: float, opt_type: str) -> float:
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param r: Risk-free Rate (in %)
    :param opt_type: Either "c", "p", "c+p" or "c-p"
    :return: Black-Scholes Theta
    """
    d1 = bs_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    switcher = {
        "c": -np.exp(-r * t) * ((f * snorm(d1) * v) /
                                (2 * np.sqrt(t)) - (r * f * snorm(d1)) + (r * k * snorm(d2))),
        "p": -np.exp(-r * t) * ((f * snorm(d1) * v) /
                                (2 * np.sqrt(t)) + (r * f * snorm(-d1)) - (r * k * snorm(-d2))),
        "c+p": bs_theta(f, k, t, v, r, "c") + bs_theta(f, k, t, v, r, "p"),
        "c-p": bs_theta(f, k, t, v, r, "c") - bs_theta(f, k, t, v, r, "p"),
    }
    return switcher.get(opt_type.lower(), 0)


def bs_vega(f: float, k: float, t: float, v: float, r: float, opt_type: str) -> float:
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param r: Risk-free Rate (in %)
    :param opt_type: Either "c", "p", "c+p" or "c-p"
    :return: Black-Scholes Vega
    """
    d1 = bs_d1(f, k, t, v)
    fd1 = normal_distrib(d1)
    switcher = {
        "c": np.exp(-r * t) * f * fd1 * np.sqrt(t),
        "p": np.exp(-r * t) * f * fd1 * np.sqrt(t),
        "c+p": 2 * np.exp(-r * t) * f * fd1 * np.sqrt(t),
        "c-p": 0,
    }
    return switcher.get(opt_type.lower(), 0)


def bs_vanna(f: float, k: float, t: float, v: float, r: float, opt_type: str) -> float:
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param r: Risk-free Rate (in %)
    :param opt_type: Either "c", "p", "c+p" or "c-p"
    :return: Black-Scholes Vanna
    """
    d1 = bs_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    fd1 = normal_distrib(d1)
    switcher = {
        "c": -np.exp(-r * t) * fd1 * d2 / v,
        "p": -np.exp(-r * t) * fd1 * d2 / v,
        "c+p": -2 * np.exp(-r * t) * fd1 * d2 / v,
        "c-p": 0,
    }
    return switcher.get(opt_type.lower(), 0)


def bs_volga(f: float, k: float, t: float, v: float, r: float, opt_type: str) -> float:
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param v: Constant Annual Volatility (in %)
    :param r: Risk-free Rate (in %)
    :param opt_type: Either "c", "p", "c+p" or "c-p"
    :return: Black-Scholes Volga
    """
    d1 = bs_d1(f, k, t, v)
    d2 = d1 - v * np.sqrt(t)
    fd1 = normal_distrib(d1)
    switcher = {
        "c": np.exp(-r * t) * f * np.sqrt(t) * fd1 * np.sqrt(t) * d1 * d2,
        "p": np.exp(-r * t) * f * np.sqrt(t) * fd1 * np.sqrt(t) * d1 * d2,
        "c+p": 2 * np.exp(-r * t) * f * np.sqrt(t) * fd1 * np.sqrt(t) * d1 * d2,
        "c-p": 0,
    }
    return switcher.get(opt_type.lower(), 0)


def bs_iv_newton_raphson(f: float, k: float, t: float, mkt_price: float,
                         r: float, opt_type: str) -> float:
    """
    :param f: Forward (in %)
    :param k: Strike (in %)
    :param t: Maturity (in Years)
    :param mkt_price: Option Market Price (in %)
    :param r: Risk-free Rate (in %)
    :param opt_type: Either "c", "p", "c+p" or "c-p"
    :return: Black-Scholes Implied Volatility
    """
    iter_numb = 0
    v = 0.30
    bs_price_error = mkt_price - bs_price(f, k, t, v, r, opt_type)
    while ((abs(bs_price_error) > MAX_ERROR) and (iter_numb < MAX_ITERS)):
        vega = bs_vega(f, k, t, v, r, opt_type)
        if vega == 0:
            return -1, iter_numb
        if vega != 0:
            v += bs_price_error / vega
            iter_numb += 1
    return v, iter_numb


if __name__ == "__main__":
    print("Black-Scholes utilities powered by Jincheng Gong.")
