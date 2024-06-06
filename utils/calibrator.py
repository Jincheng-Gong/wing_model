'''
@Time: 2024/4/9 2:32 PM
@Author: Jincheng Gong
@Contact: Jincheng.Gong@hotmail.com
@File: calibrator.py
@Desc: ORC Wing Model Calibrator with Durrleman Condition
'''

import random
from typing import List

import numpy as np
from scipy import optimize


def wing_model_durrleman_condition(vr_: float, sr_: float, pc_: float, cc_: float,
                                   dc_: float, uc_: float, dsm_: float, usm_: float,
                                   vcr_: float = 0, scr_: float = 0, ssr_: float = 100,
                                   atm_: float = 1, ref_: float = 1) -> List[List[float]]:
    """
    :param vr_: volatility reference
    :param sr_: slope reference
    :param pc_: put curvature
    :param cc_: call curvature
    :param dc_: down cutoff
    :param uc_: up cutoff
    :param dsm_: down smoothing range
    :param usm_: up smoothing range
    :param atm_: atm forward
    :param ref_: reference forward price
    :param vcr_: volatility change rate
    :param scr_: slope change rate
    :param ssr_: skew swimmingness rate
    :return: wing model durrleman conditon for each moneyness
    """
    moneyness_list = np.linspace(dc_, uc_, 50)
    vc_ = vr_ - vcr_ * ssr_ * ((atm_ - ref_) / ref_)
    sc_ = sr_ - scr_ * ssr_ * ((atm_ - ref_) / ref_)
    g_list = []
    for moneyness in moneyness_list:
        if moneyness <= dc_ * (1 + dsm_):
            g_list.append(1)
        elif dc_ * (1 + dsm_) < moneyness < dc_:  # dc_ * (1 + dsm_) < moneyness <= dc_
            a = - pc_ / dsm_ - 0.5 * sc_ / (dc_ * dsm_)
            b1 = -0.25 * ((1 + 1 / dsm_) * (2 * dc_ * pc_ + sc_) - 2 *
                          (pc_ / dsm_ + 0.5 * sc_ / (dc_ * dsm_)) * moneyness) ** 2
            b2 = -dc_ ** 2 * (1 + 1 / dsm_) * pc_ - 0.5 * dc_ * sc_ / \
                dsm_ + vc_ + (1 + 1 / dsm_) * (2 * dc_ * pc_ + sc_) * \
                moneyness - (pc_ / dsm_ + 0.5 * sc_ /
                             (dc_ * dsm_)) * moneyness ** 2
            b2 = 0.25 + 1 / b2
            b = b1 * b2
            c1 = moneyness * ((1 + 1 / dsm_) * (2 * dc_ * pc_ + sc_) -
                              2 * (pc_ / dsm_ + 0.5 * sc_ / (dc_ * dsm_)) * moneyness)
            c2 = 2 * (-dc_ ** 2 * (1 + 1 / dsm_) * pc_ - 0.5 * dc_ * sc_ /
                      dsm_ + vc_ + (1 + 1 / dsm_) * (2 * dc_ * pc_ + sc_) *
                      moneyness - (pc_ / dsm_ + 0.5 * sc_ / (dc_ * dsm_)) * moneyness ** 2)
            c = (1 - c1 / c2) ** 2
            g_list.append(a + b + c)
        elif dc_ <= moneyness <= 0:  # dc_ < moneyness <= 0
            g_list.append(pc_ - 0.25 * (sc_ + 2 * pc_ * moneyness) ** 2 *
                          (0.25 + 1 / (vc_ + sc_ * moneyness +
                           pc_ * moneyness * moneyness))
                          + (1 - 0.5 * moneyness * (sc_ + 2 * pc_ * moneyness) /
                              (vc_ + sc_ * moneyness + pc_ * moneyness * moneyness)) ** 2)
        elif 0 < moneyness <= uc_:
            g_list.append(cc_ - 0.25 * (sc_ + 2 * cc_ * moneyness) ** 2 *
                          (0.25 + 1 / (vc_ + sc_ * moneyness +
                           cc_ * moneyness * moneyness))
                          + (1 - 0.5 * moneyness * (sc_ + 2 * cc_ * moneyness) /
                             (vc_ + sc_ * moneyness + cc_ * moneyness * moneyness)) ** 2)
        elif uc_ < moneyness <= uc_ * (1 + usm_):
            a = - cc_ / usm_ - 0.5 * sc_ / (uc_ * usm_)
            b1 = -0.25 * ((1 + 1 / usm_) * (2 * uc_ * cc_ + sc_) - 2 *
                          (cc_ / usm_ + 0.5 * sc_ / (uc_ * usm_)) * moneyness) ** 2
            b2 = -uc_ ** 2 * (1 + 1 / usm_) * cc_ - 0.5 * uc_ * sc_ / \
                usm_ + vc_ + (1 + 1 / usm_) * (2 * uc_ * cc_ + sc_) * \
                moneyness - (cc_ / usm_ + 0.5 * sc_ /
                             (uc_ * usm_)) * moneyness ** 2
            b2 = 0.25 + 1 / b2
            b = b1 * b2
            c1 = moneyness * ((1 + 1 / usm_) * (2 * uc_ * cc_ + sc_) -
                              2 * (cc_ / usm_ + 0.5 * sc_ / (uc_ * usm_)) * moneyness)
            c2 = 2 * (-uc_ ** 2 * (1 + 1 / usm_) * cc_ - 0.5 * uc_ * sc_ /
                      usm_ + vc_ + (1 + 1 / usm_) * (2 * uc_ * cc_ + sc_) *
                      moneyness - (cc_ / usm_ + 0.5 * sc_ / (uc_ * usm_)) * moneyness ** 2)
            c = (1 - c1 / c2) ** 2
            g_list.append(a + b + c)
        elif uc_ * (1 + usm_) < moneyness:
            g_list.append(1)
    return [moneyness_list, g_list]


def wing_model(k: float,
               vr_: float, sr_: float, pc_: float, cc_: float,
               dc_: float, uc_: float, dsm_: float, usm_: float,
               vcr_: float = 0, scr_: float = 0, ssr_: float = 100,
               atm_: float = 1, ref_: float = 1) -> float:
    """
    :param k: log forward moneyness
    :param vr_: volatility reference
    :param sr_: slope reference
    :param pc_: put curvature
    :param cc_: call curvature
    :param dc_: down cutoff
    :param uc_: up cutoff
    :param dsm_: down smoothing range
    :param usm_: up smoothing range
    :param atm_: atm forward
    :param ref_: reference forward price
    :param vcr_: volatility change rate
    :param scr_: slope change rate
    :param ssr_: skew swimmingness rate
    :return: wing model volatility
    """
    vc_ = vr_ - vcr_ * ssr_ * ((atm_ - ref_) / ref_)
    sc_ = sr_ - scr_ * ssr_ * ((atm_ - ref_) / ref_)
    if k < dc_ * (1 + dsm_):
        res = vc_ + dc_ * (2 + dsm_) * (sc_ / 2) + \
            (1 + dsm_) * pc_ * pow(dc_, 2)
    elif dc_ * (1 + dsm_) < k <= dc_:
        res = vc_ - (1 + 1 / dsm_) * pc_ * pow(dc_, 2) - sc_ * dc_ / (2 * dsm_) + (1 + 1 / dsm_) * \
            (2 * pc_ * dc_ + sc_) * k - \
            (pc_ / dsm_ + sc_ / (2 * dc_ * dsm_)) * pow(k, 2)
    elif dc_ < k <= 0:
        res = vc_ + sc_ * k + pc_ * pow(k, 2)
    elif 0 < k <= uc_:
        res = vc_ + sc_ * k + cc_ * pow(k, 2)
    elif uc_ < k <= uc_ * (1 + usm_):
        res = vc_ - (1 + 1 / usm_) * cc_ * pow(uc_, 2) - sc_ * uc_ / (2 * usm_) + (1 + 1 / usm_) * \
            (2 * cc_ * uc_ + sc_) * k - \
            (cc_ / usm_ + sc_ / (2 * uc_ * usm_)) * pow(k, 2)
    elif uc_ * (1 + usm_) < k:
        res = vc_ + uc_ * (2 + usm_) * (sc_ / 2) + \
            (1 + usm_) * cc_ * pow(uc_, 2)
    else:
        raise ValueError("log forward moneyness value input error!")
    return res


def wing_model_loss_function(wing_model_params_list_solve: List[float],
                             wing_model_params_list_input: List[float],
                             moneyness_inputs_list: List[float],
                             mkt_implied_vol_list: List[float],
                             mkt_vega_list: List[float],
                             butterfly_arbitrage_free_cond: 'bool' = True) -> float:
    """
    :param wing_model_params_list_solve: [vr_, sc_, cc_, pc_]
    :param wing_model_params_list_input: [dc_, uc_, dsm_, usm_]
    :param moneyness_inputs_list: [k_1, k_2, k_3, ...]
    :param mkt_implied_vol_list: [vol_1, vol_2, vol_3, ...]
    :param mkt_vega_list: [vega_1, vega_2, vega_3, ...]
    :param butterfly_arbitrage_free_cond: add penality if Durrleman condition is not respected
    :return: wing model calibration error
    """
    max_mkt_vega = max(mkt_vega_list)
    # Mean Squared Error (MSE)
    se = 0
    for i, moneyness_inputs in enumerate(moneyness_inputs_list):
        wing_model_vol = wing_model(k=moneyness_inputs,
                                    vr_=wing_model_params_list_solve[0],
                                    sr_=wing_model_params_list_solve[1],
                                    pc_=wing_model_params_list_solve[2],
                                    cc_=wing_model_params_list_solve[3],
                                    dc_=wing_model_params_list_input[0],
                                    uc_=wing_model_params_list_input[1],
                                    dsm_=wing_model_params_list_input[2],
                                    usm_=wing_model_params_list_input[3])
        se += ((wing_model_vol - mkt_implied_vol_list[i]) * mkt_vega_list[i] / max_mkt_vega) ** 2
    mse = ((se) ** 0.5) / len(moneyness_inputs_list)
    # Butterfly Arbitrage Penality (Durrleman Condition)
    butterfly_arbitrage_penality = 0
    if butterfly_arbitrage_free_cond:
        _, g_list = wing_model_durrleman_condition(vr_=wing_model_params_list_solve[0],
                                                   sr_=wing_model_params_list_solve[1],
                                                   pc_=wing_model_params_list_solve[2],
                                                   cc_=wing_model_params_list_solve[3],
                                                   dc_=wing_model_params_list_input[0],
                                                   uc_=wing_model_params_list_input[1],
                                                   dsm_=wing_model_params_list_input[2],
                                                   usm_=wing_model_params_list_input[3])
        if min(g_list) < 0:
            butterfly_arbitrage_penality = 10e5
    return mse + butterfly_arbitrage_penality


def wing_model_calibrator(wing_model_params_list_input: List[float],
                          moneyness_inputs_list: List[float],
                          mkt_implied_vol_list: List[float],
                          mkt_vega_list: List[float],
                          is_bound_limit: bool = False,
                          epsilon: float = 1e-16) -> float:
    """
    :param wing_model_params_list_input: [dc_, uc_, dsm_, usm_]
    :param moneyness_inputs_list: [k_1, k_2, k_3, ...]
    :param mkt_implied_vol_list: [vol_1, vol_2, vol_3, ...]
    :param mkt_vega_list: [vega_1, vega_2, vega_3, ...]
    :param is_bound_limit: add optimize bound limit if set to True
    :param epsilon: optimize accuracy
    :return: wing model solved params dict
    """
    # Set initial guess for wing_model_params_list_solve
    # wing_model_params_list_guess = [0.124577, random.random(), random.random(), random.random()]
    wing_model_params_list_guess = [random.random(), random.random(),
                                    random.random(), random.random()]
    if is_bound_limit:
        bounds = ([-1e3, 1e3], [-1e3, 1e3], [-1e3, 1e3], [-1e3, 1e3])
        # bounds = ([0.114577, 0.134577], [-1e3, 1e3], [0, 1e3], [0, 1e3])
    else:
        bounds = ([None, None], [None, None], [None, None], [None, None])
    args = (wing_model_params_list_input,
            moneyness_inputs_list,
            mkt_implied_vol_list,
            mkt_vega_list)
    res = optimize.minimize(fun=wing_model_loss_function,
                            x0=wing_model_params_list_guess,
                            args=args,
                            method="SLSQP",
                            bounds=bounds,
                            tol=epsilon)
    # assert res.success
    # print(res.success)
    wing_model_solve = list(res.x)
    res_dict = {"success": res.success,
                "vr_": wing_model_solve[0],
                "sr_": wing_model_solve[1],
                "pc_": wing_model_solve[2],
                "cc_": wing_model_solve[3]}
    print(res_dict)
    return res_dict


if __name__ == "__main__":
    print("ORC wing model calibrator powered by Jincheng Gong.")
