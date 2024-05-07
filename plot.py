'''
@Time: 2024/4/17 11:10 AM
@Author: 龚锦程
@Contact: 广发商贸/做市业务部/Jincheng.Gong@hotmail.com
@File: plot.py
@Desc: ORC Wing Model Solution Space Plot
'''

import warnings

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    wing_sim_args = pd.read_csv("wing_sim_args.csv", header=0)
    wing_sim_args_true = wing_sim_args[wing_sim_args['wing_model_combined_test']]

    sr_unique = wing_sim_args_true['sr_'].unique()
    sol_space = pd.DataFrame(columns=wing_sim_args_true.columns)
    for i in sr_unique:
        sol_space_i = wing_sim_args_true[wing_sim_args_true['sr_'] == i]
        sol_space = pd.concat([sol_space,
                            sol_space_i[sol_space_i['cc_'] == sol_space_i['cc_'].max()],
                            sol_space_i[sol_space_i['cc_'] == sol_space_i['cc_'].min()],
                            sol_space_i[sol_space_i['pc_'] == sol_space_i['pc_'].max()],
                            sol_space_i[sol_space_i['pc_'] == sol_space_i['pc_'].min()]])

    ax = plt.subplot(projection='3d')
    ax.scatter(sol_space['cc_'],
            sol_space['pc_'],
            sol_space['sr_'],
            c='red',
            edgecolors='black',
            linewidths=0.5)
    plt.show()
