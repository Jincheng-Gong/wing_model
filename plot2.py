'''
@Time: 2024/4/17 14:23 PM
@Author: 龚锦程
@Contact: 广发商贸/做市业务部/Jincheng.Gong@hotmail.com
@File: plot.py
@Desc: ORC Wing Model Solution Space Plot
'''

import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    wing_sim_args = pd.read_csv("wing_sim_args.csv", header=0)
    wing_sim_args_true = wing_sim_args[wing_sim_args['wing_model_combined_test']]

    sr_unique = wing_sim_args_true['sr_'].unique()
    sol_space = []

    for i in sr_unique:
        sol_space_i = wing_sim_args_true[wing_sim_args_true['sr_'] == i]
        sol_space.append(sol_space_i[sol_space_i['cc_'] == sol_space_i['cc_'].max()])
        sol_space.append(sol_space_i[sol_space_i['cc_'] == sol_space_i['cc_'].min()])
        sol_space.append(sol_space_i[sol_space_i['pc_'] == sol_space_i['pc_'].max()])
        sol_space.append(sol_space_i[sol_space_i['pc_'] == sol_space_i['pc_'].min()])
    sol_space = pd.concat(sol_space)
    sol_space_positive = sol_space[sol_space['sr_'] >= 0]
    sol_space_negative = sol_space[sol_space['sr_'] <= 0]
    # * For solution space param test.
    # sol_space_negative['cc2_'] = sol_space_negative['pc_']
    # sol_space_negative['pc2_'] = sol_space_negative['cc_']
    # sol_space_negative['sr2_'] = -1 * sol_space_negative['sr_']
    # sol_space_negative.to_csv('negative.csv', index=False)
    # sol_space_positive.to_csv('positive.csv', index=False)

    fig = plt.figure(1)
    ax1 = plt.subplot(2, 1, 1, projection='3d')
    ax1.scatter(sol_space_positive['cc_'],
                sol_space_positive['pc_'],
                sol_space_positive['sr_'],
                c='red',
                edgecolors='black',
                linewidths=0.5)
    ax1.set_xlabel("cc")
    ax1.set_ylabel("pc")
    ax1.set_zlabel("sc")
    ax2 = plt.subplot(2, 1, 2, projection='3d')
    ax2.scatter(sol_space_negative['cc_'],
                sol_space_negative['pc_'],
                sol_space_negative['sr_'],
                c='red',
                edgecolors='black',
                linewidths=0.5)
    ax2.set_xlabel("cc")
    ax2.set_ylabel("pc")
    ax2.set_zlabel("sc")
    plt.show()
