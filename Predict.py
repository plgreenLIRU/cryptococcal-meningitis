from utils import run_model, load_data
from Bayes import sample_posterior, threshold_line
import numpy as np
from matplotlib import pyplot as plt


def predict_new_trajectory(new_trajectory, trajectories):

    # Initialise plot
    fig, ax = plt.subplots(ncols=2)
    ax[0].set_xlim([0, 336])
    ax[0].grid()
    ax[0].set_xlabel('Time (hours)')
    ax[0].set_ylabel('log CFU per ml')

    # Plot gold-standard
    for trajectory in trajectories:
        ax[0].plot(trajectory['times'], trajectory['log y'], color='black', alpha=0.1)

    # Plot data point from new trajectory
    ax[0].plot(new_trajectory['times'], new_trajectory['log y'], marker='o', color='red')

    # Infer parameters for new trajectory
    samples = sample_posterior([new_trajectory], initial_proposal_width=0.01, plot=False)

    # Plot model predictions
    for s in samples:
        pd, pb = s
        logy_sim, y_sim = run_model(y_init=new_trajectory['y'][-1], pd=pd, pb=pb)
        ax[0].plot(np.arange(new_trajectory['times'][-1], new_trajectory['times'][-1] + len(logy_sim)), logy_sim, color='blue', alpha=0.1)

    # Plot MCMC samples
    pd_range = np.linspace(0, 0.999, 100)
    pb_range = threshold_line(pd_range)
    pb_clip = np.minimum(pb_range, 1.0)
    ax[1].fill_between(pd_range, 0, pb_clip, color='green', alpha=0.15, zorder=0)
    ax[1].fill_between(pd_range, pb_clip, 1.0, color='red', alpha=0.15, zorder=0)
    ax[1].plot(samples[:, 0], samples[:, 1], '.', color='blue', alpha=0.2, zorder=3)
    ax[1].plot(pd_range, pb_range, color='black', zorder=4)
    ax[1].grid()
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[1].set_xlabel(r'$p_d$')
    ax[1].set_ylabel(r'$p_b$')    
    ax[1].grid()

    plt.tight_layout()


if __name__ == '__main__':

    # Load gold-standard data    
    trajectories = load_data(data_type='gold standard')

    # Create new trajectory
    new_times = np.array([0, 24, 36, 48, 72])
    new_logy = np.array([5, 4.5, 4, 3.5, 3])

    # Gradually introduce data, doing Bayesian inference after each LP
    for i in range(2, len(new_times)+1):
        new_y = 10**new_logy[:i]
        new_trajectory = {'times': new_times[:i], 'log y': new_logy[:i], 'y': new_y[:i]}
        predict_new_trajectory(new_trajectory=new_trajectory, trajectories=trajectories)

    plt.show()
