from utils import run_model, load_data
from Bayes import sample_posterior, threshold_line
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def simulate_noisy_trajectory(y_init, pd, pb, times, rng, noise_sd=0.1):
    logy_full, y_full = run_model(y_init=y_init, pd=pd, pb=pb, modelling_approach='stochastic')
    logy_obs = logy_full[times] + rng.normal(0, noise_sd, size=len(times))
    return {
        'times': times,
        'log y': logy_obs,
        'y': 10**logy_obs,
    }


def predict_new_trajectory(new_trajectory, trajectories, title=None):

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

    # Fit linear regression model
    lr = LinearRegression()
    X = new_trajectory['times'].reshape(-1, 1)
    y = new_trajectory['log y']
    lr.fit(X, y)

    # Infer parameters for new trajectory
    samples = sample_posterior([new_trajectory], initial_proposal_width=0.01, plot=False)

    # Plot model predictions
    for s in samples:
        pd, pb = s
        logy_sim, y_sim = run_model(y_init=new_trajectory['y'][-1], pd=pd, pb=pb)
        ax[0].plot(np.arange(new_trajectory['times'][-1], new_trajectory['times'][-1] + len(logy_sim)), logy_sim, color='blue', alpha=0.1)
    
    # Plot linear regression predictions
    X_star = (np.arange(new_trajectory['times'][-1], new_trajectory['times'][-1] + len(logy_sim))).reshape(-1, 1)
    ax[0].plot(X_star, lr.predict(X_star), '--', color='yellow', linewidth=3, label='Linear Regression')
    ax[0].set_ylim([0, 8])

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

    rng = np.random.default_rng(42)

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

    # Create a growth trajectory from model parameters with crypto growth over time
    growth_pd = 0.2
    growth_pb = 0.28
    growth_times = np.array([0, 48, 96, 144, 192])  # 5 LPs -> 4 panels
    growth_trajectory_full = simulate_noisy_trajectory(
        y_init=10**2,
        pd=growth_pd,
        pb=growth_pb,
        times=growth_times,
        rng=rng,
    )

    # Gradually introduce data, doing Bayesian inference after each LP
    for i in range(2, len(growth_times) + 1):
        growth_trajectory = {
            'times': growth_trajectory_full['times'][:i],
            'log y': growth_trajectory_full['log y'][:i],
            'y': growth_trajectory_full['y'][:i],
        }
        predict_new_trajectory(
            new_trajectory=growth_trajectory,
            trajectories=trajectories
        )

    plt.show()
