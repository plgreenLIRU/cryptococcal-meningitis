import os
from pathlib import Path

from utils import run_model, load_data
from Bayes import sample_posterior, threshold_line
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parent
SIMS_DIR = PROJECT_ROOT / 'sims'


def infer_posterior_and_predict(
    observed_times,
    observed_y,
    population_trajectories,
    predict_to_time=None,
    y_scale='linear',
    modelling_approach='stochastic',
    initial_proposal_width=0.01,
):
    """
    Infer posterior samples for a single patient trajectory and optionally
    generate posterior predictive trajectories up to a requested time.

    Parameters
    - observed_times: array-like
        One or more observed times for a patient.
    - observed_y: array-like
        Observed patient values at `observed_times`.
    - population_trajectories: list
        Population trajectories kept in the API for downstream workflows.
    - predict_to_time: int or None
        If provided, generate predictions from the final observed point up to
        this absolute time.
    - y_scale: str
        Either 'linear' (observed_y are counts) or 'log10' (observed_y are
        log10 counts).
    - modelling_approach: str
        Passed to `run_model` when generating predictive trajectories.
    - initial_proposal_width: float
        Proposal width passed through to `sample_posterior`.

    Returns
    - result: dict
        Contains 'posterior_samples', 'patient_trajectory', and optionally
        'predictive_log_y', 'predictive_y', and 'prediction_times'.
    """
    times = np.asarray(observed_times, dtype=int)
    y_obs = np.asarray(observed_y, dtype=float)

    if times.size != y_obs.size:
        raise ValueError('observed_times and observed_y must have the same length')
    if times.size < 2:
        raise ValueError('Provide at least two observations to infer posterior')
    if np.any(np.diff(times) <= 0):
        raise ValueError('observed_times must be strictly increasing')

    if y_scale == 'linear':
        y_linear = y_obs
        if np.any(y_linear <= 0):
            raise ValueError('observed_y must be positive when y_scale="linear"')
        log_y = np.log10(y_linear)
    elif y_scale == 'log10':
        log_y = y_obs
        y_linear = 10**log_y
    else:
        raise ValueError('y_scale must be "linear" or "log10"')

    patient_trajectory = {
        'times': times,
        'log y': log_y,
        'y': y_linear,
    }

    # `population_trajectories` is accepted for API compatibility with existing
    # workflows; current posterior fitting uses the patient trajectory only.
    _ = population_trajectories

    posterior_samples = sample_posterior(
        [patient_trajectory],
        initial_proposal_width=initial_proposal_width,
        plot=False,
    )

    result = {
        'posterior_samples': posterior_samples,
        'patient_trajectory': patient_trajectory,
    }

    if predict_to_time is not None:
        t_last = int(times[-1])
        t_target = int(predict_to_time)
        if t_target < t_last:
            raise ValueError('predict_to_time must be >= last observed time')

        horizon = t_target - t_last
        prediction_times = np.arange(t_last, t_target + 1)

        predictive_log_y = []
        predictive_y = []
        for pd, pb in posterior_samples:
            logy_sim, y_sim = run_model(
                y_init=y_linear[-1],
                pd=pd,
                pb=pb,
                modelling_approach=modelling_approach,
            )
            predictive_log_y.append(logy_sim[:horizon + 1])
            predictive_y.append(y_sim[:horizon + 1])

        result['prediction_times'] = prediction_times
        result['predictive_log_y'] = np.asarray(predictive_log_y)
        result['predictive_y'] = np.asarray(predictive_y)

    return result


def simulate_noisy_trajectory(y_init, pd, pb, times, noise_sd=0.1):
    logy_full, y_full = run_model(y_init=y_init, pd=pd, pb=pb, modelling_approach='stochastic')
    logy_obs = logy_full[times] + np.random.normal(0, noise_sd, size=len(times))
    return {
        'times': times,
        'log y': logy_obs,
        'y': 10**logy_obs,
    }


def save_posterior_samples(samples, filename):
    samples_array = np.asarray(samples, dtype=float)
    output_path = Path(filename)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, samples_array, delimiter=',', header='pd,pb', comments='')
    print(f"Saved posterior samples ({samples_array.shape}) to {output_path}")


def save_posterior_samples_by_observation_count(
    observed_times,
    observed_y,
    population_trajectories,
    output_dir='sims',
    prefix='posterior',
    initial_proposal_width=0.01,
):
    """Save posterior samples for all partial trajectories from 2 points onward."""
    times = np.asarray(observed_times, dtype=int)
    y = np.asarray(observed_y, dtype=float)
    if times.size != y.size:
        raise ValueError('observed_times and observed_y must have the same length')
    if times.size < 2:
        raise ValueError('Provide at least two observations to infer posterior')

    created_files = []
    for n_points in range(2, times.size + 1):
        result = infer_posterior_and_predict(
            observed_times=times[:n_points],
            observed_y=y[:n_points],
            population_trajectories=population_trajectories,
            y_scale='linear',
            initial_proposal_width=initial_proposal_width,
        )
        filename = f'{prefix}_{n_points}_points.csv'
        output_path = os.path.join(output_dir, filename)
        save_posterior_samples(result['posterior_samples'], output_path)
        created_files.append(Path(output_path))

    return created_files


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
    inference = infer_posterior_and_predict(
        observed_times=new_trajectory['times'],
        observed_y=new_trajectory['y'],
        population_trajectories=trajectories,
        y_scale='linear',
        initial_proposal_width=0.01,
    )
    samples = inference['posterior_samples']

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

    np.random.seed(42)

    # Load gold-standard data    
    trajectories = load_data(data_type='gold standard')

    # Create declining synthetic trajectory from the model
    # new_times = np.array([0, 24, 48, 72, 96])
    new_times = np.array([0, 72, 144, 216, 288])
    new_trajectory_full = simulate_noisy_trajectory(
        y_init=10**5,
        pd=0.2,
        pb=0.1875,
        times=new_times,
    )

    # Gradually introduce data, doing Bayesian inference after each LP
    for i in range(2, len(new_times)+1):
        new_trajectory = {
            'times': new_trajectory_full['times'][:i],
            'log y': new_trajectory_full['log y'][:i],
            'y': new_trajectory_full['y'][:i],
        }
        predict_new_trajectory(new_trajectory=new_trajectory, trajectories=trajectories)

    final_declining_trajectory = {
        'times': new_trajectory_full['times'],
        'log y': new_trajectory_full['log y'],
        'y': new_trajectory_full['y'],
    }
    final_declining_inference = infer_posterior_and_predict(
        observed_times=final_declining_trajectory['times'],
        observed_y=final_declining_trajectory['y'],
        population_trajectories=trajectories,
        y_scale='linear',
        initial_proposal_width=0.01,
    )
    save_posterior_samples(
        final_declining_inference['posterior_samples'],
        SIMS_DIR / 'declining_example_posterior.csv',
    )
    save_posterior_samples_by_observation_count(
        observed_times=final_declining_trajectory['times'],
        observed_y=final_declining_trajectory['y'],
        population_trajectories=trajectories,
        output_dir='sims',
        prefix='declining_example_posterior',
        initial_proposal_width=0.01,
    )

    # Create a growth trajectory from model parameters with crypto growth over time
    growth_pd = 0.2
    growth_pb = 0.25
    # growth_times = np.array([0, 48, 96, 144, 192])  # 5 LPs -> 4 panels
    # growth_times = np.array([0, 168, 336])
    growth_times = np.array([0, 72, 144, 216, 288])
    growth_trajectory_full = simulate_noisy_trajectory(
        y_init=10**2,
        pd=growth_pd,
        pb=growth_pb,
        times=growth_times,
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

    final_growth_trajectory = {
        'times': growth_trajectory_full['times'],
        'log y': growth_trajectory_full['log y'],
        'y': growth_trajectory_full['y'],
    }
    final_growth_inference = infer_posterior_and_predict(
        observed_times=final_growth_trajectory['times'],
        observed_y=final_growth_trajectory['y'],
        population_trajectories=trajectories,
        y_scale='linear',
        initial_proposal_width=0.01,
    )
    save_posterior_samples(
        final_growth_inference['posterior_samples'],
        SIMS_DIR / 'growth_example_posterior.csv',
    )
    save_posterior_samples_by_observation_count(
        observed_times=final_growth_trajectory['times'],
        observed_y=final_growth_trajectory['y'],
        population_trajectories=trajectories,
        output_dir='sims',
        prefix='growth_example_posterior',
        initial_proposal_width=0.01,
    )

    plt.show()
