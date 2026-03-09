from utils import run_model, load_data
import numpy as np
from matplotlib import pyplot as plt
import Bayes
from samplers.mh import MH

def test_load_data(plot=False):
    trajectories = load_data(data_type='gold standard')

    if plot:
        fig, ax = plt.subplots()
        for trajectory in trajectories:
            ax.plot(trajectory['times'], trajectory['log y'], color='black', alpha=0.3)
        ax.grid()
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('log CFU per ml')
        plt.show()
    print("Passed test_load_data")


def test_run_model(plot=False):
    """
    Test our modelling approach, specifically comparing the response of the linear, stochastic,
    and hybrid models. The stochastic is the 'correct' appraoch though is computationally inefficient
    at high populations while the linear model should be a close approximation at high populations.
    The hybrid approach switches from linear to stochastic when the population drops below a pre-defined
    level.
    """

    np.random.seed(42)

    # Simulation parameters for test
    y_init = 200000  # Initial population
    pd = 0.14  # Fix death prob.
    pb = 0.1  # Fix birth prob.

    # Rune model using the 3 approaches
    log_y_sim_linear = run_model(y_init=y_init, pd=pd, pb=pb, modelling_approach='linear')[0]
    log_y_sim_stocha = run_model(y_init=y_init, pd=pd, pb=pb, modelling_approach='stochastic')[0]
    log_y_sim_hybrid = run_model(y_init=y_init, pd=pd, pb=pb, modelling_approach='hybrid')[0]

    # Test that the linear and hyrbid models are the same, above the treshold
    indx = np.logical_and(log_y_sim_linear > 3, log_y_sim_hybrid > 3)
    assert np.allclose(log_y_sim_linear[indx], log_y_sim_hybrid[indx])

    # Test that the hybrid and stochastic models are close, above the treshold
    indx = np.logical_and(log_y_sim_hybrid > 3, log_y_sim_stocha > 3)
    assert np.allclose(log_y_sim_hybrid[indx], log_y_sim_stocha[indx], atol=0.05)

    # Plots
    if plot:
        fig, ax = plt.subplots()
        ax.plot(log_y_sim_linear, linewidth=3, alpha=0.5, label='Linear model')
        ax.plot(log_y_sim_stocha, linewidth=3, alpha=0.5, label='Stochastic model')
        ax.plot(log_y_sim_hybrid, linewidth=3, alpha=0.5, label='Hybrid model')    
        ax.plot([0, 336], [3, 3], color='black', label='Hyrbid model switching point')
        ax.legend()
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('log CFU per ml')
        ax.grid()
        plt.show()

    print('test_run_model passed')


def test_large_population(plot=False):
    """
    For high populations the variance of our model should, in the log-space
    be very small. The 'process noise' will then become more significant for
    lower populations.
    """

    # Simulation parameters
    y_init = 200000  # Initial population
    pd = 0.14  # Death prob.
    pb = 0.1  # Birth prob.

    # Simulate 10 trajectories from the same initial conditions and parameters
    all_log_y_sim = []
    if plot:
        fig, ax = plt.subplots()
    for i in range(10):
        log_y_sim_stocha = run_model(y_init=y_init, pd=pd, pb=pb, modelling_approach='stochastic')[0]
        all_log_y_sim.append(log_y_sim_stocha)
        if plot:
            ax.plot(log_y_sim_stocha, color='black', alpha=0.2)
    if plot:
        ax.grid()
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('log CFU per ml')
        plt.show()

    # Check that the variance grows as expected (until extinction)
    var = np.var(np.array(all_log_y_sim), axis=0)
    assert var[150] > var[0]

    print('test_large_population passed')


def test_gaussian_approx(plot=False):
    """
    Test the approximation that Y_{t+k} | Y_t=y_t = N(y_t \mu^k, y_t, v, \sigma^2)
    """
    
    np.random.seed(42)

    # Simulation parameters
    y_init = 10000  # Initial population
    pd = 0.14  # Death prob.
    pb = 0.1  # Birth prob.

    # Run ensemble of simulations
    N_runs = 1000
    y_sim_samples = []
    for _ in range(N_runs):
        log_y_sim, y_sim = run_model(y_init=y_init, pd=pd, pb=pb, modelling_approach='stochastic')
        y_sim_samples.append(y_sim)

    # Estimate moments from simulation runs
    y_sim_samples = np.array(y_sim_samples)
    mu, sigma2 = Bayes.compute_mu_sigma2(pd=pd, pb=pb)

    # Compute moments using Gaussian approximation
    gaussian_mean = []
    gaussian_var = []
    for k in range(0, 337):
        v = Bayes.compute_v(mu=mu, k=k)
        gaussian_mean.append(y_init * mu ** k)
        gaussian_var.append(y_init * v * sigma2)

    # Compare Monte-Carlo vs. Gaussian approx.
    mean_mc = np.mean(y_sim_samples, axis=0)
    var_mc = np.var(y_sim_samples, axis=0)
    assert np.allclose(mean_mc, gaussian_mean, atol=5)
    assert np.allclose(var_mc, gaussian_var, atol=600)

    # Plots
    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(10, 6))

        mean_diff = gaussian_mean - mean_mc
        var_diff = gaussian_var - var_mc

        # Means
        ax[0, 0].plot(mean_mc, 'k', label='Monte Carlo')
        ax[0, 0].plot(gaussian_mean, 'r--', label='Gaussian approximation')
        ax[0, 0].set_xlabel('Time (hours)')
        ax[0, 0].set_title('Mean comparison')
        ax[0, 0].legend()
        ax[0, 0].set_ylabel('Fungal burden')
        ax[0, 0].grid()

        # Mean difference
        ax[1, 0].plot(mean_diff, 'b')
        ax[1, 0].set_title('Mean difference (Gaussian − MC)')
        ax[1, 0].grid()
        ax[1, 0].set_xlabel('Time (hours)')

        # Variances
        ax[0, 1].plot(var_mc, 'k')
        ax[0, 1].plot(gaussian_var, 'r--')
        ax[0, 1].set_title('Variance comparison')
        ax[0, 1].grid()
        ax[0, 1].set_xlabel('Time (hours)')
        ax[0, 1].set_ylabel('Fungal burden')

        # Variance difference
        ax[1, 1].plot(var_diff, 'b')
        ax[1, 1].set_title('Variance difference (Gaussian − MC)')
        ax[1, 1].set_xlabel('Time (hours)')
        ax[1, 1].grid()

        plt.tight_layout()
        
        fig, ax = plt.subplots()
        ax.plot(np.log10(y_sim_samples.T), color='black', alpha=0.2)
        ax.plot(np.log10(gaussian_mean), color='red', label='Mean (Gaussian approximation)')
        ax.plot(np.log10(gaussian_mean + 3 * np.sqrt(gaussian_var)), color='red', linestyle='--', label='$\pm$ 3 Standard deviation (Gaussian approximation)')
        ax.plot(np.log10(gaussian_mean - 3 * np.sqrt(gaussian_var)), color='red', linestyle='--')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('log CFU per ml')
        ax.set_ylim([0, np.log10(y_init)])
        ax.legend()
        ax.grid()
        
        plt.show()

    print('test_gaussian_approx passed')


def test_likelihood(plot=False):

    np.random.seed(42)

    # Simulation parameters
    y_init = 10000  # Initial population
    
             # pd, pb
    params = [0.4, 0.6]

    # Run simulation to create data
    pd_true = 0.4
    pb_true = 0.6
    log_y_sim, y_sim = run_model(y_init=y_init, pd=pd_true, pb=pb_true, modelling_approach='hybrid')
    trajectory = {'times' : np.arange(0, 337), 'log y' : log_y_sim, 'y' : y_sim}

    # Initialise plot if we're using it
    if plot:
        fig, ax = plt.subplots(ncols=3)
        ax[0].plot(trajectory['times'], trajectory['log y'], color='black')
        ax[0].set_ylabel('Log fungal burden')
        ax[0].set_xlabel('Time (hours)')
        ax[0].grid()
        ax[0].set_title('(a)')

    # Vary pd over range and compare likelihood evaluation techniques
    logp_gaussian = []
    logp_exact = []
    pd_range = np.linspace(0.3, 0.5, 10)
    for pd in pd_range:
        logp_gaussian.append(Bayes.log_likelihood(pd=pd, pb=pb_true, trajectory=trajectory, approach='Gaussian'))
        logp_exact.append(Bayes.log_likelihood(pd=pd, pb=pb_true, trajectory=trajectory, approach='exact'))

    # Test the 2 likelihood evaluation techniques are close
    assert np.allclose(logp_gaussian, logp_exact, rtol=0.05)

    # Plot results varying pd
    if plot:        
        ax[1].plot(pd_range, logp_gaussian, color='black', label='Gaussian approximation')
        ax[1].plot(pd_range, logp_exact, color='red', linestyle='--', label='Exact')
        ax[1].grid()
        ax[1].set_xlabel(r'$p_d$')
        ax[1].set_ylabel('log likelihood')
        ax[1].set_title('(b)')

    # Vary pb over range and compare likelihood evaluation techniques
    logp_gaussian = []
    logp_exact = []
    pb_range = np.linspace(0.5, 0.7, 10)
    for pb in pb_range:
        logp_gaussian.append(Bayes.log_likelihood(pd=pd_true, pb=pb, trajectory=trajectory, approach='Gaussian'))
        logp_exact.append(Bayes.log_likelihood(pd=pd_true, pb=pb, trajectory=trajectory, approach='exact'))

    # Test the 2 likelihood evaluation techniques are close
    assert np.allclose(logp_gaussian, logp_exact, rtol=0.05)

    # Plot results varying pd
    if plot:        
        ax[2].plot(pb_range, logp_gaussian, color='black', label='Gaussian approximation')
        ax[2].plot(pb_range, logp_exact, color='red', linestyle='--', label='Exact')
        ax[2].legend()
        ax[2].grid()
        ax[2].set_xlabel(r'$p_b$')
        ax[2].set_ylabel('log likelihood')
        ax[2].set_title('(c)')
        plt.tight_layout()
        plt.show()

    print('test_likelihood passed')


def test_sample_posterior(plot=False):

    np.random.seed(42)

    # Run simulation to create data
    y_init = 25000  # Initial population
    pd_true = 0.5
    pb_true = 0.95
    log_y_sim, y_sim = run_model(y_init=y_init, pd=pd_true, pb=pb_true, modelling_approach='hybrid')
    full_trajectory = {'times' : np.arange(0, 337), 'log y' : log_y_sim, 'y' : y_sim}
    trajectory = {'times' : np.arange(0, 337)[::166], 'log y' : log_y_sim[::166], 'y' : y_sim[::166]}

    if plot:
        fig, ax = plt.subplots()
        ax.plot(full_trajectory['times'], full_trajectory['log y'], color='black', alpha=0.3, label='Measured')
        ax.plot(trajectory['times'], trajectory['log y'], 'o', color='black', label='Observations')
        ax.legend()
        ax.grid()

    # Run MCMC
    samples = Bayes.sample_posterior(trajectories=[trajectory], initial_proposal_width=0.005, plot=False)

    # Model predictions using MCMC samples
    all_logy_sim = []
    for s in samples:
        pd, pb = s
        logy_sim, y_sim = run_model(y_init=trajectory['y'][0], pd=pd, pb=pb)
        all_logy_sim.append(logy_sim)

    # Compute moments of predictions
    all_logy_sim_arr = np.vstack([arr for arr in all_logy_sim])
    mean_prediction = np.mean(all_logy_sim_arr, axis=0)
    std_prediction = np.std(all_logy_sim_arr, axis=0)

    if plot:

        # Plot samples
        fig, ax = plt.subplots()
        ax.plot(samples[:, 0], samples[:, 1], '.', color='black', alpha=0.3)
        ax.plot(pd_true, pb_true, 'o', color='red')
        ax.grid()

        # Plot corresponding model predictions
        fig, ax = plt.subplots()
        for logy_sim in all_logy_sim:
            ax.plot(logy_sim, color='red', alpha=0.3)
        ax.plot(full_trajectory['times'], full_trajectory['log y'], color='black')
        ax.plot(mean_prediction, color='blue', label='Predictions')
        ax.plot(mean_prediction + 3 * std_prediction, color='blue')
        ax.plot(mean_prediction - 3 * std_prediction, color='blue')
        ax.plot(trajectory['times'], trajectory['log y'], color='black', marker='o')            
        ax.grid()
        ax.legend()
        ax.set_ylim([0, 7])
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Log Fungal Burden')
        plt.tight_layout()

    assert np.allclose(mean_prediction, log_y_sim, atol=2)
    assert np.all(mean_prediction + 3 * std_prediction > log_y_sim)
    assert np.all(mean_prediction - 3 * std_prediction < log_y_sim)

    print('test_sample_posterior passed')


if __name__ == '__main__':
    #test_load_data(plot=False)
    #test_run_model(plot=True)
    #test_large_population(plot=True)
    #test_gaussian_approx(plot=True)
    #test_likelihood(plot=True)
    #test_sample_posterior(plot=True)
    plt.show()