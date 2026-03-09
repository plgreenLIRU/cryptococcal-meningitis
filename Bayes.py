import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.special import logsumexp, gammaln
from samplers.mh import MH

def compute_mu_sigma2(pd, pb):
    """
    Compute the one-step mean multiplier `mu` and variance contribution `sigma2`
    for the model given parameters `pd` (death probability) and `pb` (birth probability).

    Parameters
    - pd: float
        Probability of death per individual (0 <= pd <= 1).
    - pb: float
        Probability of branching per individual (0 <= pb <= 1).

    Returns
    - mu: float
        Expected multiplicative factor for the population in one time step.
    - sigma2: float
        Variance contribution per individual used in the Gaussian approximation.
    """
    mu = (1 - pd) * (1 + pb)
    sigma2 = (1 - pd) * (pb * (1 - pb) + pd * (1 + pb)**2)
    return mu, sigma2


def compute_v(mu, k):
    """
    Compute the factor v(k, mu) = sum_{j=0}^{k-1} mu^{2j+1-k} (used to scale
    the variance across k steps) in a numerically stable way. The closed form
    implemented here handles the special case when `mu` is very close to 1.

    Parameters
    - mu: float
        Per-step multiplicative mean.
    - k: int
        Number of time steps between observations.

    Returns
    - v: float
        Scaling factor for variance over k steps.
    """
    # handle mu close to 1 safely
    if abs(mu - 1) < 1e-10:
        return k
    return mu**(k-1) * (mu**k - 1) / (mu - 1)


def log_likelihood(pd, pb, trajectory, approach='Gaussian'):
    """
    Compute the log-likelihood of a single observed trajectory given
    pd and pb (death and birth probability)

    Two computation approaches are supported:
    - 'Gaussian': uses a Gaussian approximation for transitions over k steps.
    - 'exact': uses an exact discrete summation (currently only for k == 1).

    Parameters
    - pd: float
        Death probability parameter.
    - pb: float
        Branching probability parameter.
    - trajectory: dict-like
        Observation dictionary containing at least `times`, `y` and `log y` (for plotting).
    - approach: str
        Either 'Gaussian' or 'exact'.

    Returns
    - logL: float
        The log-likelihood value for the provided trajectory.
    """

    if approach == 'Gaussian':

        logL = 0.0

        mu, sigma2 = compute_mu_sigma2(pd, pb)

        times = np.array(trajectory['times'])
        y = np.array(trajectory['y'])
        
        for i in range(len(y) - 1):
            
            yi = y[i]
            yi1 = y[i+1]
            k = times[i+1] - times[i]
            
            m = yi * mu**k
            v = compute_v(mu, k)
            s2 = yi * sigma2 * v
            
            # Avoid zeros
            if yi == 0:
                break

            logL += -0.5 * (np.log(2 * np.pi * s2) + (yi1 - m)**2 / s2)

    elif approach == 'exact':

        logL = 0.0
        p0 = pd
        p1 = (1 - pd) * (1 - pb)
        p2 = (1 - pd) * pb

        times = np.array(trajectory['times'])
        y = np.array(trajectory['y'])
        
        for i in range(len(y) - 1):
            
            yi = y[i].astype(int)
            yi1 = y[i+1].astype(int)
            k = times[i+1] - times[i]

            # We've only done the maths for delta t = 1
            if k != 1:
                raise ValueError(f"k must equal 1, but got {k}")

            n2_set = _possible_n2_values(yi=yi, yi1=yi1)

            log_terms = []
            for n2 in n2_set:
                n0 = yi - yi1 + n2
                n1 = yi1 - 2 * n2
                
                log_term = (gammaln(yi + 1) - gammaln(n0 + 1) - gammaln(n1 + 1) - gammaln(n2 + 1) +
                            n0 * np.log(p0) + n1 * np.log(p1) + n2 * np.log(p2))
                log_terms.append(log_term)

            logL += logsumexp(log_terms)

    else:
        raise ValueError("approach needs to be 'Gaussian' or 'exact'")

    return logL


def _possible_n2_values(yi, yi1):
    """
    Return the set of feasible integer values for `n2` given observed counts
    `yi` (current) and `yi1` (next). This enumerates the combinatorial
    possibilities used in the exact multinomial summation of transitions.

    Parameters
    - yi: int or array-like
        Observed count at time i.
    - yi1: int or array-like
        Observed count at time i+1.

    Returns
    - n2_set: ndarray
        Integer values n2 such that 0 <= n2 <= floor(yi1/2) and n2 >= yi1 - yi.
    """
    yi = yi.astype(int)
    yi1 = yi1.astype(int)
    lower =(np.maximum(0, yi1 - yi)).astype(int)
    upper = np.floor(yi1 / 2).astype(int)

    n2_set = np.arange(lower, upper + 1)

    return n2_set


def log_posterior(params, data):
    """
    Unnormalised log-posterior for parameters `params = (pd, pb)` given a
    single trajectory `data`. This uses a uniform prior on the unit square
    (implicitly) by returning -inf for out-of-bounds parameters.

    Parameters
    - params: sequence-like of length 2
        Tuple or list containing (pd, pb).
    - data: dict-like
        Trajectory data passed to `log_likelihood`.

    Returns
    - log_post: float
        The log-posterior (unnormalised).
    """
    pd, pb = params
    if pd < 0 or pd > 1 or pb < 0 or pb > 1:
        return -np.inf
    else:
        return log_likelihood(pd, pb, trajectory=data)


def threshold_line(pd_range):
    """
    Compute the deterministic threshold curve pb = pd / (1 - pd), used
    to e.g. visualise parameter space where branching balances death.

    Parameters
    - pd_range: array-like
        Array of pd values in (0, 1).

    Returns
    - pb_range: ndarray
        Corresponding pb values along the threshold curve.
    """
    pb_range = pd_range / (1 - pd_range)
    return pb_range


def sample_posterior(trajectories, initial_proposal_width=0.1, plot=False):
    """
    Run MCMC sampling (using Metropolis-Hastings) for a list of observed
    trajectories. For each trajectory this function:
    - initialises multiple chains,
    - runs sampling with `samplers.mh.MH`, and
    - produces diagnostic plots of samples and model fits.

    Parameters
    - trajectories: list of dict-like
        Each element should be a trajectory containing keys used by
        `log_likelihood` and for plotting (`times`, `y`, `log y`).

    Returns
    - None
        The function shows plots and prints acceptance rates; it does not
        currently return programmatic results.
    """

    # Initialise MCMC
    mcmc = MH(log_target=log_posterior)

    # This is used to store all of the samples we've got
    global_samples = []

    # Just used for plots
    pd_range = np.linspace(0, 0.999, 100)

    # Loop over trajectories, running MCMC for each
    for i, trajectory in enumerate(trajectories):

        print('Analysing trajectory', i)

        # Define initial parameters
        params0 = []
        n_chains = 10
        for _ in range(n_chains):
            pd_init = 0.5 * np.random.rand()
            pb_init = threshold_line(pd_init)
            params0.append([pd_init, pb_init])

        # Define initial proposal width
        proposal_width = np.copy(initial_proposal_width)

        # Have a few attempts at tuning proposal width
        for _ in range(5):
            samples, ar = mcmc.generate_samples(params_current=params0, data=trajectory, proposal_width=proposal_width,
                                                n_samples=12000, n_chains=n_chains, plot_live=False)

            # Automatically tune acceptance ratio
            av_acceptance_rate = np.mean(ar)
            if av_acceptance_rate < 5:
                proposal_width /= 2
                print('Re-running with proposal width', proposal_width)
            elif av_acceptance_rate > 40:
                proposal_width *= 2
                print('Re-running with proposal width', proposal_width)
            else:
                break

        # Remove burn-in and keep every 100th sample
        burn_in = 2000
        final_samples = np.vstack([arr[burn_in:] for arr in samples])
        final_samples = final_samples[::100]
        global_samples.append(final_samples)

        # Plot samples and model predictions
        if plot:
            fig, ax = plt.subplots(ncols=2)

            # Shade region below threshold green and above threshold red
            pb_range = threshold_line(pd_range)
            pb_clip = np.minimum(pb_range, 1.0)
            ax[1].fill_between(pd_range, 0, pb_clip, color='green', alpha=0.15, zorder=0)
            ax[1].fill_between(pd_range, pb_clip, 1.0, color='red', alpha=0.15, zorder=0)
            ax[1].plot(final_samples[:, 0], final_samples[:, 1], '.', color='blue', alpha=0.2, zorder=3)
            ax[1].plot(pd_range, pb_range, color='black', zorder=4)
            ax[1].grid()
            ax[1].set_xlim([0, 1])
            ax[1].set_ylim([0, 1])
            ax[1].set_xlabel(r'$p_d$')
            ax[1].set_ylabel(r'$p_b$')

            # Plot corresponding model predictions
            for s in final_samples:
                pd, pb = s
                logy_sim, y_sim = utils.run_model(y_init=trajectory['y'][0], pd=pd, pb=pb)
                ax[0].plot(logy_sim, color='blue', alpha=0.3)
            ax[0].plot(trajectory['times'], trajectory['log y'], color='black', marker='o')            
            ax[0].grid()
            ax[0].set_ylim([0, 7])
            ax[0].set_xlabel('Time (hours)')
            ax[0].set_ylabel('log CFU per ml')
            plt.tight_layout()
            plt.show()

    # Concatenate samples
    global_samples = np.vstack([arr for arr in global_samples])

    return global_samples


def main():

    # Load data
    trajectories = utils.load_data(data_type='gold standard')

    # MCMC
    samples = sample_posterior(trajectories, plot=True)

    # Save samples to disk
    '''
    try:
        np.savetxt('population_samples_temp.csv', samples, delimiter=',',
                   header='pd,pb', comments='')
        print(f"Saved population_samples ({samples.shape}) to population_samples_temp.csv")
    except Exception as e:
        print(f"Warning: failed to save samples to CSV: {e}")
    '''

if __name__ == '__main__':
    main()
