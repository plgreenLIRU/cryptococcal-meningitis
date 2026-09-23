import numpy as np

from Predict import save_posterior_samples_by_observation_count


def test_save_posterior_samples_by_observation_count(tmp_path):
    times = np.array([0, 24, 48, 72])
    y = np.array([1e5, 1e4, 1e3, 1e2])

    created = save_posterior_samples_by_observation_count(
        observed_times=times,
        observed_y=y,
        population_trajectories=[],
        output_dir=tmp_path,
        prefix='patient',
        initial_proposal_width=0.01,
    )

    assert len(created) == len(times) - 1
    for path in created:
        assert path.exists()
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        assert data.shape[1] == 2
