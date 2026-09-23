from pathlib import Path

import numpy as np
import pandas


DATA_DIR = Path(__file__).resolve().parent / 'data'


def _stochastic_transition_count(y_current, pd, pb):
    """
    Sample the next population count using a multinomial draw over the three
    possible per-individual outcomes.

    This is equivalent to sampling each individual separately and summing the
    outcomes, but it is much faster for large populations.
    """
    counts = np.random.multinomial(
        n=int(y_current),
        pvals=(pd, (1 - pd) * (1 - pb), (1 - pd) * pb),
    )
    return counts[1] + 2 * counts[2]


def load_data(data_type : str):
    """
    type is either 'gold standard' or 'animal'
    """

    # Load raw dataset
    df = pandas.read_csv(DATA_DIR / 'Arm2.csv', header=1)

    # Remove OUT == "."
    df = df[df["OUT"] != "."]

    # Convert remaining OUT values to be numeric data type
    df['OUT'] = pandas.to_numeric(df['OUT'])

    # Convert #ID to numeric in case it's read as string
    df["#ID"] = pandas.to_numeric(df["#ID"], errors="coerce")

    # Get rid of empty rows
    df = df[df["#ID"] > 0].copy()

    # Rows where #ID <= 18 are rabbit data
    if data_type == 'gold standard':
        df = df[df["#ID"] > 18].copy()
    elif data_type == 'animal':
        df = df[df["#ID"] <= 18].copy()
    else:
        print("Didn't recognise data type")  # should make this an error message

    # Extract unique IDs
    unique_ids = df["#ID"].unique().tolist()

    # Assemble data into dictionary
    trajectories = []
    for id in unique_ids:

        times = df.loc[df["#ID"] == id, "TIME"].astype(int).to_numpy()
        log_y = df.loc[df["#ID"] == id, "OUT"].values

        # Only process if the initial fungal burden is greater than zero
        if log_y[0] == 0:
            pass
        else:
            if len(times) > 1:
                trajectories.append({'log y': log_y, 'times': times, 'y': 10**log_y, 'data type': data_type})

    return trajectories


def run_model(y_init, pd, pb, modelling_approach='stochastic'):
    """
    Run a model that predicts a single trajectory, starting from initial population y_init.

    modelling_approach can be 'hybrid', 'stochastic', or 'linear'
    """

    # Compute gradient needed for linear model
    mu = (1 - pd) * (1 + pb)

    # Initial conditions
    y_sim = [y_init]

    # Threshold for hybrid model
    switching_point = 1000

    if modelling_approach == 'hybrid':
        for k in np.arange(1, 337):
            if y_sim[-1] > switching_point and modelling_approach == 'hybrid':
                y_sim.append(y_sim[-1] * mu)
            else:
                if y_sim[-1] < 0.1:
                    y_sim.append(0)
                else:
                    y_sim.append(_stochastic_transition_count(y_sim[-1], pd, pb))
    elif modelling_approach == 'stochastic':
        for k in np.arange(1, 337):
            y_sim.append(_stochastic_transition_count(y_sim[-1], pd, pb))
    elif modelling_approach == 'linear':
        for k in np.arange(1, 337):
            y_sim.append(y_sim[-1] * mu)
    else:
        print("no modelling approach specified")

    
    y_sim = np.array(y_sim)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_y_sim = np.log10(y_sim)

    # Replace -infs with zeros
    indx = log_y_sim < 0
    log_y_sim[indx] = 0

    return log_y_sim, y_sim
