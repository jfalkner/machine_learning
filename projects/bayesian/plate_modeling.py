import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def load_data():
    """Create fake data that represents several runs of a 96-well plate assay.

    Normally, this would be exported from a LIMS or similar. For the sake of the example, a fake
    data set is created. This data is made with known effects that will hopefully be learned by the
    model. e.g. telling us what runs, tech(s) and reagent(s) are abnormal.
    """
    # Each assay run consists of four 96-well plates. Each plate is prepared by one tech using one
    # set of reagents; however, the four plates in an run may be any combo of techs and reagents.
    plates_per_run = 4
    wells_per_plate = 96

    # Run 0, 1 and 2 all are pretty good. 100 is our made-up, expected result.
    # Run 3 is substantially worse (lower mean, higher noise). We'll assume instrument issues.
    run_mu = {'Run_0': 100, 'Run_1': 102, 'Run_2': 98, 'Run_3': 70}
    run_sigma = {'Run_0': 5, 'Run_1': 5, 'Run_2': 5, 'Run_3': 15}
    runs = run_mu.keys()

    # Reagent lots A and B are good. No abnormal effect on the assay.
    # Reagent lot C has a negative bias and adds more noise.
    reagent_mu_effect = {'A': 0, 'B': 0, 'C': -10}
    reagent_sigma_effect = {'A': 1, 'B': 1, 'C': 8}
    reagents = list(reagent_mu_effect.keys())

    # Jane is the senior tech and runs the assay without any negative effect.
    # Bob is still learning. Whatever he is doing, it has negative bias and adds noise.
    tech_mu_effect = {'Jane': 0, 'Bob': -15}
    tech_sigma_effect = {'Jane': 1, 'Bob': 2}
    technicians = list(tech_mu_effect.keys())

    # Make all the plates for each run. Randomly picking reagent lots and techs.
    data = []
    for run in runs:
        for p_idx in range(plates_per_run):
            reagent = np.random.choice(reagents)
            tech = np.random.choice(technicians)

            # Combine means and variances (sum of variances for independent sources)
            total_mean = run_mu[run] + reagent_mu_effect[reagent] + tech_mu_effect[tech]
            total_sigma = np.sqrt(run_sigma[run]**2 + reagent_sigma_effect[reagent]**2 + tech_sigma_effect[tech]**2)

            plate_results = np.random.normal(total_mean, total_sigma, wells_per_plate)

            for well_val in plate_results:
                data.append({
                    'run': run,
                    'reagent': reagent,
                    'technician': tech,
                    # The value is whatever the assay measured. Assume 100 means a good result for a
                    # control, and we're measuring plates full of control samples.
                    'value': well_val
                })

    return pd.DataFrame(data)

def model_data(df):
    """Create a Bayesian model that will fit the assay data.

    This model doesn't know anything about the effects that load_data() used. All the model knows
    about are the columns of our data set: runs, reagents and techs. These three things are assumed
    to be normal distributions that will be fit to best match the observed data.
    """
    runs = sorted(df['run'].unique())
    reagents = sorted(df['reagent'].unique())
    technicians = sorted(df['technician'].unique())

    # 2. Bayesian Modeling
    with pm.Model(coords={"run": runs, "reagent": reagents, "tech": technicians}) as model:
        # Indices
        run_idx = pd.Categorical(df['run'], categories=runs).codes
        reagent_idx = pd.Categorical(df['reagent'], categories=reagents).codes
        tech_idx = pd.Categorical(df['technician'], categories=technicians).codes

        # Priors for Run effects (Absolute)
        # Run mu starts at 100 since that is the expected output of a good assay
        mu_run = pm.Normal('mu_run', mu=100, sigma=20, dims="run")
        sigma_run = pm.Exponential('sigma_run', lam=0.1, dims="run")

        # Priors for Reagent effects (Offsets)
        # Reagent: mu starts at 0 because we assume that all reagent lots are good.
        mu_reagent = pm.Normal('mu_reagent', mu=0, sigma=10, dims="reagent")
        sigma_reagent = pm.Exponential('sigma_reagent', lam=0.1, dims="reagent")

        # Priors for Technician effects (Offsets)
        # Tech mu starts at 0 because we assume all trained techs will run the assay correctly.
        mu_tech = pm.Normal('mu_tech', mu=0, sigma=10, dims="tech")
        sigma_tech = pm.Exponential('sigma_tech', lam=0.1, dims="tech")

        # Expected value and noise
        mu = mu_run[run_idx] + mu_reagent[reagent_idx] + mu_tech[tech_idx]

        # Combine noise components (Sum of variances)
        var_total = (sigma_run[run_idx]**2 +
                     sigma_reagent[reagent_idx]**2 +
                     sigma_tech[tech_idx]**2)
        sigma = pm.Deterministic('sigma', pm.math.sqrt(var_total))

        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=df['value'])

        # Sampling
        return pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True)

def print_model_and_make_plot(trace):
    # Prints out a tabular form of the fitted model
    print("Model Summary:")
    print(az.summary(trace, var_names=['mu_run', 'mu_reagent', 'mu_tech']))

    # Create a nice plot showing the model. Save it to a png file and also display it.
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    az.plot_forest(trace, var_names=['mu_run', 'mu_reagent', 'mu_tech'], ax=axes[0])
    axes[0].set_title("Mean Effects (Run, Reagent, Tech)")
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5, label='Zero Offset')
    axes[0].axvline(100, color='green', linestyle='--', alpha=0.5, label='Expected Baseline')
    axes[0].legend()

    az.plot_forest(trace, var_names=['sigma_run', 'sigma_reagent', 'sigma_tech'], ax=axes[1])
    axes[1].set_title("Variability Effects (Standard Deviations)")
    axes[1].axvline(0, color='black', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig("bayesian_model.png")
    print("Plot saved to bayesian_model.png")
    plt.show()

def run_example():
    # Make this example deterministic
    np.random.seed(42)

    # 1. Load an export of data from several experiments.
    df = load_data()

    # 2. Fit a Bayesian model to the data.
    trace = model_data(df)

    # 3. Print out the fitted model and save/show a plot of the fitted model.
    print_model_and_make_plot(trace)

if __name__ == '__main__':
    run_example()
