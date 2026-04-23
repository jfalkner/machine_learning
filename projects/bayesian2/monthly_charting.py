import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_data():
    """Create fake data for a series of assay runs over eight weeks.
    
    Early runs: Tech 'Bob' is still learning and has a negative bias.
    Later runs: Bob has improved and matches the performance of 'Jane'.
    Sporadic: Certain runs have instrument-related issues (low mean, high noise).
    """
    np.random.seed(42)
    
    num_runs = 8
    plates_per_run = 6
    wells_per_plate = 96
    
    data = []
    
    # Baseline for a good run
    base_mu = 100
    base_sigma = 5
    
    technicians = ['Jane', 'Bob']
    reagents = ['A', 'B']
    
    start_date = datetime(2026, 1, 5) # A Monday
    
    for i in range(num_runs):
        run_date = start_date + timedelta(weeks=i)
        iso_date = run_date.strftime('%Y-%m-%d')
        
        # Logic for Bob's improvement (first 4 runs vs last 4)
        is_early_period = (i < 4)
        
        # Instrument issues (sporadic) on specific dates
        is_clogged = (i == 3 or i == 7)
        
        run_mu_offset = -25 if is_clogged else 0
        run_sigma_offset = 10 if is_clogged else 0
        
        for p_idx in range(plates_per_run):
            tech = np.random.choice(technicians)
            reagent = np.random.choice(reagents)
            
            # Bob's performance improves over time
            if tech == 'Bob':
                tech_mu = -15 if is_early_period else 0
                tech_sigma = 5 if is_early_period else 1
            else: # Jane
                tech_mu = 0
                tech_sigma = 1
                
            total_mean = base_mu + run_mu_offset + tech_mu
            total_sigma = np.sqrt(base_sigma**2 + run_sigma_offset**2 + tech_sigma**2)
            
            plate_results = np.random.normal(total_mean, total_sigma, wells_per_plate)
            
            for well_val in plate_results:
                data.append({
                    'date': iso_date,
                    'reagent': reagent,
                    'technician': tech,
                    'value': well_val
                })
                
    return pd.DataFrame(data)

def model_data(df):
    """Fit a Bayesian model to the data, grouped by calendar month."""
    results = {}
    
    # Extract month from ISO date for grouping
    df['group_month'] = df['date'].apply(lambda x: x[:7]) # YYYY-MM
    
    for month in sorted(df['group_month'].unique()):
        month_df = df[df['group_month'] == month]
        runs = sorted(month_df['date'].unique())
        reagents = sorted(month_df['reagent'].unique())
        techs = sorted(month_df['technician'].unique())
        
        with pm.Model(coords={"run": runs, "reagent": reagents, "tech": techs}) as model:
            run_idx = pd.Categorical(month_df['date'], categories=runs).codes
            reagent_idx = pd.Categorical(month_df['reagent'], categories=reagents).codes
            tech_idx = pd.Categorical(month_df['technician'], categories=techs).codes
            
            mu_run = pm.Normal('mu_run', mu=100, sigma=20, dims="run")
            sigma_run = pm.Exponential('sigma_run', lam=0.1, dims="run")
            
            mu_reagent = pm.Normal('mu_reagent', mu=0, sigma=10, dims="reagent")
            sigma_reagent = pm.Exponential('sigma_reagent', lam=0.1, dims="reagent")
            
            mu_tech = pm.Normal('mu_tech', mu=0, sigma=10, dims="tech")
            sigma_tech = pm.Exponential('sigma_tech', lam=0.1, dims="tech")
            
            mu = mu_run[run_idx] + mu_reagent[reagent_idx] + mu_tech[tech_idx]
            var_total = (sigma_run[run_idx]**2 +
                         sigma_reagent[reagent_idx]**2 +
                         sigma_tech[tech_idx]**2)
            sigma = pm.Deterministic('sigma', pm.math.sqrt(var_total))
            
            obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=month_df['value'])
            
            trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True, progressbar=True)
            results[month] = trace
            
    return results

def plot_results(results):
    """Plot the results for each month to show trends."""
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 8), sharex=True)
    
    if len(results) == 1:
        axes = [axes]

    for i, (month_label, trace) in enumerate(sorted(results.items())):
        az.plot_forest(trace, var_names=['mu_tech', 'mu_run'], ax=axes[i], combined=True)
        axes[i].set_title(f"Period {month_label} Effects")
        axes[i].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[i].axvline(100, color='blue', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig('monthly_chart.png')
    plt.show()

if __name__ == "__main__":
    df = load_data()
    print("Data loaded. Runs per period:")
    print(df.groupby(df['date'].str[:7])['date'].nunique())
    
    results = model_data(df)
    plot_results(results)
    
    for month_label, trace in sorted(results.items()):
        print(f"\nPeriod {month_label} Summary:")
        print(az.summary(trace, var_names=['mu_tech', 'mu_run']))
