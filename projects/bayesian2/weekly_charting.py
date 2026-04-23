import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_data():
    """Create fake data with effects for Run, Technician, and Reagent."""
    np.random.seed(42)

    num_runs = 11
    plates_per_run = 8
    wells_per_plate = 96

    base_mu, base_sigma = 100, 2
    technicians = ['Jane', 'Mike', 'Bob']
    reagents = ['A', 'B']
    start_date = datetime(2026, 1, 5)

    # Generate data for each of the weekly runs.
    data = []
    for i in range(num_runs):
        run_date = start_date + timedelta(weeks=i)
        iso_date = run_date.strftime('%Y-%m-%d')
        is_early = (i < 7)
        is_clogged = (i == 4 or i == 7) # Instrument issues
        run_mu_off, run_sigma_off = (-25, 2) if is_clogged else (0, 0)
        
        for p_idx in range(plates_per_run):
            tech = np.random.choice(technicians[:-1] if i < 4 else technicians)
            reagent = np.random.choice(reagents)
            
            # Bob improves over time
            if tech == 'Bob':
                tech_mu = -15 if is_early else 0
                tech_sigma = 3 if is_early else 1
            else:
                tech_mu, tech_sigma = 0, 1

            # Reagent B has a slight negative bias
            reagent_mu = 0 if reagent == 'B' else 0

            total_mean = base_mu + run_mu_off + tech_mu + reagent_mu
            total_sigma = np.sqrt(base_sigma**2 + run_sigma_off**2 + tech_sigma**2)
            plate_results = np.random.normal(total_mean, total_sigma, wells_per_plate)
            
            for well_val in plate_results:
                data.append({
                    'date': iso_date, 
                    'reagent': reagent, 
                    'technician': tech, 
                    'value': well_val
                })
    return pd.DataFrame(data)

def analyze_weekly(df):
    """Run a rolling window model and collect Tech, Reagent, and Run effects."""
    dates = sorted(df['date'].unique())
    all_stats = []

    # Analyze each window starting from the 4th week (index 3) to ensure a full 4-run window.
    for i in range(3, len(dates)):
        window_dates = dates[i-3 : i+1]
        analysis_date = dates[i]
        print(f"Analyzing window ending at: {analysis_date}")
        
        window_df = df[df['date'].isin(window_dates)].copy()
        reagents = sorted(window_df['reagent'].unique())
        techs = sorted(window_df['technician'].unique())
        runs = sorted(window_df['date'].unique())

        # Model each know categorical thing recorded in the data: reagent, tech, and run.
        with pm.Model(coords={"reagent": reagents, "tech": techs, "run": runs}) as model:
            reagent_idx = pd.Categorical(window_df['reagent'], categories=reagents).codes
            tech_idx = pd.Categorical(window_df['technician'], categories=techs).codes
            run_idx = pd.Categorical(window_df['date'], categories=runs).codes

            # Start the model assuming all runs are normal and all techs and reagents are fine.
            mu_run = pm.Normal('mu_run', mu=100, sigma=3, dims="run")
            mu_reagent = pm.Normal('mu_reagent', mu=0, sigma=1, dims="reagent")
            mu_tech = pm.Normal('mu_tech', mu=0, sigma=1, dims="tech")
            sigma = pm.Exponential('sigma', lam=0.1)
            
            mu = mu_run[run_idx] + mu_reagent[reagent_idx] + mu_tech[tech_idx]
            obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=window_df['value'])

            # Warm-up the model with 500 samples, then sample 1000.
            trace = pm.sample(1000, tune=500, target_accept=0.9, return_inferencedata=True, progressbar=True)

            # Record the HDI 94% for each of the three effects.
            # 1. Collect Technician effects
            summary_tech = az.summary(trace, var_names=['mu_tech'])
            for tech_name in techs:
                stat = summary_tech.loc[f"mu_tech[{tech_name}]"]
                all_stats.append({
                    'date': analysis_date, 'type': 'Technician', 'label': tech_name,
                    'mean': stat['mean'], 'lower': stat['hdi_3%'], 'upper': stat['hdi_97%']
                })
                
            # 2. Collect Reagent effects
            summary_reagent = az.summary(trace, var_names=['mu_reagent'])
            for reagent_name in reagents:
                stat = summary_reagent.loc[f"mu_reagent[{reagent_name}]"]
                all_stats.append({
                    'date': analysis_date, 'type': 'Reagent', 'label': reagent_name,
                    'mean': stat['mean'], 'lower': stat['hdi_3%'], 'upper': stat['hdi_97%']
                })
                
            # 3. Collect Run effect (Specifically for the current run in the window)
            summary_run = az.summary(trace, var_names=['mu_run'])
            run_stat = summary_run.loc[f"mu_run[{analysis_date}]"]
            all_stats.append({
                'date': analysis_date, 'type': 'Run Baseline', 'label': 'Current Run',
                'mean': run_stat['mean'], 'lower': run_stat['hdi_3%'], 'upper': run_stat['hdi_97%']
            })
            
    return pd.DataFrame(all_stats)

def plot_weekly_chart(stats_df):
    """Plot trends for Technician, Reagent, and Run Baseline in subplots."""
    effect_types = ['Technician', 'Reagent', 'Run Baseline']
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    for i, effect_type in enumerate(effect_types):
        ax = axes[i]
        subset = stats_df[stats_df['type'] == effect_type]
        
        for label in subset['label'].unique():
            label_df = subset[subset['label'] == label]
            ax.errorbar(label_df['date'], label_df['mean'], 
                         yerr=[label_df['mean'] - label_df['lower'], 
                               label_df['upper'] - label_df['mean']],
                         fmt='o-', label=label, capsize=5)

        baseline = 100 if effect_type == 'Run Baseline' else 0
        ax.axhline(baseline, color='black', linestyle='--', alpha=0.3)
        ax.set_title(f"{effect_type} Effects Over Time")
        ax.set_ylabel("Estimated Mean")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.2)

    plt.xlabel("Analysis Date (End of Rolling Window)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('weekly_chart.png')
    plt.show()

if __name__ == "__main__":
    data = load_data()
    data.to_csv('weekly_data.csv', index=False)
    stats = analyze_weekly(data)
    plot_weekly_chart(stats)
    print("\nRolling Analysis Complete. Results saved to weekly_chart.png")
