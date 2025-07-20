import numpy as np
from scipy.stats import ttest_rel
prmftp_data = {
    "aiming": [0.697, 0.703, 0.705, 0.695, 0.698],
    "coverage": [0.663, 0.670, 0.673, 0.668, 0.671],
    "accuracy": [0.646, 0.654, 0.654, 0.649, 0.652],
    "absolute_true": [0.586, 0.598, 0.594, 0.593, 0.594],
    "1-absolute_false": [1-0.032, 1-0.030, 1-0.031, 1-0.031, 1-0.031]
}

etfc_data = {
    "aiming": [0.724, 0.724, 0.724, 0.725, 0.722],
    "coverage": [0.717, 0.718, 0.717, 0.717, 0.718],
    "accuracy": [0.684, 0.684, 0.683, 0.685, 0.682],
    "absolute_true": [0.618, 0.617, 0.617, 0.618, 0.615],
    "1-absolute_false": [1-0.036, 1-0.036, 1-0.036, 1-0.036, 1-0.036]
}

pscfa_data = {
    "aiming": [0.742, 0.745, 0.751, 0.736, 0.746],
    "coverage": [0.715, 0.716, 0.726, 0.711, 0.716],
    "accuracy": [0.695, 0.695, 0.705, 0.691, 0.697],
    "absolute_true": [0.636, 0.633, 0.646, 0.632, 0.640],
    "1-absolute_false": [1-0.034, 1-0.035, 1-0.033, 1-0.034, 1-0.034]
}

mcmfpp_data = {
    "aiming": [0.772, 0.768, 0.782, 0.773, 0.778],
    "coverage": [0.739, 0.734, 0.747, 0.742, 0.742],
    "accuracy": [0.721, 0.714, 0.731, 0.724, 0.725],
    "absolute_true": [0.661, 0.652, 0.672, 0.664, 0.666],
    "1-absolute_false": [1 - 0.031, 1 - 0.032, 1 - 0.029, 1 - 0.030, 1 - 0.030]
}

num_metrics = len(mcmfpp_data.keys())
alpha = 0.01

alpha_corrected = alpha / 5  # The significance level after Bonferroni correction

print(f"The significance level after Bonferroni correction: {alpha_corrected:.4f}\n")

# Store results
all_results = []

# Compare MCMFPP with all other methods
methods = {
    "PRMFTP": prmftp_data,
    "ETFC": etfc_data,
    "PSCFA": pscfa_data
}

for method_name, method_data in methods.items():
    print(f"Comparison results with {method_name}:")
    print("-" * 70)
    results = []
    for metric in mcmfpp_data.keys():
        x, y = mcmfpp_data[metric], method_data[metric]
        # Calculate means and standard deviations
        mean_x, mean_y = np.mean(x), np.mean(y)
        std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)
        # Perform paired t-test
        stat, p = ttest_rel(x, y)
        # Determine significance (using corrected alpha)
        significant = p < alpha_corrected
        # Calculate mean difference
        mean_diff = mean_x - mean_y
        # Store results
        results.append({
            'metric': metric,
            'mean_mcmfpp': mean_x,
            'mean_method': mean_y,
            'mean_diff': mean_diff,
            'std_mcmfpp': std_x,
            'std_method': std_y,
            'p_value': p,
            'significant': significant
        })

    # Print results
    print(f"{'Metric':<15} {'MCMFPP Mean':>10} {method_name} Mean {'Mean Diff':>10} {'p-value':>10} {'Significant':>10}")
    print("-" * 70)
    for res in results:
        p_value_scientific = f"{res['p_value']:.1e}"  # Scientific notation
        print(f"{res['metric']:<15} {res['mean_mcmfpp']:>10.4f} {res['mean_method']:>10.4f} "
              f"{res['mean_diff']:>10.4f} {p_value_scientific:>10} {str(res['significant']):>10}")
    print("\n")