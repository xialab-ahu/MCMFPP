import numpy as np
from scipy.stats import ttest_rel
pscfa_data = {
    "aiming": [0.726, 0.716, 0.720, 0.708, 0.708, 0.694, 0.706, 0.716, 0.721, 0.716],
    "coverage": [0.689, 0.676, 0.689, 0.672, 0.688, 0.667, 0.681, 0.686, 0.696, 0.679],
    "accuracy": [0.671, 0.657, 0.665, 0.655, 0.661, 0.641, 0.658, 0.661, 0.673, 0.659],
    "absolute_true": [0.608, 0.587, 0.597, 0.596, 0.602, 0.569, 0.600, 0.589, 0.618, 0.586],
    "1-absolute_false": [1-0.035, 1-0.036, 1-0.038, 1-0.037, 1-0.038, 1-0.039, 1-0.039, 1-0.038, 1-0.038, 1-0.037]
}

etfc_data = {
    "aiming": [0.689, 0.713, 0.702, 0.695, 0.688, 0.702, 0.704, 0.712, 0.707, 0.702],
    "coverage": [0.678, 0.693, 0.687, 0.678, 0.687, 0.684, 0.698, 0.696, 0.708, 0.699],
    "accuracy": [0.652, 0.665, 0.657, 0.652, 0.647, 0.657, 0.665, 0.667, 0.670, 0.659],
    "absolute_true": [0.595, 0.595, 0.585, 0.594, 0.582, 0.589, 0.602, 0.602, 0.602, 0.582],
    "1-absolute_false": [1-0.039, 1-0.037, 1-0.038, 1-0.039, 1-0.040, 1-0.038, 1-0.039, 1-0.037, 1-0.037, 1-0.039]
}

mcmfpp_data = {
    "aiming": [0.746, 0.744, 0.727, 0.736, 0.732, 0.728, 0.728, 0.733, 0.747, 0.733],
    "coverage": [0.707, 0.702, 0.684, 0.698, 0.706, 0.687, 0.691, 0.694, 0.719, 0.702],
    "accuracy": [0.692, 0.688, 0.672, 0.684, 0.683, 0.673, 0.677, 0.679, 0.702, 0.682],
    "absolute_true": [0.632, 0.626, 0.613, 0.631, 0.622, 0.611, 0.620, 0.617, 0.645, 0.618],
    "1-absolute_false": [1-0.034, 1-0.033, 1-0.036, 1-0.035, 1-0.035, 1-0.037, 1-0.036, 1-0.035, 1-0.033, 1-0.035]
}
num_metrics = len(mcmfpp_data.keys())
alpha = 0.01
alpha_corrected = alpha / 10  # The significance level after Bonferroni correction
print(f"The significance level after Bonferroni correction: {alpha_corrected:.4f}\n")
all_results = []

# Compare MCMFPP with all other methods
methods = {
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