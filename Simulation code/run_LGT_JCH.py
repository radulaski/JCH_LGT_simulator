import os
import json
from LGT_JCH_funcs import simulate_JCH, plotting, save_results, simulate_lgt

# Parameters
w = 450
g = 300
J = 0.9
delt = 100
lam = 36
param = 0
t_max = 500/J
init = [0, 2, 0]

# Run simulation of JCH model
data = simulate_JCH(w, g, J, lam, init, param, t_max, n_sites=len(init))

# Run simulation of spin-1/2 QLM
data_qlm = simulate_lgt(init, t=0.0182202, m=0.005795, t_max=500/J, n_t=15000)

output = data["output"]
tlist = data["tlist"]
n_sites = data["n_sites"]
filename = data["filename"]
timestamp = data["timestamp"]

# Plot
plotting(output, tlist, n_sites, filename, data_qlm)

# Save CSV results
save_results(output, tlist, n_sites, filename)

# Collect metadata
params = {
    "w": w,
    "g": g,
    "J": J,
    "lam": lam,
    "param": param,
    "t_max": t_max,
    "init": init,
    "n_sites": n_sites,
}

# Save all
folder = filename[:-4]
with open(os.path.join(folder, f"{folder}_params.json"), "w") as f:
    json.dump(params, f, indent=2)

print(f"Done. Results saved to '{folder}/'")