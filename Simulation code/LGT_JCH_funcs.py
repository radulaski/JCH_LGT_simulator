import numpy as np
import os
import pandas as pd
from qutip import (
    sigmam, sigmap, sigmaz, qeye, tensor, basis, Options, sesolve, destroy
)
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib as mpl
import matplotlib.gridspec as gridspec

# Color palette for plotting
MATTER_COLORS = ["#E69F00", "#D55E00", "#CC79A7", "#F0E442"]
GAUGE_COLORS  = ["#0072B2", "#009E73", "#56B4E9", "#000000"]
GAUSS_COLORS  = ["#E69F00", "#D55E00", "#CC79A7", "#F0E442"]
MAG_COLORS    = ["#0072B2", "#009E73", "#56B4E9"]

# mpl parameters
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "text.usetex":        False,
    "mathtext.fontset":   "cm",
    "axes.linewidth":     0.8,
    "axes.labelsize":     12,
    "axes.titlesize":     12,
    "axes.titlepad":      5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "xtick.minor.size":   2.0,
    "ytick.minor.size":   2.0,
    "legend.fontsize":    8.3,
    "legend.framealpha":  0,
    "legend.edgecolor":   "0.8",
    "legend.borderpad":   0.4,
    "legend.handlelength": 1.5,
    "lines.linewidth":    1.2,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
})


def simulate_lgt(init, n_sites=2, t=1.0, m=0.5, t_max=20, n_t=200):
    n_gauge = n_sites - 1
    total_sites = n_sites + n_gauge

    def embed(op, idx):
        ops = [qeye(2) for _ in range(total_sites)]
        ops[idx] = op
        return tensor(ops)

    psi_ops = []
    for i in range(n_sites):
        ops = []
        for j in range(total_sites):
            if j < i:
                ops.append(sigmaz() if j < n_sites else qeye(2))
            elif j == i:
                ops.append(sigmam())
            else:
                ops.append(qeye(2))
        psi_ops.append(tensor(ops))

    n_ops = [psi.dag() * psi for psi in psi_ops]
    q_ops = [(1-psi.dag() * psi) if i % 2 == 0 else (psi.dag() * psi)
             for i, psi in enumerate(psi_ops)]

    Splus = [embed(sigmap(), n_sites + l) for l in range(n_gauge)]
    Sminus = [embed(sigmam(), n_sites + l) for l in range(n_gauge)]
    Sz = [(-1)**l * 0.5 * embed(sigmaz(), n_sites + l) for l in range(n_gauge)]

    H = 0 * embed(qeye(2), 0)

    for l in range(n_gauge):
        H -= (t) * (
            psi_ops[l] * Splus[l] * psi_ops[l + 1] +
            psi_ops[l + 1].dag() * Sminus[l] * psi_ops[l].dag()
        )

    for l in range(n_sites):
        H += m * n_ops[l]

    psi_init = [basis(2, 0) if init[2*i]==1 else basis(2, 1)
                for i in range(n_sites)]
    gauge_init = [basis(2, 1) if init[2*j+1]==0 else basis(2,0) for j in range(n_gauge)]
    psi0 = tensor(psi_init + gauge_init)

    tlist = np.linspace(0, t_max, n_t)
    e_ops = q_ops + Sz

    opts = Options(store_states=True, nsteps=5000)
    result = sesolve(H, psi0, tlist, e_ops, options=opts)

    data = {
        "tlist": tlist,
        "q_expect": np.array(result.expect[:n_sites]),
        "Sz_expect": np.array(result.expect[n_sites:]),
        "n_sites": n_sites,
        "n_gauge": n_gauge
    }

    return data

def E1m(Delta, g):
    return -Delta + Delta/2 - np.sqrt((Delta/2)**2 + g**2)

def E2m(Delta, g):
    return -2*Delta + Delta/2 - np.sqrt((Delta/2)**2 + 2*g**2)

def get_detunings_staggered(Delta_1, g, lam, n_sites):
    chi1 = np.sqrt(4*g**2 + Delta_1**2)

    inner = (-16*g**2 + lam**2 - 2*lam*Delta_1 - 7*Delta_1**2
             - 2*lam*chi1 + 2*Delta_1*chi1 
             + 9*(4*g**2 + Delta_1**2))
    Delta_2 = (1/4)*(-3*lam + 3*Delta_1 + 3*chi1 - np.sqrt(inner))

    numer = (-4*g**2*lam + 2*lam**3 + 2*g**2*Delta_1
             - 3*lam**2*Delta_1 + lam*Delta_1**2
             - lam**2*chi1 + lam*Delta_1*chi1)
    denom = 2*(g**2 - lam**2 + lam*Delta_1)
    Delta_3 = numer / denom
    
    pattern = [Delta_1, Delta_2, Delta_3, Delta_2]
    return np.array([pattern[i % 4] for i in range(n_sites)])

def simulate_JCH(we,g,J,lam,init, param, t_max, n_sites):
    # Physical parameters
    N_cav = 3
    N_q = 2
    
    # Create lists to store operators
    a_full = []  # cavity annihilation operators
    sm_full = []    # qubit lowering operators
    
    # Build tensor product operators for each site
    for i in range(n_sites):
        # Create identity operators for all spaces
        ops_cav = [qeye(N_cav) for _ in range(n_sites)]
        ops_q = [qeye(N_q) for _ in range(n_sites)]
        
        # Set the i-th cavity operator to destroy
        ops_cav[i] = destroy(N_cav)
        
        # Build full tensor product
        full_ops = []
        for j in range(n_sites):
            full_ops.extend([ops_cav[j], ops_q[j]])
        
        a_full.append(tensor(*full_ops))
        
        # Reset and create qubit operator
        ops_cav = [qeye(N_cav) for _ in range(n_sites)]
        ops_q = [qeye(N_q) for _ in range(n_sites)]
        ops_q[i] = destroy(N_q)
        
        full_ops = []
        for j in range(n_sites):
            full_ops.extend([ops_cav[j], ops_q[j]])
            
        sm_full.append(tensor(*full_ops))
    
    # Build Jaynes-Cummings Hamiltonian
    H_JC = 0

    # Detuning parameters
    deltas = get_detunings_staggered(param, g, lam, n_sites)

    for i in range(n_sites):
        wc = we - deltas[i]
            
        # Add cavity and qubit free evolution
        H_JC += wc * a_full[i].dag() * a_full[i]
        H_JC += we * sm_full[i].dag() * sm_full[i]
        
        # Add interaction term
        H_JC += g * (a_full[i] * sm_full[i].dag() + a_full[i].dag() * sm_full[i])
    
    # Build hopping Hamiltonian
    H_hop = 0
    for i in range(n_sites):
        if i != n_sites - 1:  # Last site
            H_hop += (a_full[i] * a_full[i+1].dag() + 
                      a_full[i].dag() * a_full[i+1])
    
    # Total Hamiltonian
    H = H_JC - J * H_hop
    
    # Create initial state
    site_states = []
    for i in range(n_sites):
        if init[i]==1: 
            psi_site = (tensor(basis(N_cav,1), basis(N_q,0)) - 
                       tensor(basis(N_cav,0), basis(N_q,1))).unit()
        elif init[i]==2:
            psi_site = (tensor(basis(N_cav,2), basis(N_q,0)) - 
                       tensor(basis(N_cav,1), basis(N_q,1))).unit()
        else:
            psi_site = tensor(basis(N_cav,0), basis(N_q,0))
    
        site_states.append(psi_site)
    
    psi0 = tensor(*site_states).unit()
    
    # Time evolution
    tlist = np.linspace(0, t_max, 2000)
    
    # Expectation value operators
    e_ops = []
    
    # Cavity photon numbers
    for i in range(n_sites):
        e_ops.append(a_full[i].dag() * a_full[i])
    
    # Qubit excitation numbers  
    for i in range(n_sites):
        e_ops.append(sm_full[i].dag() * sm_full[i])

    # Projection onto first polariton
    lp1_proj = []
    for i in range(n_sites):
        ops_list = []
        # Four terms in the projector expansion
        terms = [
            ( 1, basis(N_cav,1)*basis(N_cav,1).dag(), basis(N_q,0)*basis(N_q,0).dag()),
            (-1, basis(N_cav,1)*basis(N_cav,0).dag(), basis(N_q,0)*basis(N_q,1).dag()),
            (-1, basis(N_cav,0)*basis(N_cav,1).dag(), basis(N_q,1)*basis(N_q,0).dag()),
            ( 1, basis(N_cav,0)*basis(N_cav,0).dag(), basis(N_q,1)*basis(N_q,1).dag()),
        ]
        proj = 0
        for sign, cav_op, q_op in terms:
            ops_cav = [qeye(N_cav)] * n_sites
            ops_q   = [qeye(N_q)]   * n_sites
            ops_cav[i] = cav_op
            ops_q[i]   = q_op
            full_ops = []
            for j in range(n_sites):
                full_ops.extend([ops_cav[j], ops_q[j]])
            proj += sign * tensor(*full_ops)
        lp1_proj.append(proj / 2)

    # LP2 projectors
    lp2_proj = []
    for i in range(n_sites):
        terms = [
            ( 1, basis(N_cav,2)*basis(N_cav,2).dag(), basis(N_q,0)*basis(N_q,0).dag()),
            (-1, basis(N_cav,2)*basis(N_cav,1).dag(), basis(N_q,0)*basis(N_q,1).dag()),
            (-1, basis(N_cav,1)*basis(N_cav,2).dag(), basis(N_q,1)*basis(N_q,0).dag()),
            ( 1, basis(N_cav,1)*basis(N_cav,1).dag(), basis(N_q,1)*basis(N_q,1).dag()),
        ]
        proj = 0
        for sign, cav_op, q_op in terms:
            ops_cav = [qeye(N_cav)] * n_sites
            ops_q   = [qeye(N_q)]   * n_sites
            ops_cav[i] = cav_op
            ops_q[i]   = q_op
            full_ops = []
            for j in range(n_sites):
                full_ops.extend([ops_cav[j], ops_q[j]])
            proj += sign * tensor(*full_ops)
        lp2_proj.append(proj / 2)

    # P^phys projectors
    phys_proj = []
    for i in range(n_sites):
        ops_cav = [qeye(N_cav)] * n_sites
        ops_q   = [qeye(N_q)]   * n_sites
        ops_cav[i] = basis(N_cav, 0) * basis(N_cav, 0).dag()
        ops_q[i]   = basis(N_q,  0) * basis(N_q,  0).dag()
        full_ops = []
        for j in range(n_sites):
            full_ops.extend([ops_cav[j], ops_q[j]])
        vac_proj = tensor(*full_ops)
        phys_proj.append(vac_proj)

    for i in range(n_sites):
        e_ops.append(phys_proj[i]+lp1_proj[i])
    
    for i in range(n_sites):
        e_ops.append(phys_proj[i]+lp2_proj[i])
   
    # Time evolution of SEQ
    options = Options(nsteps = 200000, rtol=1e-10, atol=1e-10, max_step=0.01, progress_bar = 'tqdm')
    output = sesolve(H, psi0, tlist, e_ops, options=options)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{n_sites}_site_{timestamp}.png"

    data = {
        "output": output,
        "tlist": tlist,
        "n_sites":n_sites,
        "filename": filename,
        "timestamp": timestamp
    }
    
    return data

def plotting(output, tlist, n_sites, filename, data=None):
    n_expect = np.array(output.expect[:n_sites])
    q_expect = np.array(output.expect[n_sites:])

    even_idx = np.arange(0, n_sites, 2)
    odd_idx  = np.arange(1, n_sites, 2)

    n_even, q_even = n_expect[even_idx], q_expect[even_idx]
    n_odd,  q_odd  = n_expect[odd_idx],  q_expect[odd_idx]

    even_occ = n_even + q_even
    odd_occ  = ((n_odd + q_odd) - 1) / 2

    for i in range(len(odd_occ)):
        odd_occ[i] *= (-1)**i
    for i in range(len(even_occ)):
        if i % 2 == 0:
            even_occ[i] = (1 - even_occ[i])

    n_matter = len(even_occ)
    n_gauge  = len(odd_occ)

    if data is not None:
        tlist_qlm  = data["tlist"]
        q_expect_qlm = data["q_expect"]
        Sz_expect_qlm = data["Sz_expect"]

        fig2 = plt.figure(figsize=(9.5, 3))
        gs2 = gridspec.GridSpec(
            1, 3, figure=fig2,
            hspace=0.38, wspace=0.35,
            left=0.09, right=0.97, top=0.95, bottom=0.09
        )
        ax_m = fig2.add_subplot(gs2[0, 0])
        ax_g  = fig2.add_subplot(gs2[0, 1])

        subgs = gs2[0, 2].subgridspec(3, 1, 
                hspace=0.5, wspace=0.9)
        
        ax_GL = fig2.add_subplot(subgs[0, 0])
        ax_lp1 = fig2.add_subplot(subgs[1,0])
        ax_lp2 = fig2.add_subplot(subgs[2,0], sharex=ax_lp1)

        ax_lp1.tick_params(labelbottom=False)
        ax_GL.tick_params(labelbottom=False)

        # Matter sites
        for i in range(n_matter):
            jch_val = even_occ[i]
            c = MATTER_COLORS[i % len(MATTER_COLORS)]
            ax_m.plot(tlist, jch_val,
                      color=c, lw=1, zorder=2,
                      label=f"JCH $m_{{{i}}}$")
        for i in range(n_matter):    
            c = MATTER_COLORS[i % len(MATTER_COLORS)]
            ax_m.plot(tlist_qlm, q_expect_qlm[i],
                      color=c, lw=3, ls="--", alpha=0.6, zorder=3,
                      label=f"QLM $m_{{{i}}}$")

        ax_m.set_xlabel("Time ($J^{-1}$)")
        ax_m.set_ylabel(r"$\langle \psi^\dagger_l \psi_l \rangle$")
        ax_m.set_title("(a) Matter sites")
        ax_m.set_ylim(-0.05, 1.19)

        # Gauge sites
        for i in range(n_gauge):
            jch_val = odd_occ[i]
            c = GAUGE_COLORS[i % len(GAUGE_COLORS)]
            ax_g.plot(tlist, jch_val,
                      color=c, lw=1, zorder=2,
                      label=f"JCH $l_{{{i}}}$")
        for i in range(n_gauge):
            c = GAUGE_COLORS[i % len(GAUGE_COLORS)]
            ax_g.plot(tlist_qlm, Sz_expect_qlm[i],
                      color=c, lw=3.0, ls="--", alpha=0.6, zorder=3,
                      label=f"QLM $l_{{{i}}}$")

        ax_g.set_xlabel("Time ($J^{-1}$)")
        ax_g.set_ylabel(r"$\langle S^z_{l} \rangle$")
        ax_g.set_ylim(-0.5,0.5)
        ax_g.set_title("(b) Gauge fields")

        ax_m.legend().set_visible(False)

        handles_g, labels_g = ax_g.get_legend_handles_labels()
        ordered_handles_g, ordered_labels_g = [], []
        for i in range(n_gauge):
            ordered_handles_g.append(handles_g[i])
            ordered_labels_g.append(labels_g[i])
            ordered_handles_g.append(handles_g[n_gauge + i])
            ordered_labels_g.append(labels_g[n_gauge + i])

        ax_g.legend(
            ordered_handles_g, ordered_labels_g,
            ncol=2,
            loc="best",
            borderaxespad=0,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=0.8,
            borderpad=0.4,
            fontsize=8.5,
        ).get_frame().set_linewidth(0.6)

        handles_m, labels_m = ax_m.get_legend_handles_labels()
        ordered_handles_m, ordered_labels_m = [], []
        for i in range(n_matter):
            ordered_handles_m.append(handles_m[i])
            ordered_labels_m.append(labels_m[i])
            ordered_handles_m.append(handles_m[n_matter + i])
            ordered_labels_m.append(labels_m[n_matter + i])

        ax_m.legend(
            ordered_handles_m, ordered_labels_m,
            ncol=3,
            loc="best",
            borderaxespad=0,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing = 0.8,
            fontsize=8.5,
        ).get_frame().set_linewidth(0.6)

    for i in range(n_matter):
        if i > 0 and i < n_matter - 1:
            G = abs(abs(even_occ[i] - (odd_occ[i] - odd_occ[i-1])-1)) if i%2==0 \
                 else abs(abs(even_occ[i] + odd_occ[i-1] - odd_occ[i]))
        elif i == 0:
            G = abs(abs(even_occ[i] - (odd_occ[i] - 0.5)-1))
        elif i % 2 == 0:
            G = abs(abs(even_occ[i] - (-0.5 - odd_occ[i-1])-1))
        else:
            G = abs(abs(even_occ[i] - (0.5 - odd_occ[i-1])))

        ax_GL.plot(tlist, G,
                   color=GAUSS_COLORS[i % len(GAUSS_COLORS)],
                   lw=1.3, label=f"$G_{{{i}}}$, max: {np.max(G):.1e}")


    ax_GL.set_ylabel(r"$|\langle G_l \rangle|$")
    ax_GL.set_ylim(-0.05, 1.05)
    ax_GL.set_title("(c) Gauss's law")
    _add_legend(ax_GL, ncol=2)

    proj_matter = np.array(output.expect[2*n_sites:3*n_sites])
    proj_gauge = np.array(output.expect[3*n_sites:4*n_sites])

    p_gauge  = proj_gauge[odd_idx]
    p_matter = proj_matter[even_idx]

    for i in range(n_gauge):
        ax_lp1.plot(
            tlist, abs(1-p_gauge[i]),
            color=GAUGE_COLORS[i % len(GAUGE_COLORS)],
            lw=1.3, label=f"$l_{{{i}}}$, max: {np.max(1-p_gauge[i]):.1e}"
        )
    ax_lp1.set_ylabel(r"$\langle P_{\text{phys}} \rangle$")
    ax_lp1.set_title(r"(d) $P_{\text{phys}}$ projection — gauge sites")
    ax_lp1.set_ylim(-0.05, 1.05)
    _add_legend(ax_lp1, ncol=2)

    for i in range(n_matter):
        ax_lp2.plot(
            tlist, abs(1-p_matter[i]),
            color=MATTER_COLORS[i % len(MATTER_COLORS)],
            lw=1.3, label=f"$m_{{{i}}}$, max: {np.max(1-p_matter[i]):.1e}"
        )
    ax_lp2.set_xlabel("Time ($J^{-1}$)")
    ax_lp2.set_ylabel(r"$\langle P_{\text{phys}} \rangle$")
    ax_lp2.set_title(r"(e) $P_{\text{phys}}$ projection — matter sites")
    ax_lp2.set_ylim(-0.05, 1.05)
    _add_legend(ax_lp2, ncol=2)

    folder = filename[:-4]
    os.makedirs(folder, exist_ok=True)
    fig2.savefig(os.path.join(folder, f"{os.path.basename(folder)}.pdf"))

def _add_legend(ax, ncol=1):
    leg = ax.legend(ncol=ncol, loc="best",
                    handlelength=1.2, handletextpad=0.4,
                    columnspacing=0.8, borderpad=0.4)
    leg.get_frame().set_linewidth(0.6)

def save_results(output, tlist, n_sites, filename):
    n_expect = np.array(output.expect[:n_sites])
    q_expect = np.array(output.expect[n_sites:])
    
    df = pd.DataFrame({"time": tlist})
    
    for i in range(n_sites):
        df[f"n_site_{i}"] = n_expect[i]
        df[f"q_site_{i}"] = q_expect[i]
    
    save_path = os.path.join(filename[:-4], f"{filename[:-4]}.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")