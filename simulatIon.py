# simulatIon.py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from TVCM_SO import SocialTVCM
from TVCM_simple import SimpleTVCM

# Base TVCM parameters
KW = dict(
    Gx=6, Gy=6, cell=100,
    n_comm=10, n_nodes=505,
    p_range=(0.6, 0.95), v_range=(1.0, 3.0),
    radio=20, seed=42
)


SO_KW = dict(slot_len=3600.0, n_slots=24)


# Simulation parameters
HOURS = 24         # Run longer to establish strong links
DT = 10.0            # Time step
P_SPREAD = 0.9      # Threshold for spread calculation

def simulate(model, hrs, dt):
    steps = int(hrs * 3600 / dt)
    w = {}
    for _ in range(steps):
        pos = model.step(dt)
        for i, j in model.contacts(pos):
            e = (i, j) if i < j else (j, i)
            w[e] = w.get(e, 0.0) + dt
    return w


def strong_edges(weights, home):
    """
    A link (k,l) is *strong* iff
          w_kl  >  median(W) + 3·σ(W)
    where  W  is the set of **all** inter-community weights (raw seconds).
    """
    # 1) collect every inter-community weight once
    inter = [(e, w) for e, w in weights.items()
             if home[e[0]] != home[e[1]] and w > 0.0]

    if not inter:  # no cross-community contacts
        return []

    w_all = np.fromiter((w for _, w in inter), float)
    
    # Normalize weights by total simulation time
    total_time = max(w_all)
    w_norm = w_all / total_time
    
    # Use a lower threshold multiplier (2 instead of 3)
    thr = (np.median(w_norm) + 2 * w_norm.std(ddof=0)) * total_time

    # 2) keep links above threshold
    strong = [e if e[0] < e[1] else (e[1], e[0])
             for e, w in inter if w > thr]
    return strong


def node_spread(node, target_comm, home, weights):
    """Calculate node spread S(k → Cj) = |C_k|/|C_j|"""
    # Get all nodes in target community (|C_j|)
    comm_nodes = [n for n, h in enumerate(home) if h == target_comm]
    if not comm_nodes:
        return 1.0
    
    # Calculate all weights w_ki from k to community j
    w_ki = []
    for i in comm_nodes:
        edge = tuple(sorted([node, i]))
        w_ki.append((i, weights.get(edge, 0.0)))
    
    total_weight = sum(w for _, w in w_ki)
    if total_weight == 0:
        return 1.0
    
    # Sort weights descending to find smallest subset
    w_ki.sort(key=lambda x: x[1], reverse=True)
    
    # Find |C_k| - smallest subset containing 90% of weight
    acc_weight = 0
    for k, (_, weight) in enumerate(w_ki, 1):
        acc_weight += weight
        if acc_weight >= 0.9 * total_weight:
            # Return |C_k|/|C_j| as specified
            return k / len(comm_nodes)
    
    return 1.0

def edge_spreads(edges, home, weights):
    """Calculate edge spread S(k,l) = max{S(k → Ci), S(l → Cj)}"""
    spreads = []
    for k, l in edges:
        # Get communities of both nodes
        ci, cj = home[k], home[l]
        # Calculate both node spreads and take maximum
        spread = max(
            node_spread(k, cj, home, weights),  # S(k → Cj)
            node_spread(l, ci, home, weights)   # S(l → Ci)
        )
        spreads.append(spread)
    return spreads

def main():
 # Plain TVCM
    tvcm = SimpleTVCM(**KW)
    w_plain = simulate(tvcm, HOURS, DT)
    strong_p = strong_edges(w_plain, tvcm.home_of_node)
    s_plain = edge_spreads(strong_p, tvcm.home_of_node, w_plain)
    print(f"Plain TVCM: {len(strong_p)} strong links, mean spread={np.mean(s_plain):.2f}")

    # TVCM + Social Overlay
    so = SocialTVCM(tvcm)
    w_so = simulate(so, HOURS, DT)
    strong_so = strong_edges(w_so, so.home_of_node)
    s_so = edge_spreads(strong_so, so.home_of_node, w_so)
    print(f"TVCM+SO: {len(strong_so)} strong links, mean spread={np.mean(s_so):.2f}")

    # Plot histograms
    bins = np.linspace(0, 1, 11)
        
    plt.figure(figsize=(3.2, 2.7))
    plt.hist(s_plain, bins=bins, color="C0")
    plt.xlabel("Spread")
    plt.ylabel("# Links")
    plt.title(f"TVCM (m={np.mean(s_plain):.2f})")
    plt.tight_layout()
    plt.savefig("spread_TVCM.png")
    plt.close()

    plt.figure(figsize=(3.2, 2.7))
    plt.hist(s_so, bins=bins, color="C1")
    plt.xlabel("Spread")
    plt.ylabel("# Links")
    plt.title(f"TVCM-SO (m={np.mean(s_so):.2f})")
    plt.tight_layout()
    plt.savefig("spread_TVCM_SO.png")
    plt.close()
  

if __name__ == "__main__":
    main()
