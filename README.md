# TVCM Simulation

This repository implements the Time-Variant Community Model (TVCM) and its Social-Overlay (SO) extension to reproduce bridging-link behavior as described in 
"Putting Contacts into Context: Mobility Modeling beyond Inter-Contact Times"
By Theus Hossmann, Thrasyvoulos Spyropoulos, Franck Legendre ,Communication Systems Group, ETH Zürich, Switzerland; Mobile Communications, EURECOM, France.

The SO module synchronizes meetings of nodes from different communities outside their home locations, creating narrow, strong inter-community ties which reflect better an aspect of the nature of human mobility.

In real human mobility we don’t just bump into people at random , we routinely make synchronized, co-located meetings with members of our social circle outside of our “home” base, and these gatherings often produce the critical bridges that knit different communities together


## Features

- **TVCM Model:** Simulates node mobility with community structure.
- **Social Overlay:** Adds scheduled group meetings to the baseline mobility.
- **Strong Link Analysis:** Identifies strong inter-community links based on accumulated contact time.
- **Spread Metrics:** Measures how well strong links connect across communities.
- **Visualization:** Saves histograms of spread metrics for both models.

## Files

- `simulatIon.py` — Main simulation and analysis script.
- `TVCM_simple.py` — Baseline TVCM model.
- `TVCM_SO.py` — Social overlay extension for TVCM.


## Reproducibility

- The simulation uses a fixed random seed (`seed=42` in `simulatIon.py`) for reproducible results. Change or remove the seed for different random outcomes.

## Customization

- Adjust simulation parameters (number of nodes, communities, radio range, etc.) in the `KW` and `SO_KW` dictionaries in `simulatIon.py`.
- Modify the strong link threshold or spread calculation as needed.

