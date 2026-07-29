# Personalization & Client Selection

Real-world clients are trivially non-IID, label distributions diverge which causes
-   client drift
-   aggregated updates pointing to inconsistent directions
-   Slower convergence speed
-   Global accuracy degreades

## Personalized Federated Learning

Personalized Federated Learning _(PFL)_ is the idea of clustering clients and training one model per cluster, the three main families are:

1.  **Regularization-based**, keeping local objectives close to the global one
    -   FedProx
    -   FedOpt/FedAgrad/FedYogi/FedAdam
    -   FedAvgM
    -   FedNova
    -   FedDyn
2.  **Selection-based**, prioritizing or filter which clients participate
    -   Power-of-Choice
    -   HACCS
    -   FedCLS
    -   FedCor
3.  **Clustering-based**, partitioning clients into distributionally homogenous groups and trian one model per group
    -   CFL
    -   FedSoft
    -   Clust-PSI-PFL


### Clust-PSI-PFL

**Idea** <br>
Use the WPSI $^t$ metric to measure non-IIDness and cluster clients into distributionally homogenous groups, training one FedAvg model per cluster.

It directly tackles the issue by quantifying the right thing, _distributional drift_.

**Pipeline**
1.  Collect counts per-client label frequency
2.  Compute PSI
3.  Cluster using K-Means++ on PSI
4.  Train & aggregate one FedAvg model per cluster

**Limitations**
-   No built-in privacy
    -   Label-count sharing leaks information.
-   Scoped to label skew
    -   PSI features are class-frequency based, if volumes differ clusters may be suboptimal
-   Small-client noise
    -   Clients with few samples have noisy PSI signatures


### Cherry-Picking within clusters

Each round a subset of clients should be selected for the model to train on, but how?

The idea is to cluster clients by label-distribution similarity (Hellinger distance) and then within the highest-loss clusters, pick the highest-loss clients, this speeds convergence.

### Price on data: the PDE value function proposal

FedLECC's proposes a valuation for client i:

$$ V_i = \alpha \cdot \Delta Acc_i - \beta \cdot CommCost_i + \gamma \cdot RobustnessGain_i $$

Where

-   $\Delta Acc_i$ Marginal increase in global accuracy when client participates
-   $CommCost_i$ Normalized bandwith/latency cost of involving client i
-   $RobustnessGain_i$ Improvement under adversarial/high non-IID conditions attributable to i's data
-   $\alpha, \beta, \gamma$: Weights reflecting the target application's priorities 