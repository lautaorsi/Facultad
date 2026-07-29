# Non-IID 

Non Independent and Identically Distributed datasets can post a serious issue when trying to perform Federated Learning, luckily there are some workarounds that mitigate a lot of the issues.


###   Centralization: 

All samples get pooled and shuffled, this aims to mitigate skewness in the client's dataset size thus making each individual model more "reliable"

### Issues
Overrepresentation for clients with small datasets
Noisier votes from small clients
local overfitting 

### Solution
Calculating a weighted sum divided by how much data it produced  




**Heterogeneity** 

Diffenreces in data, models, resources, or participation. Not necessarily distributional

## Skewdeness types

### Data skew

Having too big of a spread on the amount of data collected by each label

### Label skew 

Different labels between models, non convergence is pretty much guaranteed unless there is some big intersection



FedProx
---
Given a client $k$ at round $t$

Add a penalty for wandering, 

$$\min_{w}{F_k(w)}$$
$$min_{w}{h_{k}(w:w^{t}) = F_{k}(w) + \frac{\mu}{2} ||w-w^{t}||^2}$$


