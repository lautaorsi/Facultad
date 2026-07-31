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





> **Heterogeneity** <br> Having differences in data, models, resources, or participation. Not necessarily distributional

## Skew types

### 1. Data skew

Having too big of a spread on the amount of data collected by each label

### 2. Label skew 

Different labels between models, non convergence is pretty much guaranteed unless there is some big intersection

### 3. Attribute Skew

-   **Distribution based**
    -   Same aatributes, different distributions 
-   **Partial attribute selection**
    -   Clients observe an incomplete usbset of attributes for their samples
-  **Noisy attributes**
    -   Attribute values are corrupted or noisy for some clients
-   **Vertical skew**
    -   Different clinets hold different attribute subsets for the same samples (vertical FL)

### 4. Participation Skew

It reflects how often and under what conditions clients take part in training or evaluation.

Two forms:
-   **Party selection and subsampling.** Strategic or random selection of a subset of clients each round.
-   **Client dropout.** Clients unexepectedly disconnect mid-round because of network issues, batery or other constraints.


### 5. Modality skew

Clients hold data from sutrcturally different input modalities







FedAvg was derived under IID data 
