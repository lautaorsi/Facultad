# Foundations


Basic Notation
----
-   $K$ parties <br>
-   $D_K$ dataset of party $K$ <br> 
-   $D$ centralized dataset <br>
-   $D_i = \{(x_1,y_1)...(x_n,y_n)\}$ => y label


IID Independent and Identically Distributed




## Centralized Learning
-   Clients send local data to unified dataset
-   Standard AI/ML assumes all data can be gathered

Benefits
---
Statistical Control
Straightforward debugging / reproducibility
Scales cleanly with compute
Mature tools

Problems
---
Privacy/Security of data
Regulations
Storage
Server bottleneck





## Federated Learning

-   Clients send small local models 
-   The server combines them into one common model
-   Process is repeated until it converges

Example: GBoard, voice assistance, banks, etc

### Agregation Functions
AGregation functions represent methods to combine model weights into global models.

Aggregation ensures balanced learning from diverse data


### Architecture scenarios



| Type      | Number of clients | Dataset size | Availability & Reliability | Chances of Malicious behavior |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Site      | High (e.g. Cellphones) | Small       | Limited       | High       |
| Silo   | Small (e.g. Hospitals)        | Medium        | High        | Low        |

### Cross Site 

-   Massive number of parties
-   Small dataset
-   Limited availability and reliability
-   High chances of malicious behaviour

### Cross Silo
-   Smaller number of parties
-   Medium dataset
-   Higher availability and reliability
-   Lower chances of malicious behaviour




FedSGD

Caclulate averages of weights


