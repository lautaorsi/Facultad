# Fairness

_Does the global model work well for every person?_

There are 2 kinds of fairness, Group and Client level, which do NOT imply the other one is satisfied. 

**Client fairness** tackles the sustainability of the federation, ensuring that Non-IIDness does not impact the model's construiction.<br>
**Group fairness** aims to protect the people the model is applied to, ensuring the demographic group is treated well by the model.


## Measuring Fairness

### 1. Demographic Parity

**Criterion** - The decision is independent of the protected attribute.

$$Pr\{Ŷ = 1 | A = 0\} = Pr\{Ŷ = 1 | A = 1\}$$

**Measure** - The absolute gap in positive-prediction rate

$$DP = |Pr \{ Ŷ = 1 | A = 0 \} - Pr \{ Ŷ = 1 | A = 1\}$$

---

### 2. Equal Opportunity Difference

**Criterion** - The favourable outcome is equally reachable by qualified members of either group

$$Pr\{Ŷ = 1  | A = 0, Y = 1\} = Pr\{Ŷ = 1 | A = 1, Y = 1\}$$

**Measure**

$$EOD = \Delta TPR = | TPR_1 + TPR_0 |$$

---

### 3. Average Odds (Primary target)

**Criterion** - Equality must hold on both sides of the label

$$Pr\{Ŷ = 1  | A = 0, Y = y\} = Pr\{Ŷ = 1 | A = 1, Y = y\}$$

**Measure** - The mean absolute deviation of the two rates

$$AO = \frac{1}{2} (|TPR_1 - TPR_0| + |FPR_1 - FPR_0|)$$

---


Purely local debiasing

Apply a centralised techinque independently on each device. Nothing beyond model updates leave the device.

Server-coordinated group fairness


