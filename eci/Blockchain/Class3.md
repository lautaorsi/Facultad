# Consensus    

Fundamentals to achieve consensus: 
1. Agreement
2. Termination (having a limited time to make the choice) 
3. Validity (change being proposed by a node) 

## Byzantine consensus and Practical Byzantine Fault Tolerance

**Lemma:** There is no solution with fewer than 3m+1 generals to cope with m traitors (_the problem is solvable only if there is more than two thirds majority_) <br>
Solutions for this:
- **Oral message** is for the general **(C)** to tell each soldier **M** the command which then compare the messages with all the rest **(M-1)** and vote by majority
- **Signed Oral message** is where each message is signed by those who agree/send it, in this case a solution for the case of 3m generals exists.   

These protocols have a complexity due to one-to-one communication and the knowledge of every node in the network, it doesnt scale nor work in permissionless open networks, as well as not taking into account virtualization (unlimited "machines" for the attackers).

**_Gossip_** is the idea of adding every proposal to the chain individually and then keeping the longest chain, each new block lowers the probability of a malicious node trying to build another valid chain -> This can only be functional under the assumption of _Proof Of Work_ <br>
For this, we must be able to get new proposals faster than we can add blocks (otherwise the branch could become not the longest and reject the consensus), we must thus add a timeout for adding blocks

Proof of stakes implies that those who have a lot of computing power and a lot in stake won't act in a malicious way since that would compromise themselves

