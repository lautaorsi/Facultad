## Tech Ingredients

- Rpc technology allows clients to trust full nodes so as to not verify the whole blockchain (implies trusting third parties), the advantage of using this is simplicity and the fact that you can always check locally by downloading the actual blockchain.
- Art forks are caused by a difference in the consensus system
- Blockchain addresses are basically asymmetric public keys
- Mining bitcoin is trying to solve a puzzle, whoever solves the puzzle the fastest gets the "leader" status and can thus add that to the blockchain (and rewarded)
- Identity is protected by your private key, but records of account movements are public

### Proving a block's authenticity
Using a **_Merkle Tree_** allows validating a transaction as a light node, it requires asking for proof from a full node by giving your hash function and verifying that the root from the full node's merke tree is the same as the one you know. <br>

### Failures
Links are reliable <br>
Nodes are not, they are prone to crashes and/or behave in malicious ways (note: they cannot behave against the protocol as it would be discarded, they attack by exploiting the protocol)

### Message passing
1. Synchronous, where messages while be delivered in a fixed amount of time, protocols can be organized in rounds and nodes have a time limit to reply.
- We have a f+1 rounds consensus algorithm that can tolerate f failures (assuming f less than node amnt)
2. Asychronous, where there is no time limit on receiving, processing or responding to messages.
- When using async messaging, a single failure makes it impossible to reach consensus.
 




