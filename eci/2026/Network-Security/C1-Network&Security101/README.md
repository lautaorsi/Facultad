# Network & Security 101



> ## Index
> 1. [Network Stack](#1-layered-models)
>     * [Layered Model](#12-tcpip)
> 2. [Cryptology](#2-cryptology)
>     * [Four elements of Securing Communications](#22-four-elements-of-securing-communications)
---




## 1. Layered Models

Computer networks are complex systems containing modular layers



### 1.2. TCP/IP

- **Physical layer:**
  > The transceiver that drives the signals on the network

- **Data Link Layer (MAC):**
  > Responsible of creating the frames that move across the network

- **Network Layer (IP):**
  > Responsible for creating the packets that move across the network

- **Transport Layer _(TCP/UDP)_:**
  > Responsible for establishing the connection between aplications on different hosts

- **Application Layer:**
  > Group of applications requiring network communication


# 2. Cryptology

### 2.2. Four elements of Securing Communications

- **Data Integrity:**
  > Guarantees that the message was not altered
- **Origin Authenticity:**
  > Guarantees that the message is not a forgery
- **Data Confidentiality:**
  > Guarantees only the authorized user can access it
- **Data Non-Repudiation:**
  > Guarantees that the sender cannot refute the validity of the message



### 2.3. Assymetric Encryption Algorithms


Diffie-Hellman Algorithm <br>
Exchange is a secret key exchange protocol over an insecure communication channel



### 2.4. Hash Guessing

Given a $k$-long combination of $n$ alphabet characters 

Brute Force: 
Trying all possible combinations of characters with k length.

$ P = n^k$

Dictionary: 

Smallest set of candidates (passwords that "make sense")
Derived from the language
Somebody else used them as password


Rule Based:

Generate passwords based on some pattern
regexx, grammar, Markov