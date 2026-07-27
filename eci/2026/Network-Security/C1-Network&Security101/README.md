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

