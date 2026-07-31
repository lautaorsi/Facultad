# Layer 2 & Layer 3 Attacks

**Infrastructure Security**

-   Access
    - Control at port level

-   Distribution
    - Packet filtering

-   Core
    - Switch packets quickly

-   Server farm
    - Provide application services, includes network management system.


## Layer 2 Attack Methods

### 1. Mac Layer Attacks

-   **Mac Address Flooding:**

Frames w/ unique, invalid source MAC addresses flood the switch, exhausting content addressable memory table space, disallowing new entries from valid hosts.

> Mitigated by enabling Port Security. MAC address VLAN access maps.

---

### 2. VLAN Attacks

-   **VLAN Hopping**

By altering the VLAN ID on packets encapsultaed for trunking, an attacking device can send or receive packets on various VLANs, bypassing Layer 3 security measures.

> Mitigated by tightening up trunk configurations and the negotiation state of unused ports.

-   **Attacks between Devices on a Common VLAN**

Devices might need protection from one another, this is specially true on service-provider segments that support devices from multiple customers

> Mitigated by implementing private VLANs (PVLAN)

---

### 3. Spoofing Attacks

-   **DHCP Starvationd & DHCP Spoofing**

Attacking devices can exhaust the addres space available on the DHCP servers for a period of time or establish itself as DHCP server in Man-in-the-Middle attacks.

> Mitigated via DHCP snooping

-   **Spanning-tree Compromises**

Attacking device spoofs the root bridge in the STP

> Mitigated via DHCP snooping

-   **MAC Spoofing**

ATtacking device spoofs the MAC address of a valid host. The switch then forwards frames destined for the host to the attacking device

> Mitigated via DHCP snooping & port security

-   **Address Resolution Protocol Spoofing**

Attacking device crafts ARP replies intended for valid hosts. The attacking device MAC address then becomes the desitnation address found in the Layer 2 frames sent by the valid network device.

> Mitigated by using dynamic ARP inspection, DHCP snooping & port security.

---


### 4. Switch Device Attacks

- **Cisco Discovery Protocol Manipulation**

Inforamtion sent through CDP is trnasmitted in clear text and unauthenticated, allowing it to be captured and divulge network topology information.

> Mitigated by disabling CDP on all ports where it is not intentionally used.

- **Secure Shell Protocol and Telnet Attacks**

Telnet packets can be read in clear text. SSH is an option but has security issues in version 1

> Mitigated by using SSH version 2 & Telnet with vty ACLs

### Cam Overflow

Ffilling the cam table w/ invalid mac addresses makes the switch forward the messages to all linked devices.
It works only until CAM recod timeout and only allows for sniffing


### DHCP spoofing
DHCP is a protocol that assigns IP addresses to devices on a network. This can be exploited by either DDoS'ing the DHCP server or to attaching a DHCP server to the network and having it assume the role of DHCP server, thus giving false DHCP information.
