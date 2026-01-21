# Project 1: Mininet Topology and Firewall

## Overview
This project contains two parts exploring Mininet topologies and Software Defined Networking (SDN) controllers using POX.

## Setup Resources
- **Project Page:** [UW CSE 561 Project 1](https://courses.cs.washington.edu/courses/csep561/26wi/projects/project1/)
- **Mininet Environment:** [Setup Guide](https://gitlab.cs.washington.edu/561p-course-staff/mininet-environment)

---

## Part 1: Basic Topology

### Implementation Details
The `topos/part1.py` script defines the `part1_topo` class which builds a Star topology:
- **Switch:** `s1`
- **Hosts:** `h1`, `h2`, `h3`, `h4`
- **Links:** Each host is directly connected to the switch `s1`.

### Usage
To start the topology and enter the Mininet CLI:

```bash
sudo python3 topos/part1.py
```

### Sample Output
Results from `iperf` (bandwidth test) and `pingall` (reachability test):

```console
mininet> iperf h1 h4
*** Results: ['70.1 Gbits/sec', '70.5 Gbits/sec']

mininet> pingall
*** Results: 0% dropped (12/12 received)
```

---

## Part 2: Firewall Controller

### Implementation Details
This part introduces a **Remote Controller** implementing a firewall logic alongside a specific network topology.

#### Topology (`topos/part2.py`)
- **Switch:** `s1` (Connected to a RemoteController)
- **Hosts:**
  - `h1`: `10.0.1.2/24` (MAC: `00:00:00:00:00:01`)
  - `h2`: `10.0.0.2/24` (MAC: `00:00:00:00:00:02`)
  - `h3`: `10.0.0.3/24` (MAC: `00:00:00:00:00:03`)
  - `h4`: `10.0.1.3/24` (MAC: `00:00:00:00:00:04`)

#### Controller Logic (`pox/a1part2controller.py`)
The controller (`Firewall` class) installs OpenFlow rules to manage traffic:
1.  **Allow ICMP:** Priority 1000 (Action: Flood)
2.  **Allow ARP:** Priority 1000 (Action: Flood)
3.  **Drop IPv4:** Priority 10 (Action: Drop) - Drops all other IPv4 traffic (e.g., TCP/UDP).

### Usage

1.  **Start the Controller:**
    Open a terminal and run the POX controller:
    ```bash
    # Adjust the path to pox.py as needed
    sudo ~/pox/pox.py log.level --DEBUG project1.a1part2controller
    ```

2.  **Start the Topology:**
    In a separate terminal, start the Mininet topology:
    ```bash
    sudo python3 topos/part2.py
    ```

### Expected Behavior
- **Ping (ICMP):** Should succeed between all hosts.
- **Iperf (TCP):** Should fail (blocked by the drop rule).