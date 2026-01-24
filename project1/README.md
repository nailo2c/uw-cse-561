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

---

## Extension 1: Learning Switch (MAC Learning)

### Implementation Details
The controller in `pox/a1ext1controller.py` implements a basic learning switch:
- **MAC learning:** Records `src MAC -> in_port` on each `PacketIn`.
- **Known destination:** Installs a unicast flow (priority 1000) and forwards out the learned port.
- **Unknown destination:** Floods the packet.

### Usage
1.  **Start the Controller:**
    ```bash
    sudo ~/pox/pox.py log.level --DEBUG project1.a1ext1controller
    ```
2.  **Start a Topology (single-switch):**
    ```bash
    # Example
    sudo python3 topos/part2.py
    ```

### Expected Behavior
- First packet to an unknown destination is flooded; subsequent traffic is unicast via installed flows.
- `pingall` should succeed.

---

## Extension 2: Spanning Tree + Loop-Free Flooding

### Implementation Details
- **Topology:** `topos/ext2.py` defines three switches (`s1`, `s2`, `s3`) with redundant inter-switch links plus four hosts.
- **Controller:** `pox/a1ext2controller.py` uses `openflow.discovery` to learn links, computes a spanning tree (BFS from lowest DPID), and installs high-priority drop rules on non-tree inter-switch ports. It also avoids flooding LLDP.

### Usage
1.  **Start the Controller:**
    ```bash
    sudo ~/pox/pox.py log.level --DEBUG project1.a1ext2controller
    ```
2.  **Start the Topology:**
    ```bash
    sudo python3 topos/ext2.py
    ```

### Expected Behavior
- End hosts can reach each other (e.g., `pingall` succeeds).
- Redundant inter-switch links are blocked to prevent loops/broadcast storms.
