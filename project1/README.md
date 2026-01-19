# Project 1: Mininet Topology

## Overview
This project implements a basic Mininet topology consisting of a single switch connected to four hosts (Star topology).

## Setup Resources
- **Project Page:** [UW CSE 561 Project 1](https://courses.cs.washington.edu/courses/csep561/26wi/projects/project1/)
- **Mininet Environment:** [Setup Guide](https://gitlab.cs.washington.edu/561p-course-staff/mininet-environment)

## Implementation Details
The `part1.py` script defines the `part1_topo` class which builds the following network:
- **Switch:** `s1`
- **Hosts:** `h1`, `h2`, `h3`, `h4`
- **Links:** Each host is directly connected to the switch `s1`.

## Usage
To start the topology and enter the Mininet CLI:

```bash
sudo python3 part1.py
```

## Sample Output

Results from `iperf`, `dump`, and `pingall`:

```console
mininet> iperf h1 h4
*** Iperf: testing TCP bandwidth between h1 and h4 
*** Results: ['70.1 Gbits/sec', '70.5 Gbits/sec']

mininet> dump
<Host h1: h1-eth0:10.0.0.1 pid=15554> 
<Host h2: h2-eth0:10.0.0.2 pid=15559> 
<Host h3: h3-eth0:10.0.0.3 pid=15561> 
<Host h4: h4-eth0:10.0.0.4 pid=15563> 
<OVSSwitch s1: lo:127.0.0.1,s1-eth1:None,s1-eth2:None,s1-eth3:None,s1-eth4:None pid=15568> 
<Controller c0: 127.0.0.1:6653 pid=15547> 

mininet> pingall
*** Ping: testing ping reachability
h1 -> h2 h3 h4 
h2 -> h1 h3 h4 
h3 -> h1 h2 h4 
h4 -> h1 h2 h3 
*** Results: 0% dropped (12/12 received)
```
