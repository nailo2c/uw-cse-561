# CSEP-561 Project 2

This directory contains starter and support code for project 2.

For details of the project itself, see [the Project 2 page of the course
website](https://courses.cs.washington.edu/courses/csep561/26wi/projects/project2/).

## Project Goals

One of our goals is to implement network policies that allow hosts to communicate according to the following table:

### Expected Ping Results (pingall)

| Source → Dest | h10 | h20 | h30 | hnotrust | serv1 |
|---------------|-----|-----|-----|----------|-------|
| **h10** | - | ✅ | ✅ | ❌ | ✅ |
| **h20** | ✅ | - | ✅ | ❌ | ✅ |
| **h30** | ✅ | ✅ | - | ❌ | ✅ |
| **hnotrust** | ❌ | ❌ | ❌ | - | ❌ |
| **serv1** | ✅ | ✅ | ✅ | ❌ | - |

- ✅ = Ping should succeed
- ❌ = Ping should fail (blocked by security policy)

### Security Policies

1. **Block ICMP from hnotrust**: hnotrust cannot send ICMP packets to internal hosts or servers
2. **Block hnotrust from datacenter**: hnotrust cannot access serv1 entirely (all traffic blocked)

### Expected Results

- **pingall**: 12/20 received, 40% dropped
- **iperf hnotrust1 h10**: Should succeed (TCP allowed)
- **iperf h10 serv1**: Should succeed
- **iperf hnotrust1 serv1**: Should fail (blocked)

### Expected Iperf Results

| Test | Expected Result | Reason |
|------|-----------------|--------|
| `iperf hnotrust1 h10` | ✅ Success | Only ICMP is blocked, TCP is allowed |
| `iperf h10 serv1` | ✅ Success | Normal traffic between trusted hosts |
| `iperf hnotrust1 serv1` | ❌ Timeout | All traffic from hnotrust to serv1 is blocked at dcs31 |
