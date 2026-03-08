# CSEP-561 Project 3: Mininet Bufferbloat

This directory contains starter and support code for project 3. You may want to
version control this file for your team and will need to update this readme file
with your own information in your final submission.

For details of the project itself, see the
[Project 3 page](https://courses.cs.washington.edu/courses/csep561/26wi/projects/project3/)
of the course website.

# Submission Details

Name: <br/>
UWNetID:

Name: <br/>
UWNetID:

## Instructions to reproduce your results:
  1. For Reno, run `sudo ./run.sh`
  2. For BBR, run `sudo ./run_bbr.sh`

## Answers to the questions:

### Part 2
  1. When q=20, the average fetch time is ~2.729s with a standard deviation of ~0.804s. When q=100, the average fetch time is ~13.350s with a standard deviation of ~3.576s.
  2. With a smaller buffer size (q=20), incoming packets will be discarded when the buffer is full and detected by Reno, causing Reno to slow down the sending rate. This explains why the sawtooth pattern appears. With q=100, the larger buffer means packets are rarely dropped, so Reno never receives a congestion signal and continues sending at full speed. As a result, the queue stays persistently full, causing queueing delay to accumulate and latency to continuously increase. We can observe that the RTT increases as the buffer fills up over time.
  3. (maximum) transmit queue length = 1000. The interface reports an MTU of 1500 bytes. (1000 packet * 1500 bytes/packet) / (100 Mb/s) = 0.12s
  4. For q=20, the RTT oscillates between 100ms and 300ms and presents a sawtooth pattern, reflecting TCP Reno's periodic congestion backoff. For q=100, the RTT steadily increases from 750ms to 1500ms without recovering. In general, a larger queue size leads to higher and more persistent RTT, because more packets are queued at the bottleneck, resulting in greater queueing delay.
  5. First thing I can come up with is keeping the buffer size small. Second idea is switch to a different algorithm like BBR.

### Part 3
  1. When q=20, the average fetch time is ~2.488s with a standard deviation of ~1.024s. When q=100, the average fetch time is ~2.023s with a standard deviation of ~0.555s.
  2. q=100 gives a lower fetch time. Compared to part 2 it's a significant improvement! The reason is that BBR does not rely on packet loss to detect bandwidth, so it would not occupy all of the queue.
  3. For q=20, the graphs are very similar between Reno and BBR. For q=100, BBR uses less queue size than Reno.
  4. Not really, as you can see at the tail of the graph of BBR, it jumps to the max number of the queue size. I think the reason is that BBR was doing bandwidth probing, so the queue became full during that process. Therefore, it does not fully resolve the bufferbloat.
