# Part 2 of UWCSE's Mininet-SDN project2
#
# based on Lab Final from UCSC's Networking Class
# which is based on of_tutorial by James McCauley

from pox.core import core
import pox.openflow.libopenflow_01 as of
from pox.lib.addresses import IPAddr, IPAddr6, EthAddr

from pox.lib.packet.arp import arp
from pox.lib.packet.ethernet import ethernet

log = core.getLogger()

# Convenience mappings of hostnames to ips
IPS = {
    "h10": "10.0.1.10",
    "h20": "10.0.2.20",
    "h30": "10.0.3.30",
    "serv1": "10.0.4.10",
    "hnotrust": "172.16.10.100",
}

# Convenience mappings of hostnames to subnets
SUBNETS = {
    "h10": "10.0.1.0/24",
    "h20": "10.0.2.0/24",
    "h30": "10.0.3.0/24",
    "serv1": "10.0.4.0/24",
    "hnotrust": "172.16.10.0/24",
}


class Part4Controller(object):
    """
    A Connection object for that switch is passed to the __init__ function.
    """
    # ip_info = {$ip: {"mac": $mac, "port": $port}}
    ip_info = {}

    def __init__(self, connection):
        print(connection.dpid)
        # Keep track of the connection to the switch so that we can
        # send it messages!
        self.connection = connection

        # This binds our PacketIn event listener
        connection.addListeners(self)
        # use the dpid to figure out what switch is being created
        if connection.dpid == 1:
            self.s1_setup()
        elif connection.dpid == 2:
            self.s2_setup()
        elif connection.dpid == 3:
            self.s3_setup()
        elif connection.dpid == 21:
            self.cores21_setup()
        elif connection.dpid == 31:
            self.dcs31_setup()
        else:
            print("UNKNOWN SWITCH")
            exit(1)

    def s1_setup(self):
        # Allow all traffic to h10
        ip_allow = of.ofp_flow_mod()
        ip_allow.priority = 900
        ip_allow.match.dl_type = 0x0800  # IPv4
        ip_allow.actions.append(of.ofp_action_output(port=of.OFPP_FLOOD))
        self.connection.send(ip_allow)

    def s2_setup(self):
        # Allow all traffic to h20
        ip_allow = of.ofp_flow_mod()
        ip_allow.priority = 900
        ip_allow.match.dl_type = 0x0800  # IPv4
        ip_allow.actions.append(of.ofp_action_output(port=of.OFPP_FLOOD))
        self.connection.send(ip_allow)

    def s3_setup(self):
        # Allow all traffic to h30
        ip_allow = of.ofp_flow_mod()
        ip_allow.priority = 900
        ip_allow.match.dl_type = 0x0800  # IPv4
        ip_allow.actions.append(of.ofp_action_output(port=of.OFPP_FLOOD))
        self.connection.send(ip_allow)

    def cores21_setup(self):
        # Block all ICMP traffic from hnotrust
        icmp_drop = of.ofp_flow_mod()
        icmp_drop.priority = 1000
        icmp_drop.match.dl_type = 0x0800  # IPv4
        icmp_drop.match.nw_proto = 1  # Memo: ICMP=1, TCP=6, UDP=17
        icmp_drop.match.nw_src = IPAddr("172.16.10.100") # hnotrust's IP
        self.connection.send(icmp_drop)

        # If distination ip is 10.0.1.X, send to s1
        s1_route = of.ofp_flow_mod()
        s1_route.priority = 900
        s1_route.match.dl_type = 0x0800  # IPv4
        s1_route.match.nw_dst = IPAddr("10.0.1.10")
        s1_route.actions.append(of.ofp_action_dl_addr.set_dst(EthAddr("00:00:00:00:00:01")))
        s1_route.actions.append(of.ofp_action_output(port=1))
        self.connection.send(s1_route)

        # If distination ip is 10.0.2.X, send to s2
        s2_route = of.ofp_flow_mod()
        s2_route.priority = 900
        s2_route.match.dl_type = 0x0800  # IPv4
        s2_route.match.nw_dst = IPAddr("10.0.2.20")
        s2_route.actions.append(of.ofp_action_dl_addr.set_dst(EthAddr("00:00:00:00:00:02")))
        s2_route.actions.append(of.ofp_action_output(port=2))
        self.connection.send(s2_route)

        # If distination ip is 10.0.3.X, send to s3
        s3_route = of.ofp_flow_mod()
        s3_route.priority = 900
        s3_route.match.dl_type = 0x0800  # IPv4
        s3_route.match.nw_dst = IPAddr("10.0.3.30")
        s3_route.actions.append(of.ofp_action_dl_addr.set_dst(EthAddr("00:00:00:00:00:03")))
        s3_route.actions.append(of.ofp_action_output(port=3))
        self.connection.send(s3_route)

        # If distination ip is 10.0.4.X, send to serv1
        serv1_route = of.ofp_flow_mod()
        serv1_route.priority = 900
        serv1_route.match.dl_type = 0x0800  # IPv4
        serv1_route.match.nw_dst = IPAddr("10.0.4.10")
        serv1_route.actions.append(of.ofp_action_dl_addr.set_dst(EthAddr("00:00:00:00:00:04")))
        serv1_route.actions.append(of.ofp_action_output(port=4))
        self.connection.send(serv1_route)

        # If distination ip is 172.16.10.X, send to hnotrust
        hnotrust_route = of.ofp_flow_mod()
        hnotrust_route.priority = 900
        hnotrust_route.match.dl_type = 0x0800  # IPv4
        hnotrust_route.match.nw_dst = IPAddr("172.16.10.100")
        hnotrust_route.actions.append(of.ofp_action_dl_addr.set_dst(EthAddr("00:00:00:00:00:05")))
        hnotrust_route.actions.append(of.ofp_action_output(port=5))
        self.connection.send(hnotrust_route)

    def dcs31_setup(self):
        # Block all traffic from hnotrust
        ip_drop = of.ofp_flow_mod()
        ip_drop.priority = 1000
        ip_drop.match.dl_type = 0x0800  # IPv4
        ip_drop.match.nw_src = IPAddr("172.16.10.100") # hnotrust's IP
        self.connection.send(ip_drop)

        # Allow all other traffic to serv1
        ip_allow = of.ofp_flow_mod()
        ip_allow.priority = 900
        ip_allow.match.dl_type = 0x0800  # IPv4
        ip_allow.actions.append(of.ofp_action_output(port=of.OFPP_FLOOD))
        self.connection.send(ip_allow)

    # used in part 4 to handle individual ARP packets
    # not needed for part 3 (USE RULES!)
    # causes the switch to output packet_in on out_port
    def resend_packet(self, packet_in, out_port):
        msg = of.ofp_packet_out()
        msg.data = packet_in
        action = of.ofp_action_output(port=out_port)
        msg.actions.append(action)
        self.connection.send(msg)

    def _handle_PacketIn(self, event):
        """
        Packets not handled by the router rules will be
        forwarded to this method to be handled by the controller
        """

        packet = event.parsed  # This is the parsed packet data.
        if not packet.parsed:
            log.warning("Ignoring incomplete packet")
            return

        packet_in = event.ofp  # The actual ofp_packet_in message.
        print(
            "Unhandled packet from " + str(self.connection.dpid) + ":" + packet.dump()
        )

        if packet.type == packet.ARP_TYPE:
            arp_packet = packet.payload
            sender_ip = arp_packet.protosrc
            sender_mac = arp_packet.hwsrc
            in_port = event.port
            self.ip_info[sender_ip] = {"mac": sender_mac, "port": in_port}

            # Handle ARP request to get the gateway's MAC address
            if arp_packet.opcode == 1: # ARE request
                target_ip = arp_packet.protodst
                if str(target_ip).endswith(".1"): # if target_ip is in the same subnet as one of the hosts, reply with that host's MAC
                    arp_reply = arp()
                    arp_reply.opcode = arp.REPLY
                    arp_reply.hwsrc = EthAddr("00:00:00:00:00:99") # Fake Gateway MAC
                    arp_reply.hwdst = sender_mac      # sender MAC, e.g. h10's MAC
                    arp_reply.protosrc = target_ip    # Gateway's IP
                    arp_reply.protodst = sender_ip    # Sender's IP

                    eth = ethernet()
                    eth.type = ethernet.ARP_TYPE
                    eth.src = EthAddr("00:00:00:00:00:99")
                    eth.dst = sender_mac
                    eth.payload = arp_reply

                    self.resend_packet(eth.pack(), in_port)

            elif arp_packet.opcode == 2: # ARE reply
                pass
        elif packet.type == packet.IP_TYPE:
            ip_packet = packet.payload
            sender_ip = ip_packet.srcip
            receiver_ip = ip_packet.dstip
            sender_mac = packet.src
            in_port = event.port
            self.ip_info[sender_ip] = {"mac": sender_mac, "port": in_port}

            if receiver_ip in self.ip_info:
                dst_port = self.ip_info[receiver_ip]["port"]
                dst_mac = self.ip_info[receiver_ip]["mac"]

                # learn flow rule
                fm = of.ofp_flow_mod()
                fm.priority = 1000
                fm.match.nw_dst = receiver_ip
                fm.match.dl_type = 0x0800
                fm.actions.append(of.ofp_action_dl_addr.set_dst(dst_mac))
                fm.actions.append(of.ofp_action_output(port=dst_port))
                self.connection.send(fm)

                # resend the packet
                msg = of.ofp_packet_out()
                msg.data = packet_in
                msg.actions.append(of.ofp_action_dl_addr.set_dst(dst_mac))
                msg.actions.append(of.ofp_action_output(port=dst_port))
                self.connection.send(msg)

        else:
            print("Unknown packet type: " + str(packet.type))
            return

def launch():
    """
    Starts the component
    """

    def start_switch(event):
        log.debug("Controlling %s" % (event.connection,))
        Part4Controller(event.connection)

    core.openflow.addListenerByName("ConnectionUp", start_switch)
