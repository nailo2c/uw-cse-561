# Part 1 of UWCSE's Mininet-SDN project2
#
# based on Lab Final from UCSC's Networking Class
# which is based on of_tutorial by James McCauley

from pox.core import core
import pox.openflow.libopenflow_01 as of
from pox.lib.addresses import IPAddr, IPAddr6, EthAddr

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


class Part3Controller(object):
    """
    A Connection object for that switch is passed to the __init__ function.
    """

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
        s1_route.actions.append(of.ofp_action_output(port=1))
        self.connection.send(s1_route)

        # If distination ip is 10.0.2.X, send to s2
        s2_route = of.ofp_flow_mod()
        s2_route.priority = 900
        s2_route.match.dl_type = 0x0800  # IPv4
        s2_route.match.nw_dst = IPAddr("10.0.2.20")
        s2_route.actions.append(of.ofp_action_output(port=2))
        self.connection.send(s2_route)

        # If distination ip is 10.0.3.X, send to s3
        s3_route = of.ofp_flow_mod()
        s3_route.priority = 900
        s3_route.match.dl_type = 0x0800  # IPv4
        s3_route.match.nw_dst = IPAddr("10.0.3.30")
        s3_route.actions.append(of.ofp_action_output(port=3))
        self.connection.send(s3_route)

        # If distination ip is 10.0.4.X, send to serv1
        serv1_route = of.ofp_flow_mod()
        serv1_route.priority = 900
        serv1_route.match.dl_type = 0x0800  # IPv4
        serv1_route.match.nw_dst = IPAddr("10.0.4.10")
        serv1_route.actions.append(of.ofp_action_output(port=4))
        self.connection.send(serv1_route)

        # If distination ip is 172.16.10.X, send to hnotrust
        hnotrust_route = of.ofp_flow_mod()
        hnotrust_route.priority = 900
        hnotrust_route.match.dl_type = 0x0800  # IPv4
        hnotrust_route.match.nw_dst = IPAddr("172.16.10.100")
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


def launch():
    """
    Starts the component
    """

    def start_switch(event):
        log.debug("Controlling %s" % (event.connection,))
        Part3Controller(event.connection)

    core.openflow.addListenerByName("ConnectionUp", start_switch)
