# Part 2 of UWCSE's Project 3
#
# based on Lab 4 from UCSC's Networking Class
# which is based on of_tutorial by James McCauley

from pox.core import core
import pox.openflow.libopenflow_01 as of

log = core.getLogger()


class Firewall(object):
    """
    A Firewall object is created for each switch that connects.
    A Connection object for that switch is passed to the __init__ function.
    """

    def __init__(self, connection):
        # Keep track of the connection to the switch so that we can
        # send it messages!
        self.connection = connection

        # This binds our PacketIn event listener
        connection.addListeners(self)

        # add switch rules here
        # Rule 1: ICMP accept
        icmp = of.ofp_flow_mod()
        icmp.priority = 1000
        icmp.match.dl_type = 0x0800 # IPv4
        icmp.match.nw_proto = 1 # Memo: ICMP=1, TCP=6, UDP=17
        icmp.actions.append(of.ofp_action_output(port=of.OFPP_FLOOD))
        connection.send(icmp)

        # Rule 2: ARP accept
        arp = of.ofp_flow_mod()
        arp.priority = 1000
        arp.match.dl_type = 0x0806  # ARP
        arp.actions.append(of.ofp_action_output(port=of.OFPP_FLOOD))
        connection.send(arp)

        # Rule 3: Drop other IPv4 packet
        drop = of.ofp_flow_mod()
        drop.priority = 10
        drop.match.dl_type = 0x0800
        connection.send(drop)

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
        print("Unhandled packet :" + str(packet.dump()))


def launch():
    """
    Starts the component
    """

    def start_switch(event):
        log.debug("Controlling %s" % (event.connection,))
        Firewall(event.connection)

    core.openflow.addListenerByName("ConnectionUp", start_switch)
