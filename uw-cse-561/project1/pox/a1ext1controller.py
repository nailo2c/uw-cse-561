# Extension 1 of UWCSE's Project 1
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

        # extension 1
        self.mac_to_port = {}

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

        # extension 1
        src_mac = packet.src
        dst_mac = packet.dst
        in_port = event.port

        # backwards learning
        self.mac_to_port[src_mac] = in_port

        if dst_mac in self.mac_to_port:
            out_port = self.mac_to_port[dst_mac]

            fm = of.ofp_flow_mod()
            fm.priority = 1000
            fm.match.dl_dst = dst_mac
            fm.actions.append(of.ofp_action_output(port=out_port))
            self.connection.send(fm)

            msg = of.ofp_packet_out()
            msg.in_port = in_port
            msg.actions.append(of.ofp_action_output(port=out_port))
            if packet_in.buffer_id is not None and packet_in.buffer_id != -1:
                msg.buffer_id = packet_in.buffer_id
            else:
                msg.data = packet_in.data
            self.connection.send(msg)
            print((f"Installed flow for {dst_mac} -> port {out_port}"))
        else:
            out_port = of.OFPP_FLOOD
            msg = of.ofp_packet_out()

            if packet_in.buffer_id is not None and packet_in.buffer_id != -1:
                msg.buffer_id = packet_in.buffer_id
            else:
                msg.data = packet_in.data

            msg.in_port = in_port
            msg.actions.append(of.ofp_action_output(port=out_port))
            self.connection.send(msg)
            print(f"Flooding packet for {dst_mac}")


def launch():
    """
    Starts the component
    """

    def start_switch(event):
        log.debug("Controlling %s" % (event.connection,))
        Firewall(event.connection)

    core.openflow.addListenerByName("ConnectionUp", start_switch)
