# Extension 2 of UWCSE's Project 1
#
# based on Lab 4 from UCSC's Networking Class
# which is based on of_tutorial by James McCauley

from pox.core import core
import pox.openflow.libopenflow_01 as of

log = core.getLogger()

# Spanning Tree: block S2-S3 link to eliminate loop
# s2 (dpid=2): block port 3 (to s3)
# s3 (dpid=3): block port 2 (to s2)
BLOCKED_PORTS = {
    2: [3],
    3: [2],
}


class Firewall(object):
    """
    A Firewall object is created for each switch that connects.
    A Connection object for that switch is passed to the __init__ function.
    """

    def __init__(self, connection):
        self.connection = connection
        self.dpid = connection.dpid
        self.blocked_ports = BLOCKED_PORTS.get(self.dpid, [])

        connection.addListeners(self)

        # Install drop rules for blocked ports
        for port in self.blocked_ports:
            fm = of.ofp_flow_mod()
            fm.priority = 10000
            fm.match.in_port = port
            # No actions = drop
            self.connection.send(fm)
            print(f"Switch {self.dpid}: Blocked port {port} (spanning tree)")

    def _handle_PacketIn(self, event):
        packet = event.parsed
        if not packet.parsed:
            log.warning("Ignoring incomplete packet")
            return

        packet_in = event.ofp
        in_port = event.port

        # Drop packets from blocked ports (should not happen due to flow rule)
        if in_port in self.blocked_ports:
            return

        # Pure spanning tree: just flood (exclude blocked ports)
        msg = of.ofp_packet_out()
        msg.in_port = in_port

        if packet_in.buffer_id is not None and packet_in.buffer_id != -1:
            msg.buffer_id = packet_in.buffer_id
        else:
            msg.data = packet_in.data

        # Flood to all ports except in_port and blocked ports
        for port in self.connection.ports:
            if port >= of.OFPP_MAX:
                continue
            if port == in_port:
                continue
            if port in self.blocked_ports:
                continue
            msg.actions.append(of.ofp_action_output(port=port))

        self.connection.send(msg)
        print(f"Switch {self.dpid}: Flooding packet from port {in_port}")


def launch():
    """
    Starts the component
    """

    def start_switch(event):
        log.debug("Controlling %s" % (event.connection,))
        Firewall(event.connection)

    core.openflow.addListenerByName("ConnectionUp", start_switch)
