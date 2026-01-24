# Extension 2 of UWCSE's Project 3
#
# based on Lab 4 from UCSC's Networking Class
# which is based on of_tutorial by James McCauley

from pox.core import core
import pox.openflow.libopenflow_01 as of

from collections import defaultdict, deque

log = core.getLogger()


STP_DROP_PRIORITY = 10000
STP_DROP_COOKIE_BASE = 0x53545000  # 'STP\0' style marker (low 32 bits used)


def _is_lldp(pkt):
    """
    Return True if this is an LLDP packet (used by openflow.discovery).
    We must NOT flood LLDP or we'll confuse discovery/topology.
    """
    try:
        # POX packet has .find() helper
        return pkt.find('lldp') is not None
    except Exception:
        return False


class Switch(object):
    """
    Per-switch handler.
    We do:
    - Install/remove drop rules on blocked inter-switch ports
    - For PacketIn: flood only to allowed ports (excluding in_port)
    """

    def __init__(self, connection, controller):
        self.connection = connection
        self.dpid = connection.dpid
        self.ctrl = controller
        self.blocked_ports = set()
        self.stable_tree = False
        connection.addListeners(self)
        log.info("Switch connected dpid=%s", self.dpid)

    def _drop_cookie(self, port_no):
        # Cookie used to identify our own drop rules
        return (STP_DROP_COOKIE_BASE << 32) | (port_no & 0xffffffff)

    def apply_blocked_ports(self, new_blocked_ports):
        """
        Update drop rules so that ports in new_blocked_ports are blocked (drop),
        and ports no longer blocked have their drop rules removed.
        """
        new_blocked_ports = set(new_blocked_ports)
        to_add = new_blocked_ports - self.blocked_ports
        to_del = self.blocked_ports - new_blocked_ports

        # Remove old drop rules
        for port in sorted(to_del):
            fm = of.ofp_flow_mod()
            fm.command = of.OFPFC_DELETE
            fm.match = of.ofp_match(in_port=port)
            # Best-effort: restrict by cookie if supported by your OF1.0 impl.
            # POX supports cookie_mask fields in ofp_flow_mod.
            fm.cookie = self._drop_cookie(port)
            fm.cookie_mask = 0xffffffffffffffff
            self.connection.send(fm)
            log.info("dpid=%s: UNBLOCK port=%s", self.dpid, port)

        # Add new drop rules
        for port in sorted(to_add):
            fm = of.ofp_flow_mod()
            fm.priority = STP_DROP_PRIORITY
            fm.match = of.ofp_match(in_port=port)
            fm.cookie = self._drop_cookie(port)
            # No actions => drop
            self.connection.send(fm)
            log.info("dpid=%s: BLOCK port=%s (not on spanning tree)", self.dpid, port)

        self.blocked_ports = new_blocked_ports

    def _handle_PacketIn(self, event):
        pkt = event.parsed
        if not pkt.parsed:
            return

        # Ignore LLDP frames (discovery)
        if _is_lldp(pkt):
            return

        in_port = event.port

        # If controller already decided this in_port is blocked, drop (safety)
        if in_port in self.blocked_ports:
            return

        # Flood only on allowed ports (tree ports + host-facing ports)
        allowed_ports = self.ctrl.get_allowed_ports(self.dpid)

        msg = of.ofp_packet_out()
        msg.in_port = in_port

        if event.ofp.buffer_id is not None and event.ofp.buffer_id != -1:
            msg.buffer_id = event.ofp.buffer_id
        else:
            msg.data = event.ofp.data

        for port_no in sorted(allowed_ports):
            # Skip special/non-physical ports
            if port_no >= of.OFPP_MAX:
                continue
            if port_no == in_port:
                continue
            msg.actions.append(of.ofp_action_output(port=port_no))

        self.connection.send(msg)


class SpanningTreeController(object):
    """
    Global controller:
    - Track switches and discovery links
    - Compute spanning tree
    - Push per-switch blocked ports updates
    """

    def __init__(self):
        self.switches = {}  # dpid -> Switch instance

        # Link map from discovery:
        # (dpid, port) -> (dpid2, port2)
        self.link_out = {}  # directed view

        # For each switch, the set of inter-switch ports we know about
        self.interswitch_ports = defaultdict(set)

        # For each switch, the set of ports that are ON the spanning tree (interswitch only)
        self.tree_ports = defaultdict(set)

        core.openflow.addListenerByName("ConnectionUp", self._handle_ConnectionUp)

        # Listen to discovery LinkEvent when the component is ready
        core.call_when_ready(self._attach_discovery, ['openflow_discovery'])

    def _attach_discovery(self):
        core.openflow_discovery.addListenerByName("LinkEvent", self._handle_LinkEvent)
        log.info("Attached to openflow.discovery LinkEvent")

    def _handle_ConnectionUp(self, event):
        sw = Switch(event.connection, self)
        self.switches[event.dpid] = sw

        # After a new switch connects, recompute tree and apply rules
        self._recompute_and_apply()

    def _handle_LinkEvent(self, event):
        """
        Discovery reports a directed link with ports:
          event.link.dpid1, event.link.port1  -> event.link.dpid2, event.link.port2
        """
        l = event.link
        a = (l.dpid1, l.port1)
        b = (l.dpid2, l.port2)

        if event.added:
            self.link_out[a] = b
            # Mark both endpoints as inter-switch ports
            self.interswitch_ports[l.dpid1].add(l.port1)
            self.interswitch_ports[l.dpid2].add(l.port2)
        elif event.removed:
            # Remove directed mapping if exists
            if a in self.link_out and self.link_out[a] == b:
                del self.link_out[a]
            # Keep interswitch_ports as "known" (optional); it’s safe either way.
            # If you want strict behavior, you can also remove ports here.

        self._recompute_and_apply()

    def _build_undirected_edges(self):
        """
        Convert directed link map into undirected edges with port numbers on both sides.
        Returns a dict:
          adj[u] = list of (v, u_port, v_port)
        For parallel links, we keep them as distinct edges, but BFS will deterministically pick
        the lowest (u_port, v_port) when exploring neighbors.
        """
        adj = defaultdict(list)

        # Only add edge when both directions are known (more stable)
        for (u, up), (v, vp) in list(self.link_out.items()):
            # Check if reverse exists
            rev = (v, vp)
            if rev not in self.link_out:
                continue
            if self.link_out[rev] != (u, up):
                continue

            # Add undirected representation (u<->v). We will add both adjacency directions.
            adj[u].append((v, up, vp))
            adj[v].append((u, vp, up))

        # Sort adjacency lists for deterministic BFS (important with parallel links)
        for u in adj:
            adj[u].sort(key=lambda t: (t[0], t[1], t[2]))
        return adj

    def _compute_spanning_tree_ports(self):
        """
        Compute spanning tree ports sets (tree_ports[dpid] = set of interswitch ports on the tree).
        Root = smallest DPID in the discovered switch graph.
        BFS on unweighted graph.
        """
        adj = self._build_undirected_edges()
        if not adj:
            # No inter-switch links discovered yet
            self.tree_ports = defaultdict(set)
            self.stable_tree = False
            return

        root = min(adj.keys())
        parent = {root: None}          # node -> parent node
        parent_port = {}               # node -> (node_port_to_parent, parent_port_to_node)

        q = deque([root])
        while q:
            u = q.popleft()
            for (v, u_port, v_port) in adj[u]:
                if v in parent:
                    continue
                parent[v] = u
                parent_port[v] = (u_port, v_port)
                q.append(v)

        # Build tree_ports from parent relations
        tree_ports = defaultdict(set)
        for v, u in parent.items():
            if u is None:
                continue
            u_port_to_v, v_port_to_u = parent_port[v]
            tree_ports[u].add(u_port_to_v)
            tree_ports[v].add(v_port_to_u)

        self.tree_ports = tree_ports
        self.stable_tree = True
        log.info("Spanning tree computed: root dpid=%s, nodes=%s", root, sorted(parent.keys()))

    def _recompute_and_apply(self):
        """
        Recompute ST and apply blocked ports to all connected switches.
        """
        self._compute_spanning_tree_ports()

        for dpid, sw in self.switches.items():
            inter = set(self.interswitch_ports.get(dpid, set()))
            on_tree = set(self.tree_ports.get(dpid, set()))

            if not self.stable_tree:
                # Discovery / topology not ready -> don't block anything yet
                sw.apply_blocked_ports(set())
            else:
                blocked = inter - on_tree
                sw.apply_blocked_ports(blocked)

    def get_allowed_ports(self, dpid):
        """
        Return ports that are allowed for flooding:
        - all physical ports on the switch
        - except ports we decided to block (non-tree inter-switch ports)
        """
        sw = self.switches.get(dpid)
        if sw is None:
            return set()

        # All ports that OpenFlow connection knows about (includes host-facing and interswitch)
        all_ports = set(sw.connection.ports.keys())

        # Blocked ports are managed by the Switch instance
        allowed = all_ports - set(sw.blocked_ports)
        return allowed


def launch():
    """
    Launch controller.
    Make sure openflow.discovery is running; if not, start it.
    """
    # Ensure discovery is loaded
    if not core.hasComponent('openflow_discovery'):
        try:
            import pox.openflow.discovery as discovery
            discovery.launch()
            log.info("Launched openflow.discovery automatically")
        except Exception as e:
            log.error("Failed to launch openflow.discovery: %s", e)
            # Controller can still run, but will not learn links -> cannot compute ST.

    core.registerNew(SpanningTreeController)
