#!/usr/bin/env python3
"""
bd_to_dot.py -- Turn a parsed Block-Design model (model.json from parse_bd.py)
into publication-oriented Graphviz DOT, filtered by functional plane.

Usage:
    python bd_to_dot.py model.json --view complete   [--out out.dot] [--show-clutter]
    python bd_to_dot.py model.json --view data
    python bd_to_dot.py model.json --view control
    python bd_to_dot.py model.json --view clocking
    python bd_to_dot.py model.json --list-subsystems
    python bd_to_dot.py model.json --subsystem adc_path [--out adc.dot]

Design intent
-------------
This is a *first-pass* generator: it recovers connectivity correctly and colors
blocks by function, giving you a clean DOT to render and then hand-refine (add
port tables to hero blocks, nudge labels, etc.). It never invents connections --
every edge comes from a connect_bd_(intf_)net in the source.

How planes are defined
----------------------
A "plane" is the subgraph induced by a *class of connection*, which is the honest
way to slice a BD:
  data plane      = AXI-Stream + AXI-MM edges (and the blocks they touch)
  control plane   = AXI-Lite + interrupt edges
  clocking plane  = clock + reset edges
Block fill color is chosen from the IP VLNV (see PLANE_BY_VLNV); edge color/style
from the connection class. Both tables are documented in references/classification.md.

Cross-hierarchy signals are stitched with a union-find over canonicalized endpoint
names: a child boundary pin `PIN` and the parent's `child/PIN` resolve to the same
string, so a net that crosses a hierarchy boundary becomes one component.
"""
import json, re, argparse, os
from collections import defaultdict

# ----- visual language (kept in sync with references/visual_conventions.md) --
PLANE_COL = {
    "data":    ("#DAE8FC", "#1F6FC4"), "control": ("#D5E8D4", "#2E8B57"),
    "clocking":("#FFF2CC", "#C99A00"), "rf":      ("#E1D5E7", "#8E44AD"),
    "memory":  ("#F8CECC", "#B85450"), "ps":      ("#FFE6CC", "#D79B00"),
    "other":   ("#EEEEEE", "#888888"),
}
EDGE_STYLE = {
    "stream": ("#1F6FC4", "solid", 2.0, "normal"), "mm": ("#B85450", "solid", 1.7, "normal"),
    "lite":   ("#2E8B57", "solid", 1.4, "normal"), "irq": ("#8E44AD", "dashed", 1.3, "vee"),
    "clock":  ("#C99A00", "dashed", 1.3, "vee"),   "reset": ("#B85450", "dotted", 1.4, "vee"),
    "intf":   ("#555555", "solid", 1.3, "normal"), "wire": ("#777777", "solid", 1.0, "normal"),
}
PLANE_LABEL = {"data":"Data plane","control":"Control plane","clocking":"Clocking / sync",
               "rf":"RF front-end","memory":"Memory / DDR","ps":"Processor system","other":"Other"}
EDGE_LABEL  = {"stream":"AXI-Stream","mm":"AXI-MM data","lite":"AXI-Lite ctrl","irq":"interrupt",
               "clock":"clock","reset":"reset"}
EDGE_SAMPLE = {"stream":"———","mm":"———","lite":"———","irq":"– – –","clock":"– – –","reset":"······"}

# ----- classification tables -------------------------------------------------
CLUTTER_VLNV = {"xlconstant","xlslice","util_vector_logic","util_reduced_logic",
                "ila","vio","system_ila","jtag_axi","axis_ila"}
def vlnv_short(v):
    p = v.split(":"); return p[2] if len(p) >= 3 else v

def plane_of(vlnv, name=""):
    s = vlnv_short(vlnv)
    if s in ("zynq_ultra_ps_e","zynq_ps7","processing_system7","microblaze"): return "ps"
    if s in ("ddr4","ddr3","mig_7series","hbm","axi_bram_ctrl","blk_mem_gen"): return "memory"
    if re.search(r'(rf_data_converter|rfdc)', s): return "rf"
    if s in ("clk_wiz","proc_sys_reset","util_ds_buf") or s == "sync": return "clocking"
    if s.startswith("axis_") or s in ("axi_dma","axi_datamover","axi_vdma","smartconnect","adc_strm_mux"):
        return "data"
    if s in ("axi_interconnect","axi_intc","xlconcat","axi_gpio","axi_uartlite","axi_timer",
             "axi_iic","axi_quad_spi","axi_uart16550"): return "control"
    return "other"

def is_clutter(vlnv):
    return vlnv_short(vlnv) in CLUTTER_VLNV

# ----- model flattening ------------------------------------------------------
def walk(node, fn, parent=""):
    fn(node)
    for ch in node["children"].values(): walk(ch, fn)

def build_index(model):
    cells = {}      # full path -> {name,vlnv,plane,clutter,props,parent}
    hiers = set()   # hier full paths
    intf_pin_vlnv = {}  # full "path/pin" -> interface vlnv (for edge typing hints)
    def visit(node):
        path = node["path"]
        if path: hiers.add(path)
        for ip in node.get("intf_pins", []):
            intf_pin_vlnv[(path + "/" if path else "") + ip["name"]] = ip["vlnv"]
        for c in node["cells"]:
            fp = (path + "/" if path else "") + c["name"]
            cells[fp] = {"name": c["name"], "vlnv": c["vlnv"], "plane": plane_of(c["vlnv"], c["name"]),
                         "clutter": is_clutter(c["vlnv"]), "props": c.get("props", {}), "parent": path}
    walk(model["root"], visit)
    ports = {}
    for p in model["root_ports"]:      ports[p["name"]] = {"dir": p.get("dir"), "intf": False}
    for p in model["root_intf_ports"]: ports[p["name"]] = {"dir": None, "mode": p.get("mode"), "intf": True}
    return cells, hiers, ports, intf_pin_vlnv

def proc_paths(model):
    """proc name -> hierarchy path (write_bd_tcl emits one proc per instance)."""
    m = {}
    def visit(node): m[node["proc"]] = node["path"]
    walk(model["root"], visit)
    return m

# ----- union-find ------------------------------------------------------------
class DSU:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb

def canon(path, endpoint):
    return (path + "/" if path else "") + endpoint

def owner_pin(s):
    return (s.rsplit("/", 1)[0], s.rsplit("/", 1)[1]) if "/" in s else ("", s)

def stitch(model, cells, hiers, ports):
    """Return list of components; each = {'kind','pins':[canon...],'nets':[names]}."""
    pp = proc_paths(model)
    dsu = DSU(); comp_meta = {}
    def add_net(path, net, kind):
        eps = [canon(path, e) for e in net["endpoints"]]
        for e in eps: dsu.find(e)
        for e in eps[1:]: dsu.union(eps[0], e)
        for e in eps:
            comp_meta.setdefault(e, {"kind": kind, "nets": set()})
            comp_meta[e]["kind"] = kind
            if net.get("net"): comp_meta[e]["nets"].add(net["net"])
    for proc, node in model["parsed"].items():
        path = pp.get(proc, "")
        for n in node["intf_nets"]: add_net(path, n, "intf")
        for n in node["nets"]:      add_net(path, n, "scalar")
    groups = defaultdict(list)
    for e in list(dsu.p): groups[dsu.find(e)].append(e)
    comps = []
    for _, members in groups.items():
        kinds = {comp_meta[m]["kind"] for m in members if m in comp_meta}
        nets  = set().union(*[comp_meta[m]["nets"] for m in members if m in comp_meta]) if members else set()
        comps.append({"kind": ("intf" if "intf" in kinds else "scalar"),
                      "pins": members, "nets": nets})
    return comps

# ----- edge classification & endpoint resolution -----------------------------
OUT_PAT = re.compile(r'(^m_|^m\d|_m_|maxi|m_axi|m\d\d_axis|m_axis|mm2s|dout|_o$|clk_out|clkout|'
                     r'introut|irq|gpio_o|peripheral_aresetn|interconnect_aresetn|dest_out|'
                     r'ui_clk|pl_clk|pl_resetn|calib_complete)', re.I)

def edge_class(comp):
    pins = [owner_pin(s)[1] for s in comp["pins"]]
    j = " ".join(pins).lower()
    if comp["kind"] == "intf":
        # diff/reference clock interfaces (clk_wiz, IBUFDS, sys/lmk clocks) are clocks, not AXI
        if re.search(r'(clk_in|clk_ref|clk_d\b|sysref|diff_clock)', j) or ("clk" in j and "axi" not in j):
            return "clock"
        if "axis" in j or re.search(r'm\d\d_axis|s\d\d_axis', j): return "stream"
        if re.search(r'\bv(in|out)\d', j): return "stream"          # RFDC analog vin/vout dataflow
        if "lite" in j: return "lite"
        if re.search(r'(hp\d|hpm|saxigp|maxigp|s2mm|mm2s|ddr|_mem)', j): return "mm"
        if re.search(r'\bs_axi\b', j) or "\baxi_lite\b" in j:        # bare register port = control
            return "lite"
        if "axi" in j: return "mm"
        return "intf"
    if re.search(r'\b\w*clk\w*\b', j) or "aclk" in j:
        if re.search(r'(reset|resetn|rstn)', j) and not re.search(r'(aclk|_clk|clk_)', j): pass
    if re.search(r'(aclk|_clk\b|clk_|clkin|clk_in|dest_clk|src_clk|ui_clk|pl_clk)', j): return "clock"
    if re.search(r'(resetn|aresetn|rstn|_rst|sys_rst|reset)', j): return "reset"
    if re.search(r'(irq|introut|interrupt|\bintr\b)', j): return "irq"
    return "wire"

def resolve_nodes(comp, cells, ports):
    """Return list of (nodeid, pin, kind) where kind in {'cell','port'}; skip junctions."""
    out = []
    for s in comp["pins"]:
        owner, pin = owner_pin(s)
        if owner in cells:
            out.append((owner, pin, "cell"))
        elif "/" not in s and s in ports:
            out.append(("port:" + s, s, "port"))
    # de-dup by nodeid keeping first pin
    seen = {};
    for nid, pin, k in out:
        if nid not in seen: seen[nid] = (nid, pin, k)
    return list(seen.values())

def pick_source(nodes, ports):
    for nid, pin, k in nodes:
        if k == "port" and ports.get(pin, {}).get("dir") == "I": return nid
        if k == "cell" and OUT_PAT.search(pin): return nid
    return nodes[0][0] if nodes else None

# ----- DOT emission ----------------------------------------------------------
def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

ARCH_PROPS = ("NUM_MI","NUM_SI","NUM_PORTS","FIFO_DEPTH","IS_ACLK_ASYNC","C_OPERATION",
              "AXIS_TDATA_WIDTH","MODE","CONST_VAL","C0.DDR4_AxiDataWidth","C0.DDR4_DataWidth")

def node_label(fp, info):
    parts = [info["name"], vlnv_short(info["vlnv"])]
    ap = [f"{k.split('.')[-1]}={v}" for k, v in info["props"].items() if k in ARCH_PROPS]
    if ap: parts.append(" ".join(ap[:3]))
    return "\\n".join(esc(p) for p in parts)

def legend_html(planes, classes):
    rows = [(PLANE_COL[p][0], PLANE_COL[p][1], PLANE_LABEL[p]) for p in planes]
    eds  = [(EDGE_STYLE[c][0], EDGE_SAMPLE[c], EDGE_LABEL[c]) for c in classes]
    n = max(len(rows), len(eds))
    r = '<<table border="1" color="#777777" cellborder="0" cellspacing="4" cellpadding="2" bgcolor="white">'
    r += '<tr><td colspan="4" align="center"><font point-size="12"><b>Legend</b></font></td></tr>'
    for i in range(n):
        r += "<tr>"
        if i < len(rows):
            r += (f'<td bgcolor="{rows[i][0]}" width="26"><font point-size="4"> </font></td>'
                  f'<td align="left"><font point-size="10">{rows[i][2]}</font></td>')
        else: r += '<td></td><td></td>'
        if i < len(eds):
            r += (f'<td align="right"><font color="{eds[i][0]}" point-size="13"><b>{eds[i][1]}</b></font></td>'
                  f'<td align="left"><font point-size="10">{eds[i][2]}</font></td>')
        else: r += '<td></td><td></td>'
        r += "</tr>"
    return r + "</table>>"

def emit_clusters(node_ids, cells, indent="  "):
    """Nest included cell nodes under subgraph cluster_* by hierarchy path."""
    tree = {}
    for nid in node_ids:
        if nid.startswith("port:"): continue
        parts = cells[nid]["parent"].split("/") if cells[nid]["parent"] else []
        cur = tree
        for seg in parts:
            cur = cur.setdefault(("H", seg), {})
        cur.setdefault(("N", nid), None)
    out = []
    def rec(subtree, path, depth):
        pad = "  " * depth
        for key, val in subtree.items():
            kind, name = key
            if kind == "H":
                cid = (path + "_" + name).strip("_").replace("/", "_")
                out.append(f'{pad}subgraph "cluster_{cid}" {{')
                out.append(f'{pad}  label=<<b>{esc(name)}</b>>; style="rounded"; color="#999999"; '
                           f'bgcolor="#FBFBFB"; fontsize=12; margin=10;')
                rec(val, path + "/" + name, depth + 1)
                out.append(f'{pad}}}')
            else:
                info = cells[name]; fill, line = PLANE_COL[info["plane"]]
                out.append(f'{pad}"{name}" [label="{node_label(name, info)}", '
                           f'fillcolor="{fill}", color="{line}"];')
    rec(tree, "", 1)
    return "\n".join(out)

def build_dot(model, view, show_clutter=False, subsystem=None):
    cells, hiers, ports, _ = build_index(model)
    comps = stitch(model, cells, hiers, ports)

    # classify each component into an edge, keep those with >=2 real nodes
    edges = []   # (src, dst, cls, label)
    for comp in comps:
        cls = edge_class(comp)
        nodes = resolve_nodes(comp, cells, ports)
        if not show_clutter:
            nodes = [n for n in nodes if not (n[0] in cells and cells[n[0]]["clutter"])]
        if len(nodes) < 2: continue
        src = pick_source(nodes, ports)
        lbl_pin = next((p for nid, p, k in nodes if nid == src), "")
        label = lbl_pin if 0 < len(lbl_pin) <= 16 and cls in ("stream","mm","lite") else ""
        for nid, pin, k in nodes:
            if nid == src: continue
            edges.append((src, nid, cls, label)); label = ""

    # view filter
    def keep_edge(cls):
        if view == "complete": return True
        if view == "data":     return cls in ("stream","mm")
        if view == "control":  return cls in ("lite","irq")
        if view == "clocking": return cls in ("clock","reset")
        return True
    if subsystem:
        pref = subsystem.rstrip("/")
        keep_node = lambda nid: nid.startswith("port:") or nid == pref or cells.get(nid,{}).get("parent","").startswith(pref) or nid.startswith(pref + "/")
        edges = [e for e in edges if keep_node(e[0]) or keep_node(e[1])]
    else:
        edges = [e for e in edges if keep_edge(e[2])]

    # nodes actually used
    used = set()
    for a, b, cls, lbl in edges: used.add(a); used.add(b)
    cell_nodes = [n for n in used if not n.startswith("port:")]
    port_nodes = [n for n in used if n.startswith("port:")]

    planes_present = []
    for n in cell_nodes:
        pl = cells[n]["plane"]
        if pl not in planes_present: planes_present.append(pl)
    planes_present = [p for p in PLANE_COL if p in planes_present]
    classes_present = []
    for a,b,cls,lbl in edges:
        c = cls if cls in EDGE_LABEL else None
        if c and c not in classes_present: classes_present.append(c)
    classes_present = [c for c in EDGE_LABEL if c in classes_present]

    title = {"complete":"Complete Architecture","data":"Data Plane",
             "control":"Control Plane","clocking":"Clocking / Synchronization"}.get(
             view, f"Subsystem — {subsystem}")
    title = f"{model['design_name']} — {title}"

    L = []
    L.append(f'digraph "{view}" {{')
    L.append('  graph [rankdir=LR, splines=ortho, nodesep=0.32, ranksep=0.55, fontname="Helvetica",')
    L.append('         bgcolor="white", compound=true, labelloc="t", fontsize=20, forcelabels=true,')
    L.append(f'         pack=false, label=<<b>{esc(title)}</b>>];')
    L.append('  node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", penwidth=1.3];')
    L.append('  edge [fontname="Helvetica", fontsize=9, penwidth=1.4, arrowsize=0.7];')
    # ports
    for pn in port_nodes:
        nm = pn[4:]; d = ports.get(nm, {})
        sh = "invhouse" if d.get("dir") == "O" else "cds"
        L.append(f'  "{pn}" [label="{esc(nm)}", shape={sh}, fillcolor="white", color="#666666", fontsize=10];')
    # clustered cells
    L.append(emit_clusters(cell_nodes, cells))
    # edges
    seen = set()
    for a, b, cls, lbl in edges:
        if (a, b, cls) in seen: continue
        seen.add((a, b, cls))
        col, sty, pw, ah = EDGE_STYLE.get(cls, EDGE_STYLE["wire"])
        x = f', xlabel="{esc(lbl)}", fontcolor="{col}"' if lbl else ""
        L.append(f'  "{a}" -> "{b}" [color="{col}", style="{sty}", penwidth={pw}, arrowhead={ah}{x}];')
    # legend (only classes/planes actually present)
    if planes_present or classes_present:
        L.append(f'  "legend" [shape=plaintext, margin=0, label={legend_html(planes_present or ["data"], classes_present or ["stream"])}];')
    L.append("}")
    return "\n".join(L)

def list_subsystems(model):
    cells, hiers, ports, _ = build_index(model)
    tops = list(model["root"]["children"].keys())
    print("Top-level hierarchies (natural subsystems):")
    for t in tops:
        sub = [c for c in cells if c.startswith(t + "/") or c == t]
        planes = sorted({cells[c]["plane"] for c in sub})
        print(f"  {t:20s}  {len(sub):3d} cells  planes={planes}")
    # IP-defined subsystems
    for c, info in cells.items():
        if info["plane"] in ("rf","memory","ps") and "/" not in c:
            print(f"  {c:20s}  (root IP, plane={info['plane']})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--view", choices=["complete","data","control","clocking"], default="complete")
    ap.add_argument("--subsystem", default=None, help="hierarchy path to expand as its own diagram")
    ap.add_argument("--list-subsystems", action="store_true")
    ap.add_argument("--show-clutter", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model = json.load(open(a.model, encoding="utf-8"))
    if a.list_subsystems:
        list_subsystems(model); return
    dot = build_dot(model, a.view, a.show_clutter, a.subsystem)
    out = a.out or (f"{a.subsystem.replace('/','_')}.dot" if a.subsystem else f"{a.view}.dot")
    open(out, "w", encoding="utf-8").write(dot)
    print(f"Wrote {os.path.abspath(out)}")

if __name__ == "__main__":
    main()
