#!/usr/bin/env python3
"""
parse_bd.py -- Parse a Vivado IP-Integrator `write_bd_tcl` export into a
structured model, with no dependency on any particular design.

Usage:
    python parse_bd.py <create_bd.tcl> [--out model.json] [--quiet]

What it does
------------
Vivado `write_bd_tcl` emits one `proc create_hier_cell_<x>` per hierarchical
block plus a `proc create_root_design`. Each proc creates its cells, then wires
them, and hierarchies are instantiated by calls like
`create_hier_cell_foo $parent bar`. This script walks that structure and emits a
JSON model plus a human-readable inventory covering:

  * IP instances  (create_bd_cell -type ip): name, VLNV, architecture params
  * hierarchy tree (create_bd_cell -type hier via proc calls)
  * interface nets (connect_bd_intf_net): AXI / AXI-Stream / clock / etc.
  * scalar nets    (connect_bd_net): clocks, resets, interrupts, discretes
  * external ports (create_bd_port / create_bd_intf_port)
  * address map    (assign_bd_address): master -> slave segments

It extracts; it does not classify. Classification into planes/subsystems is the
job of bd_to_dot.py (and the reader), so this stays reusable.
"""
import re, json, sys, argparse, os

# ---- proc bodies (brace-balanced) -----------------------------------------
def find_procs(txt):
    procs = {}
    for m in re.finditer(r'^proc\s+(\S+)\s*\{[^}]*\}\s*\{', txt, re.M):
        name, start, depth, i = m.group(1), m.end(), 1, m.end()
        while i < len(txt) and depth > 0:
            c = txt[i]
            if c == '{': depth += 1
            elif c == '}': depth -= 1
            i += 1
        procs[name] = txt[start:i-1]
    return procs

# ---- per-proc extractors ---------------------------------------------------
def parse_cells(body):
    cells = []
    for m in re.finditer(r'create_bd_cell\s+-type\s+ip\s+-vlnv\s+(\S+)\s+([^\s\]]+)', body):
        vlnv, name = m.group(1), m.group(2)
        props = {}
        after = body[m.end():m.end()+6000]
        pm = re.search(r'set_property\s+-dict\s+\[\s*list(.*?)\]\s*\$' + re.escape(name) + r'\b', after, re.S)
        if pm:
            for cm in re.finditer(r'CONFIG\.([\w.]+)\s*\{([^}]*)\}', pm.group(1)):
                props[cm.group(1)] = cm.group(2)
        cells.append({"name": name, "vlnv": vlnv, "props": props})
    return cells

def parse_hier_calls(body):
    # create_hier_cell_x [current_bd_instance .] NAME   |   ... $hier_obj NAME
    return [(m.group(1), m.group(2)) for m in
            re.finditer(r'(create_hier_cell_\w+)\s+(?:\$\w+|\[current_bd_instance \.\])\s+(\S+)', body)]

def parse_intf_pins(body):
    return [{"mode": m.group(1), "vlnv": m.group(2), "name": m.group(3)} for m in
            re.finditer(r'create_bd_intf_pin\s+-mode\s+(\S+)\s+-vlnv\s+(\S+)\s+(\S+)', body)]

def parse_pins(body):
    pins = []
    for m in re.finditer(r'create_bd_pin\s+([^\n]*)', body):
        toks = m.group(1).split()
        if not toks: continue
        name = toks[-1]
        typ  = toks[toks.index('-type')+1] if '-type' in toks else None
        dirn = toks[toks.index('-dir')+1]  if '-dir'  in toks else None
        pins.append({"name": name, "dir": dirn, "type": typ})
    return pins

def _endpoints(rest, kind):
    return [e.strip() for e in re.findall(r'\[get_bd_%s\s+([^\]]+)\]' % kind, rest)]

def parse_intf_nets(body):
    nets = []
    for m in re.finditer(r'connect_bd_intf_net\s+(?:-intf_net\s+(\S+)\s+)?([^\n]*)', body):
        eps = _endpoints(m.group(2), r'intf_(?:pins|ports)')
        if eps: nets.append({"net": m.group(1), "endpoints": eps})
    return nets

def parse_nets(body):
    nets = []
    for m in re.finditer(r'connect_bd_net\s+(?:-net\s+(\S+)\s+)?([^\n]*)', body):
        eps = _endpoints(m.group(2), r'(?:pins|ports)')
        if eps: nets.append({"net": m.group(1), "endpoints": eps})
    return nets

def parse_ports(body):
    ports = []
    for m in re.finditer(r'create_bd_port\s+([^\n\]]*)', body):
        toks = m.group(1).strip().split()
        if not toks: continue
        ports.append({"name": toks[-1],
                      "dir": toks[toks.index('-dir')+1] if '-dir' in toks else None})
    return ports

def parse_intf_ports(body):
    return [{"mode": m.group(1), "vlnv": m.group(2), "name": m.group(3)} for m in
            re.finditer(r'create_bd_intf_port\s+-mode\s+(\S+)\s+-vlnv\s+(\S+)\s+(\S+)', body)]

def parse_proc(body):
    return {"cells": parse_cells(body), "hier_calls": parse_hier_calls(body),
            "intf_pins": parse_intf_pins(body), "pins": parse_pins(body),
            "intf_nets": parse_intf_nets(body), "nets": parse_nets(body),
            "ports": parse_ports(body), "intf_ports": parse_intf_ports(body)}

# ---- build model -----------------------------------------------------------
def build_model(text):
    procs = {n: b for n, b in find_procs(text).items()
             if n.startswith("create_hier_cell") or n == "create_root_design"}
    parsed = {n: parse_proc(b) for n, b in procs.items()}
    if "create_root_design" not in parsed:
        raise SystemExit("No create_root_design proc found -- is this a write_bd_tcl export?")

    def build(procname, path, seen):
        if procname in seen:  # guard against pathological recursion
            return {"path": path, "proc": procname, "cells": [], "children": {}}
        node = parsed[procname]
        entry = {"path": path, "proc": procname, "cells": node["cells"],
                 "intf_pins": node["intf_pins"], "children": {}}
        for child_proc, inst in node["hier_calls"]:
            if child_proc in parsed:
                cp = (path + "/" + inst) if path else inst
                entry["children"][inst] = build(child_proc, cp, seen | {procname})
        return entry

    root = build("create_root_design", "", set())

    addr = []
    for m in re.finditer(
        r'assign_bd_address\s+-offset\s+(\S+)\s+-range\s+(\S+)\s+-target_address_space\s+'
        r'\[get_bd_addr_spaces\s+([^\]]+)\]\s+\[get_bd_addr_segs\s+([^\]]+)\]', text):
        addr.append({"offset": m.group(1), "range": m.group(2),
                     "master": m.group(3).strip(), "slave": m.group(4).strip()})

    dm = re.search(r'set\s+design_name\s+(\S+)', text)
    part = re.search(r'create_project\s+\S+\s+\S+\s+-part\s+(\S+)', text)
    return {"design_name": dm.group(1) if dm else "system_bd",
            "part": part.group(1) if part else None,
            "root": root,
            "root_ports": parsed["create_root_design"]["ports"],
            "root_intf_ports": parsed["create_root_design"]["intf_ports"],
            "addresses": addr, "parsed": parsed}

# ---- inventory print -------------------------------------------------------
def short(vlnv):
    p = vlnv.split(":"); return p[2] if len(p) >= 3 else vlnv

def inventory(model):
    from collections import Counter
    out = []
    out.append(f"Design: {model['design_name']}   Part: {model['part']}")
    def counts(node, acc):
        acc.append((node["path"] or "(root)", len(node["cells"]), list(node["children"])))
        for c in node["children"].values(): counts(c, acc)
    acc = []; counts(model["root"], acc)
    out.append("\n== Hierarchy (path : #cells : children) ==")
    for p, n, ch in acc: out.append(f"  {p}: {n} cells; children={ch}")
    vc = Counter()
    def wc(node):
        for c in node["cells"]: vc[c["vlnv"]] += 1
        for c in node["children"].values(): wc(c)
    wc(model["root"])
    out.append("\n== IP cell count by VLNV ==")
    for v, n in vc.most_common(): out.append(f"  {n:3d}  {v}")
    out.append(f"\nTotal IP cells: {sum(vc.values())}")
    out.append(f"Interface ports: {len(model['root_intf_ports'])}   "
               f"Scalar ports: {len(model['root_ports'])}   "
               f"Address assignments: {len(model['addresses'])}")
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser(description="Parse a Vivado write_bd_tcl export into a JSON model + inventory.")
    ap.add_argument("tcl")
    ap.add_argument("--out", default="model.json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    with open(a.tcl, encoding="utf-8", errors="replace") as f:
        text = f.read()
    model = build_model(text)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1)
    if not a.quiet:
        print(inventory(model))
        print(f"\nModel written to {os.path.abspath(a.out)}")

if __name__ == "__main__":
    main()
