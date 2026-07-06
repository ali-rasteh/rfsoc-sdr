#!/usr/bin/env python3
"""Parser for Vivado write_bd_tcl exports (hierarchical, proc-based)."""
import re, json, sys, os, argparse

ap = argparse.ArgumentParser(
    description="Parse a Vivado write_bd_tcl export into model.json + an inventory.")
ap.add_argument("tcl", help="path to the target create_bd.tcl (write_bd_tcl export)")
ap.add_argument("--out", default=None,
                help="output model JSON path (default: arch_diagrams/model.json beside the TCL)")
args = ap.parse_args()
TCL = args.tcl

# All generated files live in arch_diagrams/ in the same folder as the TCL.
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(TCL)), "arch_diagrams")
os.makedirs(OUTDIR, exist_ok=True)
MODEL = args.out or os.path.join(OUTDIR, "model.json")

with open(TCL, encoding="utf-8", errors="replace") as f:
    lines = f.readlines()
text = "".join(lines)

# --- Split into proc bodies by brace tracking ---
def find_procs(txt):
    procs = {}
    for m in re.finditer(r'^proc\s+(\S+)\s*\{[^}]*\}\s*\{', txt, re.M):
        name = m.group(1)
        start = m.end()  # just after opening brace of body
        depth = 1
        i = start
        while i < len(txt) and depth > 0:
            c = txt[i]
            if c == '{': depth += 1
            elif c == '}': depth -= 1
            i += 1
        procs[name] = txt[start:i-1]
    return procs

procs = find_procs(text)

def parse_dict_props(body, start_idx):
    """Given body text and index right after a create_bd_cell line, look ahead for
    the set_property -dict [list ...] block that applies to that just-created cell."""
    # We'll parse props per-cell separately below using regex on the block.
    pass

def parse_cells(body):
    """Return list of dicts: instance, vlnv, props."""
    cells = []
    # find create_bd_cell -type ip -vlnv VLNV NAME
    for m in re.finditer(r'create_bd_cell\s+-type\s+ip\s+-vlnv\s+(\S+)\s+(\S+)\s*\]', body):
        vlnv = m.group(1)
        name = m.group(2)
        # look for following set_property -dict [ list ... ] $name
        props = {}
        # search a window after this match
        after = body[m.end():m.end()+4000]
        pm = re.search(r'set_property\s+-dict\s+\[\s*list(.*?)\]\s*\$'+re.escape(name)+r'\b', after, re.S)
        if pm:
            block = pm.group(1)
            for cm in re.finditer(r'CONFIG\.(\S+)\s*\{([^}]*)\}', block):
                props[cm.group(1)] = cm.group(2)
        cells.append({"name": name, "vlnv": vlnv, "props": props})
    return cells

def parse_hier_calls(body):
    """create_hier_cell_X <parent> NAME  -> (proc_name, instance_name)"""
    calls = []
    for m in re.finditer(r'(create_hier_cell_\w+)\s+(?:\$\w+|\[current_bd_instance \.\])\s+(\S+)', body):
        calls.append((m.group(1), m.group(2)))
    return calls

def parse_intf_pins(body):
    pins = []
    for m in re.finditer(r'create_bd_intf_pin\s+-mode\s+(\S+)\s+-vlnv\s+(\S+)\s+(\S+)', body):
        pins.append({"mode": m.group(1), "vlnv": m.group(2), "name": m.group(3)})
    return pins

def parse_pins(body):
    pins = []
    for m in re.finditer(r'create_bd_pin\s+(.*)', body):
        args = m.group(1).strip()
        # last token is name
        toks = args.split()
        name = toks[-1]
        typ = None
        dirn = None
        if '-type' in toks:
            typ = toks[toks.index('-type')+1]
        if '-dir' in toks:
            dirn = toks[toks.index('-dir')+1]
        pins.append({"name": name, "dir": dirn, "type": typ})
    return pins

def parse_intf_nets(body):
    nets = []
    for m in re.finditer(r'connect_bd_intf_net\s+(?:-intf_net\s+(\S+)\s+)?(.*)', body):
        netname = m.group(1)
        rest = m.group(2)
        endpoints = re.findall(r'\[get_bd_intf_(?:pins|ports)\s+([^\]]+)\]', rest)
        endpoints = [e.strip() for e in endpoints]
        nets.append({"net": netname, "endpoints": endpoints})
    return nets

def parse_nets(body):
    nets = []
    for m in re.finditer(r'connect_bd_net\s+(?:-net\s+(\S+)\s+)?(.*)', body):
        netname = m.group(1)
        rest = m.group(2)
        endpoints = re.findall(r'\[get_bd_(?:pins|ports)\s+([^\]]+)\]', rest)
        endpoints = [e.strip() for e in endpoints]
        nets.append({"net": netname, "endpoints": endpoints})
    return nets

def parse_ports(body):
    ports = []
    for m in re.finditer(r'create_bd_port\s+(.*)', body):
        args = m.group(1).strip().rstrip(']').strip()
        toks = args.split()
        name = toks[-1]
        dirn = toks[toks.index('-dir')+1] if '-dir' in toks else None
        ports.append({"name": name, "dir": dirn})
    return ports

def parse_intf_ports(body):
    ports = []
    for m in re.finditer(r'create_bd_intf_port\s+-mode\s+(\S+)\s+-vlnv\s+(\S+)\s+(\S+)', body):
        ports.append({"mode": m.group(1), "vlnv": m.group(2), "name": m.group(3)})
    return ports

# Parse each hierarchy proc
parsed = {}
for pname, body in procs.items():
    if not pname.startswith("create_hier_cell") and pname != "create_root_design":
        continue
    parsed[pname] = {
        "cells": parse_cells(body),
        "hier_calls": parse_hier_calls(body),
        "intf_pins": parse_intf_pins(body),
        "pins": parse_pins(body),
        "intf_nets": parse_intf_nets(body),
        "nets": parse_nets(body),
        "ports": parse_ports(body),
        "intf_ports": parse_intf_ports(body),
    }

# --- Build the hierarchy tree from root ---
tree = {}
def build(procname, path):
    node = parsed[procname]
    entry = {"path": path, "proc": procname,
             "cells": node["cells"], "children": {}}
    for (child_proc, inst) in node["hier_calls"]:
        cp = child_proc
        childpath = (path + "/" + inst) if path else inst
        entry["children"][inst] = build(cp, childpath)
    return entry

root = build("create_root_design", "")

# --- Address assignments (from full text) ---
addr = []
for m in re.finditer(r'assign_bd_address\s+-offset\s+(\S+)\s+-range\s+(\S+)\s+-target_address_space\s+\[get_bd_addr_spaces\s+([^\]]+)\]\s+\[get_bd_addr_segs\s+([^\]]+)\]', text):
    addr.append({"offset": m.group(1), "range": m.group(2),
                 "master": m.group(3).strip(), "slave": m.group(4).strip()})

out = {"root": root,
       "root_ports": parsed["create_root_design"]["ports"],
       "root_intf_ports": parsed["create_root_design"]["intf_ports"],
       "addresses": addr,
       "parsed": parsed}

with open(MODEL,"w") as f:
    json.dump(out, f, indent=1)

# --- Print summary ---
def walk_counts(node, acc):
    acc.append((node["path"] or "(root)", len(node["cells"]), list(node["children"].keys())))
    for c in node["children"].values():
        walk_counts(c, acc)
acc=[]
walk_counts(root, acc)
print("=== HIERARCHY (path: #cells : children) ===")
for p,n,ch in acc:
    print(f"  {p}: {n} cells; children={ch}")

print("\n=== CELL COUNT BY VLNV ===")
from collections import Counter
vc = Counter()
def wc(node):
    for c in node["cells"]:
        vc[c["vlnv"]] += 1
    for ch in node["children"].values():
        wc(ch)
wc(root)
for v,n in vc.most_common():
    print(f"  {n:2d}  {v}")

print(f"\nTotal IP cells: {sum(vc.values())}")
print(f"Root intf ports: {len(parsed['create_root_design']['intf_ports'])}")
print(f"Root scalar ports: {len(parsed['create_root_design']['ports'])}")
print(f"Address assignments: {len(addr)}")
