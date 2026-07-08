#!/usr/bin/env python3
"""Dump the ground truth from a vivado-bd-diagrams model.json before drawing.

Prints, per create_* proc: cells (name + VLNV), hierarchy instantiations, boundary
pins, interface nets, and scalar nets (endpoint lists = clock/reset fan-out).
Every edge in a hand-authored figure must be traceable to a line of this output.

Usage:
  python dump_model.py path/to/arch_diagrams/model.json                 # everything
  python dump_model.py model.json --proc root                          # filter procs
  python dump_model.py model.json --props clk_wiz ddr4                 # cell configs
  python dump_model.py model.json --addresses                          # address map
"""

import argparse
import json
import sys


def fmt_net(net):
    name = net.get("name") or ""
    eps = "  <->  ".join(net.get("endpoints", []))
    return f"  {name} : {eps}" if name else f"  {eps}"


def dump_proc(name, proc):
    print(f"=================== {name} ===================")
    print("--- CELLS ---")
    for c in proc.get("cells", []):
        print(f"  {c.get('name')}  [{c.get('vlnv')}]")
    calls = proc.get("hier_calls", [])
    if calls:
        print("--- HIER CALLS (proc, instance) ---")
        for h in calls:
            print(f"  {h}")
    pins = proc.get("pins", [])
    if pins:
        print("--- PINS (hier boundary) ---")
        for p in pins:
            print(f"  {p.get('name')} (dir={p.get('dir')}, type={p.get('type')})")
    print("--- INTF NETS ---")
    for n in proc.get("intf_nets", []):
        print(fmt_net(n))
    print("--- NETS (scalar: clock/reset fan-out lives here) ---")
    for n in proc.get("nets", []):
        print(fmt_net(n))
    print()


def dump_props(parsed, patterns):
    pats = [p.lower() for p in patterns]
    hits = 0
    for pname, proc in parsed.items():
        for c in proc.get("cells", []):
            cname = (c.get("name") or "").lower()
            if any(p in cname for p in pats):
                hits += 1
                print(f"== {pname} :: {c.get('name')}  [{c.get('vlnv')}] ==")
                props = c.get("props") or {}
                if not props:
                    print("  (props empty - grep create_bd.tcl for this cell's CONFIG.* block)")
                for k in sorted(props):
                    print(f"  {k} = {props[k]}")
                print()
    if not hits:
        print(f"no cells matching {patterns}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to model.json")
    ap.add_argument("--proc", help="only procs whose name contains this substring")
    ap.add_argument("--props", nargs="+", metavar="CELL",
                    help="print props of cells whose name contains any substring")
    ap.add_argument("--addresses", action="store_true", help="print the address map")
    args = ap.parse_args()

    with open(args.model, encoding="utf-8") as f:
        model = json.load(f)
    parsed = model.get("parsed", {})

    if args.props:
        dump_props(parsed, args.props)
        return
    if args.addresses:
        for a in model.get("addresses", []):
            print(f"  {a.get('offset')}  range={a.get('range')}  "
                  f"{a.get('master')} -> {a.get('slave')}")
        return

    for name, proc in parsed.items():
        if args.proc and args.proc.lower() not in name.lower():
            continue
        dump_proc(name, proc)

    ports = model.get("root_intf_ports", [])
    if ports and not args.proc:
        print("=================== root interface ports ===================")
        for p in ports:
            print(f"  {p.get('name')}  mode={p.get('mode')}  [{p.get('vlnv')}]")


if __name__ == "__main__":
    main()
