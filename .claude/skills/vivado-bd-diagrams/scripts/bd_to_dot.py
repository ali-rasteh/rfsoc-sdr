# -*- coding: utf-8 -*-
"""Generate publication-quality Graphviz DOT for sounder_bbf_sivers_ddr4_2x2 (system_bd).
All instance names exact & traceable to create_bd.tcl. Clutter suppressed."""
import os, argparse
ap = argparse.ArgumentParser(
    description="Generate architecture DOT diagrams into arch_diagrams/ beside the TCL.")
ap.add_argument("tcl", help="path to the target create_bd.tcl (used to locate the output folder)")
args = ap.parse_args()
# All generated files live in arch_diagrams/ in the same folder as the TCL.
DIAG = os.path.join(os.path.dirname(os.path.abspath(args.tcl)), "arch_diagrams")
os.makedirs(DIAG, exist_ok=True)

C = dict(
    DATA_F="#DAE8FC", DATA_L="#1F6FC4", CTRL_F="#D5E8D4", CTRL_L="#2E8B57",
    CLK_F="#FFF2CC", CLK_L="#C99A00", RST="#B85450", RF_F="#E1D5E7", RF_L="#8E44AD",
    MEM_F="#F8CECC", MEM_L="#B85450", PS_F="#FFE6CC", PS_L="#D79B00", IRQ="#8E44AD",
    PORT_F="#FFFFFF", PORT_L="#666666", CDC_F="#FDF3D8")

PRE = ('digraph "{name}" {{\n'
    '  graph [rankdir=LR, splines=ortho, nodesep=0.32, ranksep=0.55, fontname="Helvetica",\n'
    '         bgcolor="white", compound=true, labelloc="t", fontsize=20, forcelabels=true,\n'
    '         pack=false, label=<<b>{title}</b>>];\n'
    '  node  [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", penwidth=1.3];\n'
    '  edge  [fontname="Helvetica", fontsize=9, penwidth=1.4, arrowsize=0.7];\n')
FOOT = "}\n"

def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _cell(pid, txt, fill, align):
    pc = f' port="{pid}"' if pid else ''
    inner = f'<font point-size="10">{esc(txt)}</font>' if txt else ' '
    return f'<td{pc} align="{align}" bgcolor="{fill}">{inner}</td>'

def html(title, rows, fill, line, tfont="white"):
    r = '<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="white">'
    r += f'<tr><td colspan="2" bgcolor="{line}"><font color="{tfont}" point-size="11"><b>{esc(title)}</b></font></td></tr>'
    for lp,lt,rp,rt in rows:
        if rp is None and not rt:   # single-column row spanning both
            lc=f' port="{lp}"' if lp else ''
            r += f'<tr><td colspan="2"{lc} align="left" bgcolor="{fill}"><font point-size="10">{esc(lt)}</font></td></tr>'
        else:
            r += '<tr>' + _cell(lp, lt, fill, "left") + _cell(rp, rt, fill, "right") + '</tr>'
    return r+'</table>>'

def node(nid,label,fill,line,shape="box",extra=""):
    return f'  "{nid}" [label="{label}", shape={shape}, fillcolor="{fill}", color="{line}"{(", "+extra) if extra else ""}];\n'
def hnode(nid,label): return f'  "{nid}" [shape=plaintext, label={label}];\n'
def port(nid,label,kind="in"):
    sh="cds" if kind=="in" else "invhouse"
    return f'  "{nid}" [label="{label}", shape={sh}, fillcolor="{C["PORT_F"]}", color="{C["PORT_L"]}", style="filled", fontsize=10];\n'

def qref(x):
    """Quote a DOT endpoint; keep node:port references as "node":"port"."""
    if ":" in x:
        n,p = x.split(":",1)
        return f'"{n}":"{p}"'
    return f'"{x}"'

def _edge(a,b,color,style,pw,lbl,arrow="normal",ltail=None,lhead=None,constraint=None):
    at=[f'color="{color}"',f'style="{style}"',f'penwidth={pw}',f'arrowhead={arrow}']
    if lbl: at.append(f'xlabel="{lbl}", fontcolor="{color}"')
    if ltail: at.append(f'ltail="{ltail}"')
    if lhead: at.append(f'lhead="{lhead}"')
    if constraint is not None: at.append(f'constraint={str(constraint).lower()}')
    return f'  {qref(a)} -> {qref(b)} [{", ".join(at)}];\n'
def e_axis(a,b,l="",**k): return _edge(a,b,C["DATA_L"],"solid",2.0,l,**k)
def e_mm(a,b,l="",**k):   return _edge(a,b,C["MEM_L"],"solid",1.7,l,**k)
def e_lite(a,b,l="",**k): return _edge(a,b,C["CTRL_L"],"solid",1.4,l,**k)
def e_clk(a,b,l="",**k):  return _edge(a,b,C["CLK_L"],"dashed",1.3,l,arrow="vee",**k)
def e_rst(a,b,l="",**k):  return _edge(a,b,C["RST"],"dotted",1.4,l,arrow="vee",**k)
def e_irq(a,b,l="",**k):  return _edge(a,b,C["IRQ"],"dashed",1.3,l,arrow="vee",**k)
def e_ctl(a,b,l="",**k):  return _edge(a,b,"#555555","solid",1.1,l,**k)

def cluster(cid,label,body,bg="#FBFBFB",line="#999999"):
    return (f'  subgraph "cluster_{cid}" {{\n    label=<<b>{label}</b>>; fontsize=13; style="rounded";'
            f' color="{line}"; bgcolor="{bg}"; penwidth=1.5; margin=12;\n{body}  }}\n')

_PLANES=[(C["DATA_F"],C["DATA_L"],"Data plane"),(C["CTRL_F"],C["CTRL_L"],"Control plane"),
    (C["CLK_F"],C["CLK_L"],"Clocking / sync"),(C["RF_F"],C["RF_L"],"RF front-end"),
    (C["MEM_F"],C["MEM_L"],"DDR memory"),(C["PS_F"],C["PS_L"],"Processor system")]
_EDGES=[(C["DATA_L"],"———","AXI-Stream"),(C["MEM_L"],"———","AXI-MM data"),
    (C["CTRL_L"],"———","AXI-Lite ctrl"),(C["CLK_L"],"– – –","clock"),
    (C["RST"],"······","reset"),(C["IRQ"],"– – –","interrupt")]

def legend_node():
    """Compact, self-contained HTML legend as one disconnected node (packed to a corner)."""
    r='<<table border="1" color="#777777" cellborder="0" cellspacing="4" cellpadding="2" bgcolor="white">'
    r+='<tr><td colspan="4" align="center"><font point-size="12"><b>Legend</b></font></td></tr>'
    for (pf,pl,pn),(ec,es,en) in zip(_PLANES,_EDGES):
        r+=('<tr>'
            f'<td bgcolor="{pf}" width="26"><font point-size="4"> </font></td>'
            f'<td align="left"><font point-size="10">{pn}</font></td>'
            f'<td align="right"><font color="{ec}" point-size="13"><b>{es}</b></font></td>'
            f'<td align="left"><font point-size="10">{en}</font></td>'
            '</tr>')
    r+='</table>>'
    return f'  "legend" [shape=plaintext, margin=0, label={r}];\n'

LEG = legend_node()

# ---- hero HTML blocks (exact ports; direction shown by edge arrowheads) ----
def L_rfdc():
    return html("usp_rf_data_converter_0  (RFDC 2.6)", [
        ("vin","vin0_01/23, vin2_01/23","m00","m00_axis"),
        ("s00","s00_axis","m20","m20_axis"),
        ("s10","s10_axis","vout","vout00..03 / 10..13"),
        ("saxi","s_axi (Lite)","irq","irq"),
        ("smpclk","adc/dac sample clk",None,""),
        ("sysref","user_sysref_adc/dac","aclk","axis aclk (rf_clk)")], C["RF_F"], C["RF_L"])
def L_ps():
    return html("zynq_ultra_ps_e_0  (Zynq US+ PS)", [
        ("hpm0","M_AXI_HPM0_FPD",None,""),("hpm1","M_AXI_HPM1_FPD",None,""),
        ("hp0","S_AXI_HP0_FPD",None,""),("hp1","S_AXI_HP1_FPD",None,""),
        ("gpio","emio_gpio_o[94:0]",None,""),("irq","pl_ps_irq0",None,""),
        ("plclk","pl_clk0 (100 MHz)",None,""),("plrst","pl_resetn0",None,"")], C["PS_F"], C["PS_L"])
def L_ddr():
    return html("ddr4_0  (DDR4 ctrl 2.2)", [
        ("saxi","C0_DDR4_S_AXI (512b)","ddr","C0_DDR4 (ddr4_pl)"),
        ("sysclk","c0_sys_clk_i","uiclk","c0_ddr4_ui_clk"),
        ("calib","c0_init_calib_complete","arstn","c0_ddr4_aresetn")], C["MEM_F"], C["MEM_L"])
def L_dma(nid,mode):
    rows=[("saxis","S_AXIS_S2MM","maxi","M_AXI_S2MM"),("lite","S_AXI_LITE","irq","s2mm_introut")] if mode=="s2mm" \
        else [("maxis","M_AXIS_MM2S","maxi","M_AXI_MM2S"),("lite","S_AXI_LITE","irq","mm2s_introut")]
    return html(nid, rows, C["DATA_F"], C["DATA_L"])

# ===================== 1. COMPLETE =====================
def diagram_complete():
    s=PRE.format(name="complete",title="RFSoC 2x2 Channel Sounder — Complete Architecture (system_bd)")
    s+=cluster("ext_in","External RF / Clock In",
        port("vin","vin0_01/23 vin2_01/23")+port("adcclk","adc0_clk adc2_clk")+
        port("dacclk","dac0_clk dac1_clk")+port("sysref","sysref_in")+
        port("lmk","lmk_clk1 lmk_clk2")+port("sysddr","sys_clk_ddr4"),bg="#F4ECF7",line=C["RF_L"])
    s+=hnode("usp_rf_data_converter_0",L_rfdc())
    adc =node("axis_combiner_0","axis_combiner_0\\n(combine ADC0+ADC1)",C["DATA_F"],C["DATA_L"])
    adc+=node("axis_flow_ctrl_0","axis_flow_ctrl_0\\n(capture gate)",C["DATA_F"],C["DATA_L"])
    adc+=node("adc_mem","mem\\nreg-slice / clk-conv / FIFO 32768",C["DATA_F"],C["DATA_L"],shape="box3d")
    adc+=hnode("adc_dma",L_dma("adc_path/axi_dma_0","s2mm"))
    adc+=node("adc_sc","smartconnect_0 (2 MI)",C["MEM_F"],C["MEM_L"])
    s+=cluster("adc_path","adc_path — ADC capture (RX)",adc,bg="#EAF1FB",line=C["DATA_L"])
    dac =hnode("dac_dma",L_dma("dac_path/axi_dma_0","mm2s"))
    dac+=node("dac_mem","mem\\nFIFOs / strm_mux / broadcaster",C["DATA_F"],C["DATA_L"],shape="box3d")
    dac+=node("dac_sc","smartconnect_0 (2 MI)",C["MEM_F"],C["MEM_L"])
    s+=cluster("dac_path","dac_path — DAC playback (TX)",dac,bg="#EAF1FB",line=C["DATA_L"])
    s+=hnode("ddr4_0",L_ddr())
    s+=node("smartconnect_0","smartconnect_0 (root)\\n3 SI -> DDR4",C["MEM_F"],C["MEM_L"])
    s+=hnode("zynq_ultra_ps_e_0",L_ps())
    s+=node("ps8_axi_periph","ps8_axi_periph\\nAXI Interconnect (5 MI)",C["CTRL_F"],C["CTRL_L"])
    s+=node("irq_concat","irq_concat\\nxlconcat (4->1)",C["CTRL_F"],C["CTRL_L"])
    s+=node("clocktreeMTS","clocktreeMTS\\nIBUFDS / clk_wiz / MTS sync",C["CLK_F"],C["CLK_L"])
    s+=node("reset_block","reset_block\\nclk_wiz + 3x proc_sys_reset",C["CLK_F"],C["CLK_L"])
    s+=cluster("ext_out","External Out",
        port("vout","vout00..03 / 10..13","out")+port("ddr4pl","ddr4_pl","out")+
        port("misc","ddr4_led / lmk_rst / gpio_test","out"),bg="#F4ECF7",line=C["RF_L"])
    s+=e_axis("vin","usp_rf_data_converter_0:vin")
    s+=e_axis("usp_rf_data_converter_0:m00","axis_combiner_0","m00")
    s+=e_axis("usp_rf_data_converter_0:m20","axis_combiner_0","m20")
    s+=e_axis("axis_combiner_0","axis_flow_ctrl_0")
    s+=e_axis("axis_flow_ctrl_0","adc_mem")
    s+=e_axis("adc_mem","adc_dma:saxis")
    s+=e_mm("adc_dma:maxi","adc_sc","S2MM")
    s+=e_mm("adc_sc","zynq_ultra_ps_e_0:hp0","PS HP0")
    s+=e_mm("adc_sc","smartconnect_0","PL DDR")
    s+=e_mm("dac_dma:maxi","dac_sc","MM2S")
    s+=e_mm("dac_sc","zynq_ultra_ps_e_0:hp1","PS HP1")
    s+=e_mm("dac_sc","smartconnect_0","PL DDR")
    s+=e_mm("zynq_ultra_ps_e_0:hpm0","smartconnect_0","PS->DDR")
    s+=e_mm("smartconnect_0","ddr4_0:saxi")
    s+=e_mm("ddr4_0:ddr","ddr4pl")
    s+=e_axis("dac_dma:maxis","dac_mem")
    s+=e_axis("dac_mem","usp_rf_data_converter_0:s00","DAC0")
    s+=e_axis("dac_mem","usp_rf_data_converter_0:s10","DAC1")
    s+=e_axis("usp_rf_data_converter_0:vout","vout")
    s+=e_lite("zynq_ultra_ps_e_0:hpm1","ps8_axi_periph","HPM1")
    s+=e_lite("ps8_axi_periph","dac_dma:lite","M00")
    s+=e_lite("ps8_axi_periph","adc_dma:lite","M01")
    s+=e_lite("ps8_axi_periph","usp_rf_data_converter_0:saxi","M02")
    s+=e_lite("ps8_axi_periph","axis_flow_ctrl_0","M03")
    s+=e_lite("ps8_axi_periph","clocktreeMTS","M04")
    s+=e_irq("usp_rf_data_converter_0:irq","irq_concat","In0")
    s+=e_irq("adc_dma:irq","irq_concat","In1")
    s+=e_irq("dac_dma:irq","irq_concat","In2")
    s+=e_irq("clocktreeMTS","irq_concat","In3")
    s+=e_irq("irq_concat","zynq_ultra_ps_e_0:irq")
    s+=e_clk("lmk","clocktreeMTS")
    s+=e_clk("sysddr","clocktreeMTS")
    s+=e_clk("clocktreeMTS","usp_rf_data_converter_0:aclk","rf_clk")
    s+=e_clk("clocktreeMTS","usp_rf_data_converter_0:sysref","sysref")
    s+=e_clk("clocktreeMTS","ddr4_0:sysclk","ddr_clk")
    s+=e_clk("clocktreeMTS","reset_block")
    s+=e_clk("zynq_ultra_ps_e_0:plclk","reset_block","pl_clk0")
    s+=e_rst("reset_block","adc_dma:lite")
    s+=e_rst("reset_block","dac_dma:lite")
    s+=e_rst("ddr4_0:uiclk","reset_block","ui_clk")
    s+=e_ctl("ddr4_0:calib","misc","led")
    s+=e_ctl("zynq_ultra_ps_e_0:gpio","misc","gpio")
    return s+LEG+FOOT

# ===================== 2. DATA PLANE =====================
def diagram_data():
    s=PRE.format(name="data",title="Data Plane — AXI-Stream / DMA / DDR Dataflow")
    s+=port("vin","vin0_01/23 vin2_01/23")
    s+=hnode("usp_rf_data_converter_0",L_rfdc())
    adc =node("axis_combiner_0","axis_combiner_0",C["DATA_F"],C["DATA_L"])
    adc+=node("axis_flow_ctrl_0","axis_flow_ctrl_0",C["DATA_F"],C["DATA_L"])
    ma =node("adc_rs","axis_register_slice_0",C["DATA_F"],C["DATA_L"])
    ma+=node("adc_cc","axis_clock_converter_0\\n(rf->ddr CDC)",C["CDC_F"],C["CLK_L"])
    ma+=node("adc_fifo","axis_data_fifo_0\\n(depth 32768)",C["DATA_F"],C["DATA_L"],shape="box3d")
    adc+=cluster("adc_mem","mem",ma,bg="#DCE9FA",line=C["DATA_L"])
    adc+=hnode("adc_dma",L_dma("adc_path/axi_dma_0","s2mm"))
    adc+=node("adc_sc","smartconnect_0 (2 MI)",C["MEM_F"],C["MEM_L"])
    s+=cluster("adc_path","adc_path — RX capture",adc,bg="#EAF1FB",line=C["DATA_L"])
    dac =hnode("dac_dma",L_dma("dac_path/axi_dma_0","mm2s"))
    md =node("dac_f1","axis_data_fifo_1\\n(async CDC, d16)",C["CDC_F"],C["CLK_L"])
    md+=node("dac_mux","dac_strm_mux\\n(live / replay)",C["DATA_F"],C["DATA_L"])
    md+=node("dac_f2","axis_data_fifo_2\\n(depth 2048)",C["DATA_F"],C["DATA_L"],shape="box3d")
    md+=node("dac_rs","axis_register_slice_0",C["DATA_F"],C["DATA_L"])
    md+=node("dac_b1","axis_broadcaster_1",C["DATA_F"],C["DATA_L"])
    md+=node("dac_f0","axis_data_fifo_0\\n(replay loop, d16)",C["DATA_F"],C["DATA_L"])
    md+=node("dac_b2","axis_broadcaster_2\\n(-> DAC0 / DAC1)",C["DATA_F"],C["DATA_L"])
    dac+=cluster("dac_mem","mem",md,bg="#DCE9FA",line=C["DATA_L"])
    dac+=node("dac_sc","smartconnect_0 (2 MI)",C["MEM_F"],C["MEM_L"])
    s+=cluster("dac_path","dac_path — TX playback",dac,bg="#EAF1FB",line=C["DATA_L"])
    s+=node("smartconnect_0","smartconnect_0 (root)\\n3 SI -> DDR4",C["MEM_F"],C["MEM_L"])
    s+=hnode("ddr4_0",L_ddr())
    s+=port("ddr4pl","ddr4_pl","out")
    s+=hnode("zynq_ultra_ps_e_0",L_ps())
    s+=port("vout","vout00..03 / 10..13","out")
    s+=e_axis("vin","usp_rf_data_converter_0:vin")
    s+=e_axis("usp_rf_data_converter_0:m00","axis_combiner_0","m00_axis")
    s+=e_axis("usp_rf_data_converter_0:m20","axis_combiner_0","m20_axis")
    s+=e_axis("axis_combiner_0","axis_flow_ctrl_0")
    s+=e_axis("axis_flow_ctrl_0","adc_rs")
    s+=e_axis("adc_rs","adc_cc"); s+=e_axis("adc_cc","adc_fifo"); s+=e_axis("adc_fifo","adc_dma:saxis")
    s+=e_mm("adc_dma:maxi","adc_sc","M_AXI_S2MM")
    s+=e_mm("adc_sc","zynq_ultra_ps_e_0:hp0","PS_MEM HP0")
    s+=e_mm("adc_sc","smartconnect_0","PL_MEM")
    s+=e_mm("zynq_ultra_ps_e_0:hp1","dac_sc","HP1 PS_MEM")
    s+=e_axis("dac_dma:maxis","dac_f1","MM2S")
    s+=e_axis("dac_f1","dac_mux","s0"); s+=e_axis("dac_mux","dac_f2","m0")
    s+=e_axis("dac_f2","dac_rs"); s+=e_axis("dac_rs","dac_b1")
    s+=e_axis("dac_b1","dac_f0","M00 replay"); s+=e_axis("dac_f0","dac_mux","s1")
    s+=e_axis("dac_b1","dac_b2","M01")
    s+=e_axis("dac_b2","usp_rf_data_converter_0:s00","DAC0 s00")
    s+=e_axis("dac_b2","usp_rf_data_converter_0:s10","DAC1 s10")
    s+=e_mm("dac_dma:maxi","dac_sc","M_AXI_MM2S")
    s+=e_mm("dac_sc","smartconnect_0","PL_MEM")
    s+=e_mm("zynq_ultra_ps_e_0:hpm0","smartconnect_0","PS->DDR")
    s+=e_mm("smartconnect_0","ddr4_0:saxi","512b")
    s+=e_mm("ddr4_0:ddr","ddr4pl")
    s+=e_axis("usp_rf_data_converter_0:vout","vout")
    return s+LEG+FOOT

# ===================== 3. CONTROL PLANE =====================
def diagram_control():
    s=PRE.format(name="control",title="Control Plane — AXI-Lite Register Access and Interrupts")
    s+=hnode("zynq_ultra_ps_e_0",L_ps())
    s+=node("ps8_axi_periph","ps8_axi_periph\\nAXI Interconnect (5 MI)",C["CTRL_F"],C["CTRL_L"])
    s+=node("irq_concat","irq_concat\\nxlconcat (4->1)",C["CTRL_F"],C["CTRL_L"])
    s+=node("dac_dma_l","dac_path/axi_dma_0\\nS_AXI_LITE  @0xB0002000",C["DATA_F"],C["DATA_L"])
    s+=node("adc_dma_l","adc_path/axi_dma_0\\nS_AXI_LITE  @0xB000A000",C["DATA_F"],C["DATA_L"])
    s+=node("rfdc_l","usp_rf_data_converter_0\\ns_axi  @0xB0040000",C["RF_F"],C["RF_L"])
    s+=node("flow_l","adc_path/axis_flow_ctrl_0\\nS00_AXI  @0xB0020000",C["DATA_F"],C["DATA_L"])
    s+=node("clkw_l","clocktreeMTS/clk_wiz_0\\ns_axi_lite  @0xB0010000",C["CLK_F"],C["CLK_L"])
    s+=node("rfdc_irq","usp_rf_data_converter_0/irq",C["RF_F"],C["RF_L"])
    s+=node("adc_irq","adc_path s2mm_introut",C["DATA_F"],C["DATA_L"])
    s+=node("dac_irq","dac_path mm2s_introut",C["DATA_F"],C["DATA_L"])
    s+=node("clk_irq","clocktreeMTS interrupt",C["CLK_F"],C["CLK_L"])
    s+=e_lite("zynq_ultra_ps_e_0:hpm1","ps8_axi_periph","HPM1")
    s+=e_lite("ps8_axi_periph","dac_dma_l","M00")
    s+=e_lite("ps8_axi_periph","adc_dma_l","M01")
    s+=e_lite("ps8_axi_periph","rfdc_l","M02")
    s+=e_lite("ps8_axi_periph","flow_l","M03")
    s+=e_lite("ps8_axi_periph","clkw_l","M04")
    s+=e_irq("rfdc_irq","irq_concat","In0"); s+=e_irq("adc_irq","irq_concat","In1")
    s+=e_irq("dac_irq","irq_concat","In2"); s+=e_irq("clk_irq","irq_concat","In3")
    s+=e_irq("irq_concat","zynq_ultra_ps_e_0:irq","pl_ps_irq0")
    return s+LEG+FOOT

# ===================== 4. CLOCKING =====================
def diagram_clocking():
    s=PRE.format(name="clocking",title="Clocking / Synchronization Plane")
    s+=cluster("ext","External Clock Inputs",
        port("lmk1","lmk_clk1")+port("lmk2","lmk_clk2")+port("sysddr","sys_clk_ddr4"),
        bg="#FFF9E6",line=C["CLK_L"])
    ct =node("ibuf_pl","IBUFDS_PL_CLK",C["CLK_F"],C["CLK_L"])
    ct+=node("ibuf_sysref","IBUFDS_SYSREF",C["CLK_F"],C["CLK_L"])
    ct+=node("ibuf_ddr","IBUFDS_DDR4",C["CLK_F"],C["CLK_L"])
    ct+=node("clkwiz","clk_wiz_0 (MMCM)",C["CLK_F"],C["CLK_L"],shape="octagon")
    ct+=node("bufg_pl","BUFG_PL_CLK",C["CLK_F"],C["CLK_L"])
    ct+=node("bufg_ddr","BUFG_DDR4",C["CLK_F"],C["CLK_L"])
    ct+=node("mts_sync","sync_0..3\\n(SYSREF CDC -> MTS)",C["CDC_F"],C["CLK_L"])
    s+=cluster("clocktreeMTS","clocktreeMTS",ct,bg="#FFF6DD",line=C["CLK_L"])
    rb =node("rb_clkwiz","clk_wiz_0",C["CLK_F"],C["CLK_L"],shape="octagon")
    rb+=node("rst_ps8","rst_ps8_100M\\nproc_sys_reset",C["CLK_F"],C["RST"])
    rb+=node("rst_ddr4","rst_ddr4_200M\\nproc_sys_reset",C["CLK_F"],C["RST"])
    rb+=node("rst_adc","rst_adc_245M\\nproc_sys_reset",C["CLK_F"],C["RST"])
    rb+=node("blc","binary_latch_counter_0\\n(locked seq.)",C["CLK_F"],C["CLK_L"])
    s+=cluster("reset_block","reset_block",rb,bg="#FFF6DD",line=C["CLK_L"])
    s+=node("rfdc","usp_rf_data_converter_0\\n(axis aclk + user_sysref)",C["RF_F"],C["RF_L"])
    s+=node("ddr4_0","ddr4_0\\n(c0_sys_clk_i)",C["MEM_F"],C["MEM_L"])
    s+=node("ps","zynq_ultra_ps_e_0\\n(pl_clk0 100MHz, pl_resetn0)",C["PS_F"],C["PS_L"])
    s+=node("fabric","adc_path / dac_path / ps8_axi_periph\\n(aclk + aresetn fan-out)",C["DATA_F"],C["DATA_L"])
    s+=e_clk("lmk1","ibuf_pl"); s+=e_clk("lmk2","ibuf_sysref"); s+=e_clk("sysddr","ibuf_ddr")
    s+=e_clk("ibuf_pl","clkwiz","ref")
    s+=e_clk("clkwiz","bufg_pl"); s+=e_clk("clkwiz","bufg_ddr")
    s+=e_clk("bufg_pl","rfdc","rf_clk 245.76"); s+=e_clk("bufg_pl","fabric","rf_clk")
    s+=e_clk("bufg_ddr","ddr4_0","ddr_clk 200")
    s+=e_clk("ibuf_sysref","mts_sync"); s+=e_clk("mts_sync","rfdc","user_sysref")
    s+=e_clk("ddr4_0","fabric","c0_ddr4_ui_clk")
    s+=e_clk("ddr4_0","blc","c0_ddr4_ui_clk")
    s+=e_clk("ps","fabric","pl_clk0"); s+=e_clk("ps","rb_clkwiz","pl_clk0")
    s+=e_clk("bufg_ddr","rb_clkwiz")
    s+=e_ctl("clkwiz","blc","locked")
    s+=e_rst("ps","rb_clkwiz","pl_resetn0")
    s+=e_rst("rst_ps8","fabric","ps8 aresetn")
    s+=e_rst("rst_ddr4","ddr4_0","ddr4 aresetn")
    s+=e_rst("rst_adc","rfdc","rf aresetn")
    s+=e_rst("rst_ddr4","fabric","ddr4 aresetn")
    return s+LEG+FOOT

# ===================== 5. RF FRONT-END =====================
def diagram_rf():
    s=PRE.format(name="rf_frontend",title="Subsystem — RF Front-End (usp_rf_data_converter_0)")
    s+=cluster("adc_in","ADC Analog In",
        port("v001","vin0_01")+port("v023","vin0_23")+port("v201","vin2_01")+port("v223","vin2_23"),
        bg="#F4ECF7",line=C["RF_L"])
    s+=cluster("clkin","Sample Clocks / SYSREF",
        port("a0","adc0_clk")+port("a2","adc2_clk")+port("d0","dac0_clk")+port("d1","dac1_clk")+port("sr","sysref_in"),
        bg="#F4ECF7",line=C["RF_L"])
    s+=hnode("usp_rf_data_converter_0",L_rfdc())
    s+=cluster("dac_out","DAC Analog Out",
        port("o00","vout00..03","out")+port("o10","vout10..13","out"),bg="#F4ECF7",line=C["RF_L"])
    s+=node("adc_path","adc_path\\n(ADC0/ADC1 AXIS out)",C["DATA_F"],C["DATA_L"])
    s+=node("dac_path","dac_path\\n(DAC0/DAC1 AXIS in)",C["DATA_F"],C["DATA_L"])
    s+=node("clocktreeMTS","clocktreeMTS\\n(rf_clk + user_sysref)",C["CLK_F"],C["CLK_L"])
    s+=node("ps8_axi_periph","ps8_axi_periph\\n(M02 -> s_axi)",C["CTRL_F"],C["CTRL_L"])
    s+=node("irq_concat","irq_concat\\n(irq In0)",C["CTRL_F"],C["CTRL_L"])
    s+=node("tieoff","xlconstant_0/1/2\\n(unused s0x/s1x tie-off)","#EEEEEE","#AAAAAA")
    for p in ["v001","v023","v201","v223"]: s+=e_axis(p,"usp_rf_data_converter_0:vin")
    s+=e_axis("usp_rf_data_converter_0:m00","adc_path","m00_axis")
    s+=e_axis("usp_rf_data_converter_0:m20","adc_path","m20_axis")
    s+=e_axis("dac_path","usp_rf_data_converter_0:s00","s00_axis")
    s+=e_axis("dac_path","usp_rf_data_converter_0:s10","s10_axis")
    s+=e_axis("usp_rf_data_converter_0:vout","o00"); s+=e_axis("usp_rf_data_converter_0:vout","o10")
    for p in ["a0","a2","d0","d1"]: s+=e_clk(p,"usp_rf_data_converter_0:smpclk")
    s+=e_clk("sr","usp_rf_data_converter_0:sysref","sysref_in")
    s+=e_clk("clocktreeMTS","usp_rf_data_converter_0:aclk","rf_clk")
    s+=e_clk("clocktreeMTS","usp_rf_data_converter_0:sysref","user_sysref")
    s+=e_lite("ps8_axi_periph","usp_rf_data_converter_0:saxi","s_axi")
    s+=e_irq("usp_rf_data_converter_0:irq","irq_concat","irq")
    s+=e_ctl("tieoff","usp_rf_data_converter_0:s10","s01/02/03 s11/12/13")
    return s+LEG+FOOT

# ===================== 6. DDR4 MEMORY =====================
def diagram_ddr():
    s=PRE.format(name="ddr4_memory",title="Subsystem — DDR4 Memory")
    s+=node("adc_sc","adc_path/smartconnect_0\\n(M01 PL_MEM_AXI)",C["MEM_F"],C["MEM_L"])
    s+=node("dac_sc","dac_path/smartconnect_0\\n(M01 PL_MEM_AXI)",C["MEM_F"],C["MEM_L"])
    s+=node("ps","zynq_ultra_ps_e_0\\n(M_AXI_HPM0_FPD)",C["PS_F"],C["PS_L"])
    s+=node("smartconnect_0","smartconnect_0 (root)\\nSmartConnect 3 SI -> 1 M",C["MEM_F"],C["MEM_L"])
    s+=hnode("ddr4_0",L_ddr())
    s+=port("ddr4pl","ddr4_pl (external DRAM)","out")
    s+=port("sysddr","sys_clk_ddr4"); s+=port("led","ddr4_led","out")
    s+=node("clocktreeMTS","clocktreeMTS\\n(ddr_clk)",C["CLK_F"],C["CLK_L"])
    s+=node("reset_block","reset_block\\n(sys_rst, c0_ddr4_aresetn)",C["CLK_F"],C["RST"])
    s+=e_mm("adc_sc","smartconnect_0","S01_AXI")
    s+=e_mm("dac_sc","smartconnect_0","S02_AXI")
    s+=e_mm("ps","smartconnect_0","S00_AXI HPM0")
    s+=e_mm("smartconnect_0","ddr4_0:saxi","M00 C0_DDR4_S_AXI")
    s+=e_mm("ddr4_0:ddr","ddr4pl","C0_DDR4")
    s+=e_clk("sysddr","clocktreeMTS")
    s+=e_clk("clocktreeMTS","ddr4_0:sysclk","c0_sys_clk_i")
    s+=e_clk("ddr4_0:uiclk","reset_block","c0_ddr4_ui_clk")
    s+=e_rst("reset_block","ddr4_0:arstn","c0_ddr4_aresetn / sys_rst")
    s+=e_ctl("ddr4_0:calib","led","init_calib_complete")
    return s+LEG+FOOT

# ===================== 7. GPIO / BOARD CONTROL =====================
def diagram_gpio():
    s=PRE.format(name="gpio_control",title="Subsystem — GPIO / Board Control (PS EMIO fan-out)")
    s+=node("ps","zynq_ultra_ps_e_0\\nemio_gpio_o[94:0]",C["PS_F"],C["PS_L"])
    s+=node("xls0","xlslice_0 (bit 84)","#EEEEEE","#AAAAAA")
    s+=node("xls1","xlslice_1 (bit 80)","#EEEEEE","#AAAAAA")
    s+=node("adc_ctl","clocktreeMTS/adc_control\\n(xlslice bit 34)","#EEEEEE","#AAAAAA")
    s+=node("dac_ctl","clocktreeMTS/dac_control\\n(xlslice bit 2)","#EEEEEE","#AAAAAA")
    s+=node("adc_path","adc_path\\n(capture enable)",C["DATA_F"],C["DATA_L"])
    s+=node("dac_path","dac_path\\n(playback enable)",C["DATA_F"],C["DATA_L"])
    s+=port("lmk_rst","lmk_rst","out"); s+=port("gpio_test","gpio_test","out")
    s+=e_ctl("ps","xls0","[94:0]"); s+=e_ctl("ps","xls1")
    s+=e_ctl("ps","adc_ctl","Din"); s+=e_ctl("ps","dac_ctl","Din")
    s+=e_ctl("xls0","lmk_rst"); s+=e_ctl("xls1","gpio_test")
    s+=e_ctl("adc_ctl","adc_path","adc_control"); s+=e_ctl("dac_ctl","dac_path","dac_control")
    return s+LEG+FOOT

DIAGRAMS={"1_complete_architecture":diagram_complete,"2_data_plane":diagram_data,
    "3_control_plane":diagram_control,"4_clocking_sync":diagram_clocking,
    "5_subsystem_rf_frontend":diagram_rf,"6_subsystem_ddr4_memory":diagram_ddr,
    "7_subsystem_gpio_control":diagram_gpio}
for fn,b in DIAGRAMS.items():
    p=os.path.join(DIAG,fn+".dot")
    open(p,"w",encoding="utf-8").write(b())
    print("wrote",p)
print("DONE")
