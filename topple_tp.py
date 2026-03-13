#!/usr/bin/env python3.11

import gurobipy as gp
from gurobipy import GRB
import sys
from functools import cmp_to_key
import re
from collections import defaultdict
import numpy as np
import json  # <-- ADDED

SINGLE_GOAL = 0
MAX_HEIGHT = 4
outfile = "out"
CANTOPPLE = True
if len(sys.argv)>1:
    CANTOPPLE = bool(int(sys.argv[1]))
    if CANTOPPLE:
        outfile = outfile+".topple"
    else:
        outfile = outfile+".notopple"
numo=0
if len(sys.argv)>2:
    numo = int(sys.argv[2])
SEED = 0
if len(sys.argv)>3:
    SEED = int(sys.argv[3])
TIME = 10
if len(sys.argv)>4:
    TIME = int(sys.argv[4])
NUMOBJ = numo
NUMLOC = min(8, NUMOBJ)
if len(sys.argv)>5:
    NUMLOC = int(sys.argv[5])
TIMESTEPS = NUMOBJ*4+3#*NUMOBJ
_infty = 100000
_small_infty = 100
_epsilon = .00001
_large_epsilon = 1
_extra = _epsilon#.1
_transfer_base_cost = 2
_topple_base_cost = 1

commodities = ["o"+str(i) for i in range(NUMOBJ)]
nodes = []
arcs = []
capacities = dict()
ncapacities = dict()
costs = dict()
actions = dict()
adjacencies = dict()
incidences = dict()

# Static pad layout (kept identical for every problem file you generate)
PAD_COORDS = {
    1: {'x': 0.55, 'y': -0.30},
    2: {'x': 0.55, 'y': -0.10},
    3: {'x': 0.55, 'y':  0.10},
    4: {'x': 0.55, 'y':  0.30},
    5: {'x': 0.30, 'y': -0.30},
    6: {'x': 0.30, 'y': -0.10},
    7: {'x': 0.30, 'y':  0.10},
    8: {'x': 0.30, 'y':  0.30},
    9: {'x': 0.55, 'y': -0.50},
    10: {'x': 0.55, 'y': 0.50},
    11: {'x': 0.30, 'y': -0.50},
    12: {'x': 0.30, 'y':  0.50},
    13: {'x': 0.40, 'y': -0.30},
    14: {'x': 0.40, 'y': -0.10},
    15: {'x': 0.40, 'y':  0.10},
    16: {'x': 0.40, 'y':  0.30},
    17: {'x': 0.40, 'y': -0.50},
    18: {'x': 0.40, 'y':  0.50},
}

def get_action(key):
    global actions
    if key in actions:
        return actions[key]
    return ""

def get_int_cost(cost):
    return int(np.ceil(cost))

def add_arc(u, v, capacity, cost, action):
    global arcs
    global capacities
    global costs
    global actions
    global adjacencies
    global incidences
    if (u,v) in arcs:
        print("Arc ",u, v, capacity, cost, action," already exists")
        return
    arcs.append((u,v))
    capacities[arcs[-1]] = capacity
    costs[arcs[-1]] = get_int_cost(cost)
    actions[arcs[-1]] = action
    adjacencies[u].append(v)
    incidences[v].append(u)

def get_inter_location_distance(lu, lv):
    if lu>=NUMLOC or lv>=NUMLOC:
        return 2
    else:
        ux = PAD_COORDS[lu+1]['x']
        uy = PAD_COORDS[lu+1]['y']
        vx = PAD_COORDS[lv+1]['x']
        vy = PAD_COORDS[lv+1]['y']
        return get_int_cost(np.ceil(np.sqrt((ux-vx)*(ux-vx) + (uy-vy)*(uy-vy))))

transferarcs = dict()
starttopplearcs = dict()
looptopplearcs = dict()
for loc in range(NUMLOC):
    transferarcs[loc] = []
    starttopplearcs[loc] = []
    looptopplearcs[loc] = []

for loc in range(NUMLOC):
    #stack at location
    for stackpos in range(NUMOBJ):
        nodes.append("l_"+str(loc)+"_"+str(stackpos))
        ncapacities[nodes[-1]] = 1
        adjacencies[nodes[-1]] = []
        incidences[nodes[-1]] = []
        add_arc(nodes[-1],nodes[-1],ncapacities[nodes[-1]],0,"")
        if(stackpos > 0):
            add_arc("l_"+str(loc)+"_"+str(stackpos),"l_"+str(loc)+"_"+str(stackpos-1),1,0,"")
            add_arc("l_"+str(loc)+"_"+str(stackpos-1),"l_"+str(loc)+"_"+str(stackpos),1,0,"")
    for loc2 in range(loc):
        add_arc("l_"+str(loc2)+"_"+str(NUMOBJ-1),"l_"+str(loc)+"_"+str(NUMOBJ-1),1, _transfer_base_cost*_small_infty + get_inter_location_distance(loc,loc2) ,"TRANSFER")
        transferarcs[loc].append(("l_"+str(loc2)+"_"+str(NUMOBJ-1),"l_"+str(loc)+"_"+str(NUMOBJ-1)))
        transferarcs[loc2].append(("l_"+str(loc2)+"_"+str(NUMOBJ-1),"l_"+str(loc)+"_"+str(NUMOBJ-1)))
        add_arc("l_"+str(loc)+"_"+str(NUMOBJ-1),"l_"+str(loc2)+"_"+str(NUMOBJ-1),1, _transfer_base_cost*_small_infty + get_inter_location_distance(loc,loc2) ,"TRANSFER")
        transferarcs[loc].append(("l_"+str(loc)+"_"+str(NUMOBJ-1),"l_"+str(loc2)+"_"+str(NUMOBJ-1)))
        transferarcs[loc2].append(("l_"+str(loc)+"_"+str(NUMOBJ-1),"l_"+str(loc2)+"_"+str(NUMOBJ-1)))

if CANTOPPLE:
    nodes.append("table")
    ncapacities[nodes[-1]] = NUMOBJ
    adjacencies[nodes[-1]] = []
    incidences[nodes[-1]] = []
    add_arc(nodes[-1],nodes[-1],ncapacities[nodes[-1]],0,"")
    for loc in range(NUMLOC):
        nodes.append("p_"+str(loc)+"_"+str(NUMOBJ))
        ncapacities[nodes[-1]] = NUMOBJ
        adjacencies[nodes[-1]] = []
        incidences[nodes[-1]] = []
        add_arc(nodes[-1],nodes[-1],ncapacities[nodes[-1]],0,"")
        looptopplearcs[loc].append(arcs[-1])
        add_arc("l_"+str(loc)+"_"+str(NUMOBJ-1), nodes[-1], NUMOBJ, 0, "")
        starttopplearcs[loc].append(arcs[-1])
        add_arc(nodes[-1], "table", NUMOBJ, (_topple_base_cost+_extra)*_small_infty + 1, "TOPPLE")

    for obj in range(NUMOBJ):
        nodes.append("on_"+str(NUMLOC+obj)+"_0")
        ncapacities[nodes[-1]] = 1
        adjacencies[nodes[-1]] = []
        incidences[nodes[-1]] = []
        add_arc(nodes[-1],nodes[-1],ncapacities[nodes[-1]],0,"")
        add_arc("table", nodes[-1], 1, 0, "")
        for loc in range(NUMLOC):
            add_arc(nodes[-1], "l_"+str(loc)+"_"+str(NUMOBJ-1), 1, get_int_cost((_transfer_base_cost+_extra)*_small_infty) + get_inter_location_distance(loc, NUMOBJ-1), "TRANSFER")

print("Nodes")
for node in nodes:
    print(node, "\n -- ", adjacencies[node], "\n -- ", incidences[node])
print("Arcs")
for arc in arcs:
    print(arc, capacities[arc], costs[arc], actions[arc])

#####################################################################
#####PROBLEM
problem = dict()

if NUMOBJ == 9:
    COLOURS = ['Red1', 'Red2', 'Red3', 'Green1', 'Green2', 'Green3', 'Blue1', 'Blue2', 'Blue3']
elif NUMOBJ == 6:
    COLOURS = ['Red1', 'Red2', 'Red3', 'Green1', 'Green2', 'Green3']
elif NUMOBJ == 4:
    COLOURS = ['Red1', 'Red2', 'Green1', 'Green2']
else:
    COLOURS = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Purple', 'Cyan', 'Magenta']

import random
from collections import defaultdict
random.seed(SEED)
locindicesi = [0 for loc in range(NUMLOC)]
locindicese = [0 for loc in range(NUMLOC)]

order = list(range(NUMOBJ))
random.shuffle(order)

stacks = [[], [], []]

if SINGLE_GOAL == 1:
    k = NUMOBJ // 2
    for n in order[:k]:
        stacks[0].append(n)
    for n in order[k:]:
        p = random.randrange(3)
        stacks[p].append(n)
else:
    k = 0
    for n in order[k:]:
        while True:
            p = random.randrange(3)
            if len(stacks[p]) > MAX_HEIGHT:
                continue
            stacks[p].append(n)
            break

pos = {}
for p in range(3):
    for h, n in enumerate(stacks[p]):
        pos[n] = (p, h)

problem = {}
P_ = min(NUMOBJ//2, 3)
gi = random.randint(0, NUMOBJ-1)
for i in range(NUMOBJ):
    p, h = pos[i]
    init_label = f"l_{p}_{h}"
    if SINGLE_GOAL == 1:
        if i == gi:
            goal_loc = "l_3_0"
        else:
            goal_loc = ""
    else:
        goal_loc = f"l_{i//P_+1}_{i%P_}"
    problem[f"o{i}"] = (init_label, goal_loc)

for k, (start_pose, end_opts) in problem.items():
    print(f"{k}: start={start_pose} | end_options({len(end_opts)}): {end_opts}")

m = gp.Model("netflow")
modelvars = dict()
tcosts = dict()

for t in range(TIMESTEPS):
    for c in commodities:
        for arc in arcs:
            inflation = 1
            cost = costs[arc]
            if cost == 0:
                cost = _epsilon
            if arc[0]==arc[1]:
                inflation = 1
                cost = 0
            tcosts[t,arc] = cost*inflation
            modelvars[(t,c,arc[0],arc[1])] = m.addVar(
                lb=0.0, ub=capacities[arc], obj=cost*inflation,
                vtype=GRB.INTEGER,
                name=str(t)+str(c)+str(arc[0])+str(arc[1])
            )

for t in range(TIMESTEPS):
    for arc in arcs:
        m.addConstr(gp.quicksum(modelvars[t,c,arc[0],arc[1]] for c in commodities) <= capacities[arc])

for t in range(TIMESTEPS):
    for node in nodes:
        m.addConstr(gp.quicksum(modelvars[t,c,node,nout] for nout in adjacencies[node] for c in commodities) <= ncapacities[node])
        m.addConstr(gp.quicksum(modelvars[t,c,nin,node] for nin in incidences[node] for c in commodities) <= ncapacities[node])

    already_crossed = []
    for arc in arcs:
        rarc = (arc[1], arc[0])
        if arc[0] == arc[1]:
            continue
        if arc not in already_crossed and rarc in arcs:
            if capacities[arc] != capacities[rarc]:
                print("Forward and reverse edges need to have the same capacity")
            m.addConstr(
                gp.quicksum(modelvars[t,c,arc[0],arc[1]] for c in commodities) +
                gp.quicksum(modelvars[t,c,rarc[0],rarc[1]] for c in commodities)
                <= capacities[arc]
            )
            already_crossed.append(arc)
            already_crossed.append(rarc)

    if t==0:
        continue

    for c in commodities:
        for node in nodes:
            m.addConstr(gp.quicksum(modelvars[t-1,c,nin,node] for nin in incidences[node]) ==
                        gp.quicksum(modelvars[t,c,node,nout] for nout in adjacencies[node]))

loopback = []

for c in commodities:
    loop = m.addVar(lb=0.0, ub=1, vtype=GRB.INTEGER, name="-1"+str(c)+str(-1)+str(-1))
    loopback.append(loop)
    modelvars[-1,c,-1,-1] = loop
    costs[(-1,-1)] = 0
    capacities[(-1,-1)] = 1

    m.addConstr(loop == 1)

    start = problem[c][0]
    print("start", start)
    m.addConstr(loop == gp.quicksum(modelvars[0,c,start,nout] for nout in adjacencies[start]))

    for arc in arcs:
        if arc[0] != start or arc[1] not in adjacencies[start]:
            m.addConstr(modelvars[0,c,arc[0],arc[1]] == 0)

    end = problem[c][1]
    print("end", end)
    if end != "":
        if "," not in end:
            m.addConstr(gp.quicksum(modelvars[TIMESTEPS-1,c,nin,end] for nin in incidences[end]) == loop)
            for arc in arcs:
                if arc[1] != end or arc[0] not in incidences[end]:
                    m.addConstr(modelvars[TIMESTEPS-1,c,arc[0],arc[1]] == 0)
        else:
            ends_vec = end.split(",")
            all_incidences = []
            for n in ends_vec:
                if len(n.split("_"))==3:
                    if int(n.split("_")[2])<NUMOBJ:
                        for nn in incidences[n]:
                            all_incidences.append((nn, n))
            m.addConstr(gp.quicksum(modelvars[TIMESTEPS-1,c,nin[0],nin[1]] for nin in all_incidences) == loop)
    else:
        all_incidences = []
        for n in nodes:
            if len(n.split("_"))==3:
                if int(n.split("_")[2])<NUMOBJ:
                    for nn in incidences[n]:
                        all_incidences.append((nn, n))
        m.addConstr(gp.quicksum(modelvars[TIMESTEPS-1,c,nin[0],nin[1]] for nin in all_incidences) == loop)

####Indicators
indicatorvals = dict()
for t in range(TIMESTEPS):
    for arc in arcs:
        M = NUMOBJ+1
        indicatorvals[t,arc[0],arc[1]] = m.addVar(vtype=GRB.BINARY)
        if capacities[arc] == 1:
            m.addConstr(indicatorvals[t,arc[0],arc[1]] == gp.quicksum(modelvars[t,c,arc[0],arc[1]] for c in commodities))
        else:
            m.addConstr(gp.quicksum(modelvars[t,c,arc[0],arc[1]] for c in commodities) + M*(1-indicatorvals[t,arc[0],arc[1]]) <= M)

####TOPPLEFIX
for t in range(TIMESTEPS):
    for loc in range(NUMLOC):
        m.addConstr(gp.quicksum(indicatorvals[t, arc[0], arc[1]] for arc in transferarcs[loc]) +
                    gp.quicksum(indicatorvals[t, arc[0], arc[1]] for arc in starttopplearcs[loc]) <= 1)
        m.addConstr(gp.quicksum(indicatorvals[t, arc[0], arc[1]] for arc in transferarcs[loc]) +
                    gp.quicksum(indicatorvals[t, arc[0], arc[1]] for arc in looptopplearcs[loc]) <= 1)

m.setObjective(gp.quicksum(indicatorvals[t,arc[0],arc[1]]*tcosts[t,arc] for arc in arcs for t in range(TIMESTEPS)), GRB.MINIMIZE)

# ------------------- ADDED: First-solution callback capture -------------------
FIRST_SOL = {
    "time_sec": None,
    "obj": None,
    # store compact representation: list of (t, obj, u, v) for action arcs only
    "action_tuples": None,
}

def _gurobi_first_solution_cb(model, where):
    """
    Record runtime + action tuples for the first incumbent solution Gurobi finds.
    Trigger: first MIPSOL callback.
    """
    global FIRST_SOL
    if where != GRB.Callback.MIPSOL:
        return
    if FIRST_SOL["time_sec"] is not None:
        return  # already captured

    try:
        t_runtime = float(model.cbGet(GRB.Callback.RUNTIME))
    except Exception:
        t_runtime = None

    try:
        obj = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
    except Exception:
        obj = None

    action_tuples = []
    try:
        # Iterate modelvars, but only keep action arcs with val>epsilon
        for key, var in model._modelvars.items():
            # key = (t, c, u, v)
            t, c, u, v = key
            act = model._get_action((u, v))
            if act == "":
                continue
            val = model.cbGetSolution(var)
            if val is None:
                continue
            if val > _epsilon:
                action_tuples.append((t, c, u, v))
    except Exception as e:
        # still record time/obj if we can
        print(f"[WARN] callback could not extract first solution actions: {e}")

    FIRST_SOL["time_sec"] = t_runtime
    FIRST_SOL["obj"] = obj
    FIRST_SOL["action_tuples"] = action_tuples
    print(f"[INFO] First incumbent captured at t={t_runtime}s, obj={obj}, n_action_tuples={len(action_tuples)}")
# ---------------------------------------------------------------------------

m.setParam(GRB.Param.TimeLimit, TIME)

# Attach what callback needs (minimal intrusion)
m._modelvars = modelvars
m._get_action = get_action

# Run optimize with callback
m.optimize(_gurobi_first_solution_cb)

def locindex(ac):
    u = ac[0]
    v = ac[1]
    lint = -1
    hint = -1
    try:
        lint = int(u.split("_")[1])
        hint = int(u.split("_")[2])
    except:
        pass
    lend = -1
    hend = -1
    try:
        lend = int(v.split("_")[1])
        hend = int(v.split("_")[2])
    except:
        pass
    return (lint,hint,lend,hend)

def compare(act1, act2):
    uv1 = locindex(act1)
    uv2 = locindex(act2)
    if uv1[0] == uv2[0]:
        if uv1[1] < uv2[1]:
            return 1
        if uv1[1] > uv2[1]:
            return -1
        return 0
    elif uv1[2] == uv2[0]:
        return 1
    elif uv1[0] == uv2[2]:
        return -1
    else:
        if uv1[0] < uv2[0]:
            return 1
        if uv1[0] > uv2[0]:
            return -1
        return 0

def cmp_to_key(mycmp):
    class K(object):
        __slots__ = ['obj']
        def __init__(self, obj):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        __hash__ = None
    return K

# Print solution
if m.SolCount > 0:
    print("------------------")
    for t in range(TIMESTEPS):
        for c in commodities:
            for arc in arcs:
                var = modelvars[(t,c,arc[0],arc[1])]
                if True:
                    action = get_action(arc)
    print("------------------")
    for var in modelvars:
        if modelvars[var].X > 0:
            obj = var[0]
            u = var[2]
            v = var[3]
            action = get_action((u,v))
    print("------------------")
    solvars = []
    solactions = dict()
    for t in range(TIMESTEPS):
        solactions[t] = []
    for var in modelvars:
        if modelvars[var].X > _epsilon:
            obj = var[1]
            u = var[2]
            v = var[3]
            action = get_action((u,v))
            if action != "":
                solvars.append(var)
                lint = int(u.split("_")[1])
                hint = int(u.split("_")[2])
                print(var, action, modelvars[var].X, costs[(u,v)])
                solactions[var[0]].append((lint,hint,var))
    print("#####SOLUTION")

    for t in range(TIMESTEPS):
        currentaction = dict()
        for act in solactions[t]:
            var = act[2]
            obj = var[1]
            u = var[2]
            v = var[3]
            if (u,v) not in currentaction.keys():
                source = u
                sink = v
                try:
                    source = "l"+str(u.split("_")[1])
                except:
                    pass
                try:
                    sink = "l"+str(v.split("_")[1])
                except:
                    pass
                currentaction[(u,v)] = [get_action((u,v))+" "+source+"->"+sink+" : "]
            else:
                if not CANTOPPLE:
                    print("!!!Unexpected: This should never happen.",var)
                else:
                    pass
            currentaction[(u,v)].append(obj)

        keys = [k for k in currentaction.keys()]
        keys.sort(key=cmp_to_key(compare))
        for key in keys:
            print("".join(currentaction[key]))
    solcost = int(m.getObjective().getValue())
    print("Objective: ",solcost)

import yaml
from functools import cmp_to_key

# ---------------------------------------------------------------------------
OBJ_TO_COLOUR = {f"o{i}": colour for i, colour in enumerate(COLOURS)}

# ---------------------------------------------------------------------------
def _rebuild_solution_lines(solactions, TIMESTEPS):
    lines = []

    def parse_loc(tok):
        if tok == "table":
            return ("table", None, None)
        m = re.match(r"^([A-Za-z]+)_(\-?\d+)_([\-]?\d+)$", tok)
        if not m:
            return (tok, None, None)
        kind, pos, h = m.group(1), int(m.group(2)), int(m.group(3))
        return (kind, pos, h)

    def pretty(tok):
        kind, pos, h = parse_loc(tok)
        if kind == "table":
            return "table"
        if pos is None or h is None:
            return tok
        return f"l{pos}_{h}"

    for t in range(TIMESTEPS):
        groups = {}
        for act in solactions.get(t, []):
            _, _, var = act
            obj, u, v = var[1], var[2], var[3]
            groups.setdefault((u, v), []).append(obj)

        for (u, v) in sorted(groups.keys(), key=cmp_to_key(compare)):
            objs = groups[(u, v)]
            action_name = get_action((u, v))
            src = pretty(u)
            dst = pretty(v)

            if v == "table":
                lines.append(f"{action_name} {src}->{dst} : " + "".join(objs))
            else:
                lines.append(f"{action_name} {src}->{dst} : " + "".join(objs))

    return lines

# ------------------- ADDED: build first-solution action lines + write JSON ----
def _build_solactions_from_action_tuples(action_tuples):
    """
    action_tuples: list of (t, c, u, v) for action arcs with positive flow.
    Return solactions-like dict: solactions[t] = [(lint,hint,(t,obj,u,v)), ...]
    """
    if not action_tuples:
        return {t: [] for t in range(TIMESTEPS)}

    out = {t: [] for t in range(TIMESTEPS)}
    for (t, c, u, v) in action_tuples:
        # match existing structure: var tuple is (t, obj, u, v)
        var = (t, c, u, v)
        try:
            lint = int(u.split("_")[1])
            hint = int(u.split("_")[2])
        except Exception:
            lint, hint = -1, -1
        out[t].append((lint, hint, var))
    return out

def _write_first_solution_stats_json(path="my_gurobi_first_solution.json"):
    """
    Write time + first-solution action lines (if captured) to JSON.
    """
    sol_time = FIRST_SOL.get("time_sec", None)
    sol_obj = FIRST_SOL.get("obj", None)
    tuples = FIRST_SOL.get("action_tuples", None)

    first_lines = []
    if tuples:
        try:
            solactions_first = _build_solactions_from_action_tuples(tuples)
            first_lines = _rebuild_solution_lines(solactions_first, TIMESTEPS)
        except Exception as e:
            print(f"[WARN] could not rebuild first-solution lines: {e}")

    payload = {
        "first_solution_time_sec": sol_time,
        "first_solution_obj": sol_obj,
        "first_solution_lines": first_lines,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] first gurobi solution stats written to '{path}'")
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def _yaml_inline_block(d):
    return '{' + ', '.join(f'{k}: {v}' for k, v in d.items()) + '}'

def _dump_manual_yaml(file_name, pads, initial, goal, actions):
    lines = []
    lines.append('pads: ')
    for idx in sorted(pads):
        p = pads[idx]
        lines.append(f'  {idx}: {{x: {p["x"]:6.2f}, y: {p["y"]:6.2f}}}')

    lines += [
        '',
        'z_base:      0.025    # table + 1/2 block',
        'z_increment: 0.05     # full block height',
        '# ----------------------------------------------------------',
        ''
    ]

    lines.append('initial:')
    for item in initial:
        entry = _yaml_inline_block({'name': item["name"], 'pad': item["pad"], 'height': item["height"]})
        lines.append(f'  - {entry}')
    lines.append('')

    lines.append('goal:')
    for item in goal:
        entry = _yaml_inline_block({'name': item["name"], 'pad': item["pad"], 'height': item["height"]})
        lines.append(f'  - {entry}')
    lines.append('')

    lines.append('actions:     # ordered action sequence')
    for act in actions:
        print(act)
        lines.append(f'  - type:  {act["type"]}')

        if act['type'] == 'Smash':
            seq = act.get('blocks')
            lines.append(f'    blocks: [{", ".join(seq)}]')
            lines.append('    from: ' + _yaml_inline_block(act['from']))
            lines.append('    dir:   +x')
        else:
            lines.append('    block: ' + act['block'])
            if 'from' in act:
                lines.append('    from: ' + _yaml_inline_block(act['from']))
            lines.append('    to:   ' + _yaml_inline_block(act['to']))
        lines.append('')

    with open(file_name, 'w') as fh:
        fh.write('\n'.join(lines))

def convert_to_yaml(problem, solactions, TIMESTEPS, file_name='problem.yaml'):
    n_objs = len(problem)
    total_pads = 2 * n_objs
    pad_stacks = {i: [] for i in range(total_pads)}

    initial_yml = []
    pad_buckets = defaultdict(list)

    for obj, (init, _) in problem.items():
        pad, h = map(int, init.split('_')[1:])
        pad_buckets[pad].append((h, obj))
        initial_yml.append({'name': OBJ_TO_COLOUR[obj], 'pad': pad + 1, 'height': h})

    for p in range(total_pads):
        items = pad_buckets.get(p, [])
        pad_stacks[p] = [obj for h, obj in sorted(items, key=lambda t: t[0], reverse=True)]

    goal_yml = []
    for obj, (_, goals) in problem.items():
        if goals:
            goal_tokens = [g.strip() for g in goals.split(',') if g.strip()]
            pads, heights = [], []
            for g in goal_tokens:
                gpad, gh = map(int, g.split('_')[1:])
                pads.append(gpad + 1)
                heights.append(gh)

            entry = {
                'name': OBJ_TO_COLOUR[obj],
                'pad': pads[0] if len(pads) == 1 else pads,
                'height': heights[0] if len(heights) == 1 else heights,
            }
            goal_yml.append(entry)
        else:
            goal_yml.append({'name': OBJ_TO_COLOUR[obj], 'pad': '-1', 'height': '-1'})

    sol_lines = _rebuild_solution_lines(solactions, TIMESTEPS)
    print("rebuilt sol lines:", sol_lines)

    re_topple = re.compile(r'^TOPPLE\s+l(-?\d+)_(-?\d+)->table\s*:\s*((?:o\d+)+)$')
    re_move   = re.compile(r'^(?:TRANSFER|MOVE)\s+l(-?\d+)_(-?\d+)->l(-?\d+)_(-?\d+)\s*:\s*(o\d+)$')

    actions_yml = []

    for line in sol_lines:
        line = line.strip()

        m = re_move.match(line)
        if m:
            s_pad = int(m.group(1))
            d_pad = int(m.group(3))
            obj   = m.group(5)

            pad_stacks.setdefault(s_pad, [])
            pad_stacks.setdefault(d_pad, [])

            if obj in pad_stacks[s_pad]:
                i_src = pad_stacks[s_pad].index(obj)
                h_src = len(pad_stacks[s_pad]) - 1 - i_src
                pad_stacks[s_pad].pop(i_src)
            else:
                h_src = 0

            h_dst = len(pad_stacks[d_pad])
            pad_stacks[d_pad].insert(0, obj)

            actions_yml.append({
                'block': OBJ_TO_COLOUR[obj],
                'type' : 'Move',
                'from' : {'pad': s_pad + 1, 'height': h_src},
                'to'   : {'pad': d_pad + 1, 'height': h_dst},
            })
            continue

        m = re_topple.match(line)
        if m:
            s_pad = int(m.group(1))
            objs  = re.findall(r'o\d+', m.group(3))

            obj_pad_map = {}
            for o in objs:
                found_pad = None
                for p, stk in pad_stacks.items():
                    if o in stk:
                        found_pad = p
                        break
                obj_pad_map[o] = found_pad

            pads_present = set(v for v in obj_pad_map.values() if v is not None)
            if (None in obj_pad_map.values()) or len(pads_present) != 1 or (s_pad not in pads_present):
                print(f"[ERROR] TOPPLE objects must all be on pad {s_pad}; got {obj_pad_map}")
                sys.exit(-10)

            stack = pad_stacks[s_pad]
            idx_map = {o: stack.index(o) for o in objs}
            bottom_in_stack = max(idx_map.items(), key=lambda kv: kv[1])[0]
            if objs[-1] != bottom_in_stack:
                print(f"[WARN] Last object in TOPPLE line ({objs[-1]}) is not bottom in stack (bottom is {bottom_in_stack}).")

            bottom_obj = bottom_in_stack

            i_bot    = stack.index(bottom_obj)
            tower_top_to_bottom = stack[:i_bot + 1]
            tower_bottom_to_top = list(reversed(tower_top_to_bottom))
            h_bottom = len(stack) - 1 - i_bot
            pad_stacks[s_pad] = stack[i_bot + 1:]

            actions_yml.append({
                'blocks': [OBJ_TO_COLOUR[o] for o in tower_bottom_to_top],
                'type'  : 'Smash',
                'from'  : {'pad': s_pad + 1, 'height': h_bottom},
                'dir'   : '+x',
            })
            continue

        raise ValueError(f"Un-recognised action line:\n  {line}")

    print("stacks:", pad_stacks)

    _dump_manual_yaml(file_name, PAD_COORDS, initial_yml, goal_yml, actions_yml)
    print(f"[OK] problem description written to '{file_name}'")

SDF_HEADER = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="blocks_world">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin
      filename="ignition-gazebo-physics-system"
      name="gz::sim::systems::Physics">
    </plugin>
    <plugin
      filename="ignition-gazebo-user-commands-system"
      name="gz::sim::systems::UserCommands">
    </plugin>
    <plugin
      filename="ignition-gazebo-scene-broadcaster-system"
      name="gz::sim::systems::SceneBroadcaster">
    </plugin>
    <plugin
      filename="ignition-gazebo-contact-system"
      name="gz::sim::systems::Contact">
    </plugin>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
      </link>
    </model>
"""

SDF_FOOTER = """  </world>
</sdf>
"""

def _format_float(val, two_dp=False):
    if two_dp:
        return f"{val:0.3f}"
    s = f"{val:0.3f}".rstrip('0').rstrip('.')
    return s if '.' in s else s + '.0'

def generate_sdf(problem, pad_coords=PAD_COORDS, z_increment=0.05, file_name='blocks_world.sdf'):
    include_blocks = []
    for obj, (init_state, _) in problem.items():
        pad_idx, height = map(int, init_state.split('_')[1:])
        pad_key = pad_idx + 1
        coords  = pad_coords[pad_key]
        x_str   = _format_float(coords['x'], two_dp=True)
        y_str   = _format_float(coords['y'], two_dp=True)
        z_str   = _format_float(height * z_increment)
        colour  = OBJ_TO_COLOUR[obj].lower()
        include_blocks.append(
            f"""    <include>
        <uri>model://wood_cube_{colour}</uri>
        <static>false</static>
        <pose>{x_str} {y_str} {z_str} 0 0 0</pose>
    </include>"""
        )

    with open(file_name, 'w') as fh:
        fh.write(SDF_HEADER + "\n")
        fh.write("\n".join(include_blocks) + "\n\n")
        fh.write(SDF_FOOTER)

    print(f"[OK] SDF scene written to '{file_name}'")

# ---------------------------------------------------------------------------
# ADDED: write first-solution stats JSON before emitting YAML/SDF (same dir)
_write_first_solution_stats_json("my_gurobi_first_solution.json")

convert_to_yaml(problem, solactions, TIMESTEPS, 'my_problem.yaml')
generate_sdf(problem, file_name='my_blocks_world.sdf')
