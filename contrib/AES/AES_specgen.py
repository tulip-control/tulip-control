""" An aircraft electric system specification generator presented in the HSCC 2013 paper.

Huan Xu (mumu@caltech.edu)
October 30, 2012
"""
import sys, os
import copy
import numpy as np
import scipy
import scipy.io
import networkx as nx
import itertools
import time


def write_envgen(genlist):
    """Declares generator environment variable

    Parameters
    ----------
    genlist : list of all generators

    """
    for i in genlist:
        f.write(f"env_vars['g{i}'] = [0,1]")
        f.write('\n')


def write_envru(rulist):
    """Declares rectifier unit environment variable

    Parameters
    ----------
    rulist : list of all rectifier units

    """
    for i in rulist:
        f.write(f"env_vars[ru{i}'] = [0,1]")
        f.write('\n')


def write_essbusspec(essbuslist,time):
    """Writes counter for essential buses

    Parameters
    ----------
    essbuslist : list of buses with essential loads

    time: int
       max time bus can be unpowered

    """
    for i in essbuslist:
        f.write(f"guarantees += '&\\n\\t[]((b{i}=0) -> (next(countb{i}) = countb{i}+1))'")
        f.write('\n')
        f.write(f"guarantees += '&\\n\\t[]((b{i}=1) -> (next(countb{i}) = 0))'")
        f.write('\n')
        f.write(f"guarantees += '&\\n\\t[](countb{i} <= {time})'")
        f.write('\n')


def write_essbusdisc(essbuslist,time):
    """Declares bus counter system variable

    Parameters
    ----------
    essbuslist : list of all essential buses

    time: int
       max time bus can be unpowered

    """
    for i in essbuslist:
        f.write(f"disc_sys_vars['countb{i}'] = [x for x in range(0,{time + 1})]")
        f.write('\n')


def write_discbus(buslist):
    """Declares bus system variables

    Parameters
    ----------
    buslist : list of all buses

    """
    for i in buslist:
        f.write(f"disc_sys_vars['b{i}'] = [0,1]")
        f.write('\n')


def write_discnull(nullist):
    """Declares null node system variables

    Parameters
    ----------
    nullist : list of all null nodes

    """
    for i in nullist:
        f.write(f"disc_sys_vars['b{i}'] = [1]")
        f.write('\n')


def write_discdc_con(G,rulist,dcbuslist,nullist):
    """Declares contactors (removes contactors between rus and dcbuses)

    Parameters
    ----------
    G : networkX graph

    rulist: list of rectifier units

    dcbuslist: list of all dc buses

    """
    remove = list()
    remove2 = list()
    for i in rulist:
        for j in dcbuslist:
            remove.append((i,j))
    L = copy.deepcopy(G)
    L.remove_edges_from(remove)
    remove2 = all_pairs(nullist)
    L.remove_edges_from(remove2)
    edges = L.edges()
    # print(edges)
    for i in range(0,len(edges)):
        # print(edges[i][0])
        # print(edges[i][1])
        f.write(f"disc_sys_vars['c{edges[i][0]}{edges[i][1]}'] = [0,1]\n")
    for j in range(0,len(remove2)):
        f.write(f"disc_sys_vars['c{remove2[j][0]}{remove2[j][1]}'] = [1]\n")


def write_discac_con(G,nullist):
    """Declares contactors (removes contactors between rus and dcbuses)

    Parameters
    ----------
    G : networkX graph

    rulist: list of rectifier units

    dcbuslist: list of all dc buses

    """
    remove = list()
    H = copy.deepcopy(G)
    if len(nullist) >= 2:
        remove = all_pairs(nullist)
        H.remove_edges_from(remove)
    edges = H.edges()
    for i in range(0,len(edges)):
        if edges[i][0] < edges[i][1]:
            f.write(f"disc_sys_vars['c{edges[i][0]}{edges[i][1]}'] = [0,1]\n")
        else:
            f.write(f"disc_sys_vars['c{edges[i][1]}{edges[i][0]}'] = [0,1]\n")


def g_disconnect(G,genlist,buslist):
    """Writes specification disconnecting contactor if generator is unhealthy

    Parameters
    ----------
    G : networkX graph

    genlist : list of all generators

    buslist: list of all ac buses

    """
    for i in genlist:
        for j in buslist:
            for e in G.edges():
                if i in e and j in e:
                    if i < j:
                        f.write(f"guarantees += '&\\n\\t[]((g{i}=0) -> (c{i}{j}=0))'")
                    else:
                        f.write(f"guarantees += '&\\n\\t[]((g{i}=0) -> (c{j}{i}=0))'")
                    f.write('\n')


def ru_disconnect(G,rulist,buslist):
    """Writes specification disconnecting contactor if rectifier is unhealthy

    Parameters
    ----------
    G : networkX graph

    rulist : list of all rectifiers

    buslist: list of all ac buses

    """
    for i in rulist:
        for j in buslist:
            for e in G.edges():
                if i in e and j in e:
                    if i < j:
                        f.write(f"guarantees += '&\\n\\t[]((ru{i}=0) -> (c{i}{j}=0))'")
                    else:
                        f.write(f"guarantees += '&\\n\\t[]((ru{i}=0) -> (c{j}{i}=0))'")
                    f.write('\n')


def all_pairs(gens):
    """Returns list of all generator pairs (potential parallels).

    Parameters
    ----------
    gens : list
       list of all generator nodes

    """
    items = itertools.combinations(gens, 2)
    return list(_all_pairs(items))


def _all_pairs(items):
    memo = set()
    itr = itertools.filterfalse(
        memo.__contains__, items)
    for item in itr:
        memo.add(item)
        yield item


def all_gens(pairs, G):
    """Finds all generator pairs that are connected through graph.

    Parameters
    ----------
    G : NetworkX graph

    pairs : tuples
       list of all generator pairs

    """
    has_path = lambda p: nx.has_path(G, p[0], p[1])
    return list(filter(has_path, pairs))


def ppaths(i,j,G):
    """Finds all contactors between two parallel sources. Converts to LTL.

    Parameters
    ----------
    G : NetworkX graph

    source i : node
       Starting node for path

    target j : node
       Ending node for path
    """
    result = nx.shortest_path(G,source=i,target=j)
    C = dict()
    guarantees = 'guarantees += '"'"'&\\n\\t[](!('
    for k in range(0,len(result)-1):
        if result[k] < result[k+1]:
            C[k] = f'c{result[k]}{result[k + 1]}'
        else:
            C[k] = f'c{result[k + 1]}{result[k]}'
    guarantees = guarantees + f'({C[0]}=1)'
    for m in range(1,len(C)):
        guarantees = guarantees+ f' & ({C[m]}=1)'
    guarantees = guarantees + '))'"'"
    return guarantees


def noparallel(G, pairs):
    """Writes non-paralleling specifications to file.

    Parameters
    ----------
    G : NetworkX graph

    pairs : tuples
       list of all generator pairs with paths in G

    """
    sourcetemp = (all_pairs(pairs))
    source = all_gens(sourcetemp,G)
    for i in range(0,len(source)):
        mat = ppaths(source[i][0], source[i][1], G)
        f.write(mat)
        f.write('\n')


def faulttol(prob,allgens,genfail):
    """Finds all combinations of failures.

    Parameters
    ----------
    prob: int
       probability up to which failure can occur

    allgens: list
       list of all components that can fail

    genfail: int
       probability of failure of single component

    """
    tuples = int(prob / genfail)
    fails = list()
    temp = list()
    if tuples <= 1:
        fails = allgens[:]
    else:
        fails = allgens[:]
        for i in range(2,tuples+1):
            for temp in itertools.combinations(allgens,i):
                fails.append(temp)
    return fails


def write_genassump(genfail,genlist):
    """Writes generator environment assumption

    Parameters
    ----------
    genfail : int
       how many generators may fail at once time

    genlist : list of all generators

    """
    f.write(f"assumptions += '&\\n\\t[]((g{genlist[0]}")
    for i in range(1,len(genlist)):
        f.write(f' + g{genlist[i]}')
    f.write(f") >= {len(genlist)-genfail})'")
    f.write('\n')


def write_ruassump(rufail,rulist):
    """Writes rectifier environment assumption

    Parameters
    ----------
    rufail : int
       how many rectifiers may fail at once time

    genlist : list of all generators

    """
    if len(rulist) > 0:
        f.write(f"assumptions += '&\\n\\t[]((ru{rulist[0]}")
        for i in range(1,len(rulist)):
            f.write(f' + ru{rulist[i]}')
        f.write(f') >= {len(rulist) - rufail)})'")
        f.write('\n')


def remove_ru_edges(G,buslist,rulist):
    pairs = list()
    H = copy.deepcopy(G)
    if len(buslist) > 0:
        for i in busac:
            for j in rus:
                pairs.append((i,j))
    return pairs


def remove_rus(G,buslist,rulist):
    pairs = list()
    H = copy.deepcopy(G)
    if len(buslist) > 0:
        for i in busac:
            for j in rus:
                pairs.append((i,j))
        H.remove_edges_from(pairs)
    return H


def buspathnodes(G,busno,source):
    """Finds determines if path exists from bus to generator.

    Parameters
    ----------
    busno: int
       node number for bus

    sources: list
       list of all generators

    G: NetworkX graph

    """
    buspaths = list()
    if nx.has_path(G,busno,source):
            buspaths.append((busno,source))
    return buspaths


def acbusprop(G,source,target):
    paths = list()
    C = list()
    temp = list()
    edges = list()
    D = copy.deepcopy(G)
    gens2 = copy.deepcopy(gens)
    gens2.remove(target)
    for m in gens2:
        temp.append((source,m))
    D.remove_edges_from(temp)
    edges = remove_ru_edges(G,busac,rus)
    D.remove_edges_from(edges)
    for path in nx.all_simple_paths(D,source,target):
        paths.append(path)
        for p in range(0,len(paths)):
            for i in range(0,len(paths[p])-1):
                C.append((paths[p][i],paths[p][i+1]))
            f.write(f"disc_props['B{source}{target}{p}'] = '")
            if paths[p][1] in gens:
                f.write(f'(g{paths[p][1]}=1)')
            elif paths[p][1] in busac:
                f.write(f'(b{paths[p][1]}=1)')
            elif paths[p][1] in null:
                f.write(f'(b{paths[p][1]}=1)')
            else:
                pass
            if len(paths[p]) > 2:
                for j in range(2,len(paths[p])):
                    if paths[p][j] in gens:
                        f.write(f' & (g{paths[p][j]}=1)')
                    elif paths[p][j] in busac:
                        f.write(f' & (b{paths[p][j]}=1)')
                    elif paths[p][j] in null:
                        f.write(f' & (b{paths[p][j]}=1)')
                    else:
                        pass
            for k in range(0,len(C)):
                if C[k][0] < C[k][1]:
                    f.write(f' & (c{C[k][0]}{C[k][1]}=1)')
                else:
                    f.write(f' & (c{C[k][1]}{C[k][0]}=1)')
            f.write("'"'\n')
            C = list()


def write_acbusprop(G,buslist,genlist):
    """Writes dc bus properties

    Parameters
    ----------
    G : networkX graph

    buslist : list of all dc buses

    genlist : list of all generators

    """
    for i in buslist:
        for j in genlist:
            acbusprop(G,i,j)


def acbusspec(G,busno,gen):
    temp = list()
    edges = list()
    D = copy.deepcopy(G)
    gens2 = copy.deepcopy(gens)
    gens2.remove(gen)
    for m in gens2:
        temp.append((busno,m))
    D.remove_edges_from(temp)
    edges = remove_ru_edges(G,busac,rus)
    D.remove_edges_from(edges)
    paths = list()
    for path in nx.all_simple_paths(D,busno,gen,cutoff=None):
        paths.append(path)
    for j in range(0,len(paths)):
        f.write(f"guarantees += '&\\n\\t[]((B{busno}{gen}{j}) -> (b{busno}=1))'\n")


def write_acbusspec(G,buslist,genlist):
    """Writes dc bus specifications

    Parameters
    ----------
    G : networkX graph

    buslist : list of all dc buses

    genlist : list of all generators

    """
    for i in buslist:
        for j in genlist:
            acbusspec(G,i,j)


def write_acbusspec2(G,buslist,genlist):
    """Writes specifications for dc bus unpowered conditions

    Parameters
    ----------
    G : networkX graph

    buslist : list of all dc buses

    genlist : list of all generators

    """
    paths = list()
    temp = list()
    edges = list()
    D = copy.deepcopy(G)
    gens2 = copy.deepcopy(gens)
    edges = remove_ru_edges(G,busac,rus)
    D.remove_edges_from(edges)
    for i in buslist:
        f.write('guarantees += '"'"'&\\n\\t[](!((0=1)')
        for j in genlist:
            gens2.remove(j)
            D.remove_nodes_from(gens2)
            for path in nx.all_simple_paths(D,i,j):
                paths.append(path)
                f.write(f' | (B{i}{j}{len(paths) - 1})')
            paths = list()
            gens2 = copy.deepcopy(gens)
            D = copy.deepcopy(G)
        f.write(f") -> (b{i}=0))'\n")


def write_dcbusalways(buslist):
    """Writes dc bus specification must always be powered

    Parameters
    ----------
    buslist : list of all dc buses

    """
    for i in buslist:
        f.write(f"guarantees += '&\\n\\t[](b{i} = 1)'\n")


def dcbusprop(G,source,target):
    """Creates discrete properties for power status of dc buses

    Parameters
    ----------
    G : networkX graph

    source : node
       dc bus

    target : node
       generator

    """
    temp = list()
    edges = list()
    D = copy.deepcopy(G)
    gens2 = copy.deepcopy(gens)
    gens2.remove(target)
    D.remove_nodes_from(gens2)
    paths = list()
    C = list()
    for path in nx.all_simple_paths(D,source,target,cutoff=None):
        paths.append(path)
    for p in range(0,len(paths)):
        for i in range(0,len(paths[p])-1):
            if paths[p][i] in busdc and paths[p][i+1] in rus:
                pass
            elif paths[p][i] in rus and paths[p][i+1] in busdc:
                pass
            else:
                C.append((paths[p][i],paths[p][i+1]))
        f.write(f"disc_props['B{source}{target}{p}'] = '")
        if paths[p][1] in gens:
            f.write(f'(g{paths[p][1]}=1)')
        elif paths[p][1] in busac:
            f.write(f'(b{paths[p][1]}=1)')
        elif paths[p][1] in rus:
            f.write(f'(ru{paths[p][1]}=1)')
        elif paths[p][1] in busdc:
            f.write(f'(b{paths[p][1]}=1)')
        elif paths[p][1] in null:
            f.write(f'(b{paths[p][1]}=1)')
        else:
            pass
        for j in range(2,len(paths[p])):
            if paths[p][j] in gens:
                f.write(f' & (g{paths[p][j]}=1)')
            elif paths[p][j] in busac:
                f.write(f' & (b{paths[p][j]}=1)')
            elif paths[p][j] in rus:
                f.write(f' & (ru{paths[p][j]}=1)')
            elif paths[p][1] in busdc:
                f.write(f' & (b{paths[p][1]}=1)')
            elif paths[p][1] in null:
                f.write(f' & (b{paths[p][1]}=1)')
            else:
                pass
        for k in range(0,len(C)):
            if C[k][0] < C[k][1]:
                f.write(f' & (c{C[k][0]}{C[k][1]}=1)')
            else:
                f.write(f' & (c{C[k][1]}{C[k][0]}=1)')
        f.write("'"'\n')
        C = list()


def write_dcbusprop(G,buslist,genlist):
    """Writes dc bus properties

    Parameters
    ----------
    G : networkX graph

    buslist : list of all dc buses

    genlist : list of all generators

    """
    for i in buslist:
        for j in genlist:
            dcbusprop(G,i,j)


def dcbusspec(G,busno,gen):
    """Creates specifications for when DC bus gets powered

    Parameters
    ----------
    G : networkX graph

    busno : node
       dc bus

    gen : node
       generator

    """
    paths = list()
    C = list()
    temp = list()
    edges = list()
    D = copy.deepcopy(G)
    gens2 = copy.deepcopy(gens)
    gens2.remove(gen)
    D.remove_nodes_from(gens2)
    for path in nx.all_simple_paths(D,busno,gen,cutoff=None):
        paths.append(path)
    for j in range(0,len(paths)):
        f.write(f"guarantees += '&\\n\\t[]((B{busno}{gen}{j}) -> (b{busno}=1))'\n")


def write_dcbusspec(G,buslist,genlist):
    """Writes dc bus specifications

    Parameters
    ----------
    G : networkX graph

    buslist : list of all dc buses

    genlist : list of all generators

    """
    paths = list()
    for i in buslist:
        for j in genlist:
            dcbusspec(G,i,j)


def write_dcbusspec2(G,buslist,genlist):
    """Writes specifications for dc bus unpowered conditions

    Parameters
    ----------
    G : networkX graph

    buslist : list of all dc buses

    genlist : list of all generators

    """
    paths = list()
    temp = list()
    edges = list()
    D = copy.deepcopy(G)
    gens2 = copy.deepcopy(gens)
    for i in buslist:
        f.write('guarantees += '"'"'&\\n\\t[](!((0=1)')
        for j in genlist:
            gens2.remove(j)
            D.remove_nodes_from(gens2)
            for path in nx.all_simple_paths(D,i,j):
                paths.append(path)
                f.write(f' | (B{i}{j{len(paths) - 1})')
            paths = list()
            gens2 = copy.deepcopy(gens)
            D = copy.deepcopy(G)
        f.write(f") -> (b{i}=0))'\n")


def write_sat_bool(complist):
    """Defines boolean components (not including contactors)

    Parameters
    ----------
    complist : list of all components

    """
    for i in complist:
        if i in gens:
            f.write(f'(define g{i}::bool)\n')
        elif i in busac:
            f.write(f'(define b{i}::bool)\n')
        elif i in busdc:
            f.write(f'(define b{i}::bool)\n')
        elif i in null:
            f.write(f'(define b{i}::bool)\n')
        elif i in rus:
            f.write(f'(define r{i}::bool)\n')


def write_sat_con(G,rulist,dcbuslist,nullist):
    """Defines contactors (removes contactors between rus and dcbuses)

    Parameters
    ----------
    G : networkX graph

    rulist: list of rectifier units

    dcbuslist: list of all dc buses

    """
    remove = list()
    remove2 = list()
    for i in rulist:
        for j in dcbuslist:
            remove.append((i,j))
    L = copy.deepcopy(G)
    L.remove_edges_from(remove)
    remove2 = all_pairs(nullist)
    L.remove_edges_from(remove2)
    edges = L.edges()
    for i in range(0,len(edges)):
        f.write(f'(define c{edges[i][0]}{edges[i][1]}::bool)\n')


def write_sat_always(complist):
    """Asserts boolean components always on/powered

    Parameters
    ----------
    complist : list of all components

    """
    remove2 = list()
    for i in complist:
        if i in busac:
            f.write(f'(assert (= b{i} true))\n')
        elif i in busdc:
            f.write(f'(assert (= b{i} true))\n')
        elif i in null:
            f.write(f'(assert (= b{i} true))\n')
    remove2 = all_pairs(null)
    edges = G.edges()
    for j in range(0,len(remove2)):
        if remove2[j] in edges:
            f.write(f'(assert (= c{remove2[j][0]}{remove2[j][1]} true))\n')


def write_sat_disconnectgen(G,genlist):
    """Writes specification disconnecting contactor if generator is unhealthy

    Parameters
    ----------
    G : networkX graph

    genlist : list of all generators
    """
    neighbor = list()
    for i in genlist:
        neighbor = G.neighbors(i)
        for j in range(0, len(neighbor)):
            if i < neighbor[j]:
                f.write(f'(assert (=> (= g{i} false) (= c{i}{neighbor[j]} false)))\n')
            elif i > neighbor[j]:
                f.write(f'(assert (=> (= g{i} false) (= c{neighbor[j]}{i} false)))\n')
        neighbor = list()


def write_sat_disconnectru(G,rulist):
    """Writes specification disconnecting contactor if rectifier is unhealthy

    Parameters
    ----------
    G : networkX graph

    genlist : list of all generators
    """
    neighbor = list()
    H = copy.deepcopy(G)
    H.remove_nodes_from(busdc)
    for i in rulist:
        neighbor = H.neighbors(i)
        for j in range(0, len(neighbor)):
            if i < neighbor[j]:
                f.write(f'(assert (=> (= r{i} false) (= c{i}{neighbor[j]} false)))\n')
            elif i > neighbor[j]:
                f.write(f'(assert (=> (= r{i} false) (= c{neighbor[j]}{i} false)))\n')
        neighbor = list()


def write_sat_noparallel(G,genlist):
    pairs = all_pairs(genlist)
    H = copy.deepcopy(G)
    H.remove_nodes_from(rus)
    for i in range(0, len(pairs)):
        for path in nx.all_simple_paths(H, pairs[i][0], pairs[i][1]):
            f.write('(assert (not (and ')
            for j in range(0,len(path)-1):
                if path[j] < path[j+1]:
                    f.write(f'(= c{path[j]}{path[j + 1]} true) ')
                else:
                    f.write(f'(= c{path[j + 1]}{path[j]} true) ')
            f.write(')))\n')


def write_sat_acbusprop1(G,buslist, genlist):
    H = copy.deepcopy(G)
    H.remove_nodes_from(rus)
    for i in buslist:
        for j in genlist:
            gen_temp = copy.deepcopy(gens)
            gen_temp.remove(j)
            H.remove_nodes_from(gen_temp)
            H.remove_nodes_from(rus)
            for path in nx.all_simple_paths(H,i,j):
                f.write('(assert (=> (and ')
                for k in range(1,len(path)):
                    if path[k] in gens:
                        f.write(f' (= g{path[k]} true)')
                    elif path[k] in busac:
                        f.write(f' (= b{path[k]} true)')
                    elif path[k] in null:
                        f.write(f' (= b{path[k]} true)')
                for m in range(0,len(path)-1):
                    if path[m] < path[m+1]:
                        f.write(f' (= c{path[m]}{path[m + 1]} true)')
                    else:
                        f.write(f' (= c{path[m + 1]}{path[m]} true)')
                f.write(f') (= b{i} true)))\n')
            H = copy.deepcopy(G)
            H.remove_nodes_from(rus)


def write_sat_acbusprop2(G,buslist, genlist):
    H = copy.deepcopy(G)
    H.remove_nodes_from(rus)
    # i = buslist[7]
    # j = genlist[5]
    for i in buslist:
        f.write('(assert (=> (not (or ')
        for j in genlist:
            gen_temp = copy.deepcopy(gens)
            gen_temp.remove(j)
            H.remove_nodes_from(gen_temp)
            H.remove_nodes_from(rus)
            for path in nx.all_simple_paths(H,i,j):
                f.write(' (and')
                for k in range(1,len(path)):
                        if path[k] in gens:
                            f.write(f' (= g{path[k]} true)')
                        elif path[k] in busac:
                            f.write(f' (= b{path[k]} true)')
                        elif path[k] in null:
                            f.write(f' (= b{path[k]} true)')
                for m in range(0,len(path)-1):
                    if path[m] < path[m+1]:
                        f.write(f' (= c{path[m]}{path[m + 1]} true)')
                    else:
                        f.write(f' (= c{path[m + 1]}{path[m]} true)')
                f.write(')')
            H = copy.deepcopy(G)
            H.remove_nodes_from(rus)
        f.write(f')) (= b{i} false)))\n')


def write_sat_dcbusprop1(G,buslist, genlist):
    H = copy.deepcopy(G)
    for i in buslist:
        for j in genlist:
            gen_temp = copy.deepcopy(gens)
            gen_temp.remove(j)
            H.remove_nodes_from(gen_temp)
            for path in nx.all_simple_paths(H,i,j):
                f.write('(assert (=> (and ')
                for k in range(1,len(path)):
                    if path[k] in gens:
                        f.write(f' (= g{path[k]} true)')
                    elif path[k] in busac:
                        f.write(f' (= b{path[k]} true)')
                    elif path[k] in null:
                        f.write(f' (= b{path[k]} true)')
                    elif path[k] in rus:
                        f.write(f' (= r{path[k]} true)')
                for m in range(0,len(path)-1):
                    if path[m] in busdc and path[m+1] in rus:
                        pass
                    elif path[m] in rus and path[m+1] in busdc:
                        pass
                    elif path[m] < path[m+1]:
                        f.write(f' (= c{path[m]}{path[m + 1]} true)')
                    elif path[m+1] < path[m]:
                        f.write(f' (= c{path[m + 1]}{path[m]} true)')
                f.write(f') (= b{i} true)))\n')
            H = copy.deepcopy(G)


def write_sat_dcbusprop2(G,buslist, genlist):
    H = copy.deepcopy(G)
    for i in buslist:
        f.write('(assert (=> (not (or ')
        for j in genlist:
            gen_temp = copy.deepcopy(gens)
            gen_temp.remove(j)
            H.remove_nodes_from(gen_temp)
            for path in nx.all_simple_paths(H,i,j):
                f.write(' (and')
                for k in range(1,len(path)):
                    if path[k] in gens:
                        f.write(f' (= g{path[k]} true)')
                    elif path[k] in busac:
                        f.write(f' (= b{path[k]} true)')
                    elif path[k] in null:
                        f.write(f' (= b{path[k]} true)')
                    elif path[k] in rus:
                        f.write(f' (= r{path[k]} true)')
                for m in range(0,len(path)-1):
                    if path[m] in busdc and path[m+1] in rus:
                        pass
                    elif path[m] in rus and path[m+1] in busdc:
                        pass
                    elif path[m] < path[m+1]:
                        f.write(f' (= c{path[m]}{path[m + 1]} true)')
                    elif path[m+1] < path[m]:
                        f.write(f' (= c{path[m + 1]}{path[m]} true)')
                f.write(')')
            H = copy.deepcopy(G)
        f.write(f')) (= b{i} false)))\n')


def write_sat_env(gfail,rfail):
    gentemp = [x for x in range(0,gfail+1)]
    rutemp = [x for x in range(0,rfail+1)]
    allgens = list()
    allrus = list()
    env_filename = 'env'
    count = 0
    for i in gentemp:
        for j in itertools.combinations(gens,len(gens)-i):
            if len(gens)-i == 1:
                for k in gens:
                    allgens.append(k)
            else:
                allgens.append(j)
    allgens = list(set(allgens))
    if len(rus) == 0:
        pass
    else:
        for i in rutemp:
            for j in itertools.combinations(rus,len(rus)-i):
                if len(rus)-i == 1:
                    for k in rus:
                        allrus.append(k)
                else:
                    allrus.append(j)
    allrus = list(set(allrus))
    for i in range(0,len(allgens)):
        for j in range(0,len(allrus)):
            env_filename = f'env{count}.ys'
            f2 = open(env_filename,"w")
            for k in gens:
                if isinstance(allgens[i],int):
                    if k == allgens[i]:
                        f2.write(f'(assert (= g{k} true))\n')
                    else:
                        f2.write(f'(assert (= g{k} false))\n')
                elif k in allgens[i]:
                    f2.write(f'(assert (= g{k} true))\n')
                else:
                    f2.write(f'(assert (= g{k} false))\n')
            for m in rus:
                if isinstance(allrus[j],int):
                    if m == allrus[j]:
                        f2.write(f'(assert (= r{m} true))\n')
                    else:
                        f2.write(f'(assert (= r{m} false))\n')
                elif m in allrus[j]:
                    f2.write(f'(assert (= r{m} true))\n')
                else:
                    f2.write(f'(assert (= r{m} false))\n')
            count = count+1
            f2.write('(check)\n')
            f2.close()


start = time.time()
file_name = 'test_spec'
#Load adjacency matrix from matfile
data = scipy.io.loadmat('SLD.mat')
datatemp = data['A']
#Failure Probabilities
genfail = 1
rufail = 1
busfail = 0
#Node definitions
busac = [2,3]
busess = [2,3]
busdc = [6,7]
null = list()
rus = [4,5]
gens = [0,1]
#Bus time
nptime = 0
#Create networkx graph from adjacency matrix
G = nx.from_numpy_array(datatemp)
print(f'number of edges {len(G.edges())}')
print(f'number of nodes {len(G.nodes())}')
#sets of failure states
# fails = faulttol(10,gens,genfail)


# Synthesize
if ('tulip' in sys.argv):
    file_name = file_name+'.py'
    f = open(file_name, "w")
    # #environment variables
    print('writing environment variables')
    write_envgen(gens)
    write_envru(rus)
    #discrete system variables
    print('writing discrete system variables: buses')
    write_discbus(busac)
    write_discbus(busdc)
    write_discnull(null)
    print(time.time()-start)
    print('writing discrete system variables: contactors')
    # write_discac_con(G,null) #only use this when there are NO DC components
    write_discdc_con(G,rus,busdc,null)
    print(time.time()-start)
    print('writing discrete system variables: bus counters')
    write_essbusdisc(busess,nptime)
    #acbus discrete properties
    print('removing ru paths')
    H = remove_rus(G,busac,rus)
    print(time.time()-start)
    print('writing discrete bus properties: AC')
    write_acbusprop(H,busac,gens)
    print(time.time()-start)
    print('writing discrete bus properties: DC')
    write_dcbusprop(G,busdc,gens)
    print(time.time()-start)
    #Environment assumptions
    print('writing environment assumptions')
    write_genassump(genfail,gens)
    write_ruassump(rufail,rus)
    print(time.time()-start)
    #acbus power guarantees
    print('writing bus power specifications: AC')
    write_acbusspec(H,busac,gens)
    write_acbusspec2(H,busac,gens)
    write_essbusspec(busess,nptime)
    print(time.time()-start)
    #disconnect unhealthy guarantees
    print('disconnecting unhealthy generators')
    g_disconnect(G,gens,busac)
    g_disconnect(G,gens,null)
    print(time.time()-start)
    print('disconnecting unhealthy rus')
    ru_disconnect(G,rus,busac)
    ru_disconnect(G,rus,null)
    print(time.time()-start)
    ##non-parallel guarantees
    print('writing no paralleling specs')
    noparallel(G,gens)
    print(time.time()-start)
    #dc bus power specifications
    print('writing bus power specifications: DC')
    write_dcbusspec(G,busdc,gens)
    write_dcbusspec2(G,busdc,gens)
    write_dcbusalways(busdc)
    f.close()
    print('It took', time.time()-start, 'seconds.')


# Synthesize
if ('yices' in sys.argv):
    file_name = file_name+'.ys'
    f = open(file_name, "w")
    #Writing component definitions
    write_sat_bool(gens)
    write_sat_bool(busac)
    write_sat_bool(busdc)
    write_sat_bool(null)
    write_sat_bool(rus)
    #Writing contactor definitions
    write_sat_con(G,rus,busdc,null)
    #Writing always power bus and null assertions
    write_sat_always(busac)
    write_sat_always(busdc)
    write_sat_always(null)
    #Writing disconnect implications
    write_sat_disconnectgen(G,gens)
    write_sat_disconnectru(G,rus)
    write_sat_noparallel(G,gens)
    write_sat_acbusprop1(G,busac,gens)
    write_sat_acbusprop2(G,busac,gens)
    write_sat_dcbusprop1(G,busdc, gens)
    write_sat_dcbusprop2(G,busdc, gens)
    #write_environment assumptions
    write_sat_env(genfail,rufail)
    f.close()
    print('It took', time.time()-start, 'seconds.')
