import sys, itertools
import tulip.gridworld as gw
from tulip import solver

def gridworld_model(Z, goal_sequence=False, sp=None):
    if sp is None:
        sp = []
    
    initials = { Z[x] : True for x in Z.init_list }
    sp.extend([ "[]<>(%s)" % Z[x] for x in Z.goal_list ])
    discvars = {}
    
    if goal_sequence:
        # Goal sequencing
        sp.append("(goal = 0)")
        for (n,g) in enumerate(Z.goal_list):
            sp.append("[]((goal != %d) | <>(%s))" % (n, Z[g]))
            sp.append("[](goal = %d -> (next(goal = %d) | (%s & next(goal = %d))))"
                        % (n, n, Z[g], n+1))
            # Avoid this goal when looking for another
            sp.append("[]((goal != %d) -> !%s)" % (n, Z[g]))
        # Reset goal count
        sp.append("[](goal = %d -> next(goal = 0))" % len(Z.goal_list))
        # Progress for 'goal': eventually reach all goals in order
        sp.append("[]<>(goal = %d)" % len(Z.goal_list))
        discvars = {"goal" : "{0...%d}" % len(Z.goal_list)}
    
    pp = Z.discreteTransitionSystem()
    gwmodel = solver.discDynamicsModel(discvars, ["", " & ".join(sp)],
                {}, pp, initials)
    return gwmodel
    
def gridworld_problem(size=(5,5), wall_density=0.2, num_init=1, num_goals=1,
        num_robots=1, goal_sequence=False):
    Z = gw.random_world(size, wall_density, num_init, num_goals)
    model = gridworld_model(Z, goal_sequence)
    slvi = solver.SolverInput()
    slvi.addModule("grid", *model, instances=num_robots)
    return (slvi, Z)

def benchmark_variable(indep, vals, deps, solver, fixed={}):
    opts = dict(fixed)
    for v in vals:
        opts[indep] = v
        try:
            (slvi, Z) = gridworld_problem(**opts)
        except ValueError:
            continue
        slvi.setSolver(solver)
        slvi.write("gw_bm.mdl")
        rlz = slvi.solve("gw_bm.aut")
        if rlz:
            aut = slvi.automaton()
            print indep, v, slvi.solveTime(), len(aut)
        else:
            print indep, v, slvi.solveTime(), "UNRLZ"
        
if __name__ == "__main__":
    benchmark_variable("size", ((x,x) for x in range(3,5) for y in range(5)), [], "SPIN", { "num_robots" : 2, "wall_density" : 0.5 })
    benchmark_variable("wall_density", (x/10.0 for x in range(0,10)), [], "NuSMV", { "num_goals" : 3 })
