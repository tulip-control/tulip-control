"""
McNaughton's algorithm for solving parity games

Reference
=========
Robert McNaughton (1993). Infinite games played on finite graphs.
Annals of Pure and Applied Logic 65(2): 149--184.
doi:10.1016/0168-0072(93)90036-D
"""
import copy
from tulip import transys as trs

def McNaughton(p):
    """Solve parity game.

    @type p: L{ParityGame}
    """
    p = copy.deepcopy(p)

    if p.has_deadends():
        raise Exception('The GameGraph has sinks !')

    c = p.max_color
    print('highest color occurring in P: c = ' + str(c))

    if c is 0 or len(p) == 0:
        return (set(p), set())

    sigma = c % 2
    W = {0:set(), 1:set()}
    while True:
        max_color_pos = {x for x in p if p.node[x]['color'] == c}

        a = attractor(max_color_pos, p, sigma)

        p_ = copy.deepcopy(p)
        p_.states.remove_from(a)

        W_ = McNaughton(p_)

        if not W_[1-sigma]:
            W[sigma] = set(p) - W[1-sigma]
            return (W[0], W[1])

        a = attractor(W_[1-sigma], p, 1-sigma)
        W[1-sigma] = W[1-sigma].union(a)

        p.states.remove_from(a)

def attractor(W, pg, player):
    """Find attractor set.

    @param w: winning region
    @type w: set

    @type pg: L{ParityGame}

    @param player: for whom to calculate attractor
    """
    W = set(W)
    Wold = set()

    # least fixed point
    while W != Wold:
        Wold = W
        W = W.union(ctrl_next(W, pg, player))
    return W

def ctrl_next(W, pg, player):
    """Find controlled predecessor set.

    These are the nodes that in one step:

      - can reach W because player controls them
      - will reach W because opponent controls them,
        but has no other choice than next(W)
    """
    cnext = set()
    for node in W:
        for pred in pg.predecessors_iter(node):
            if pg.node[pred]['player'] == player:
                print('controlled by player, good')
            elif len(pg.succ[pred]) == 1:
                print('controlled by opponent, bad only 1 outgoing')
            else:
                print('not in CNext')
                continue

            cnext.add(pred)
    return cnext

if __name__ == '__main__':
    p = trs.automata.ParityGame(c=3)

    p.states.add('p0', player=0, color=1)
    p.states.add('p1', player=1, color=0)
    p.states.add('p2', player=1, color=1)

    p.transitions.add('p0', 'p1')
    p.transitions.add('p1', 'p0')
    p.transitions.add('p0', 'p2')
    p.transitions.add('p2', 'p2')

    print(p)
    W0, W1 = McNaughton(p)

    print('player 0 wins from: W0 = ' + str(W0))
    print('player 1 wins from: W1 = ' + str(W1))
