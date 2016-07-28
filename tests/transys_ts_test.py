"""Tests for transys.transys (part of transys subpackage)"""
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logging.getLogger('tulip.transys.products').setLevel(logging.DEBUG)
from tulip import transys as trs
from tulip.transys.mathset import MathSet, PowerSet


def ts_test():
    ts = trs.FTS()

    ts.states.add('s0')
    assert('s0' in ts)
    assert('s0' in ts.node)
    assert('s0' in ts.states)

    states = {'s0', 's1', 's2', 's3'}
    ts.states.add_from(states)
    assert(set(ts.states) == states)

    ts.transitions.add('s0', 's1')
    ts.transitions.add_from([('s1', 's2'), ('s2', 's3'), ('s3', 's0')])

    ts.states.initial.add('s0')
    ts.states.initial.add_from({'s0', 's1'})

    ts.atomic_propositions.add('p')
    assert(set(ts.atomic_propositions) == {'p'})

    ts.states['s0']['ap'] = {'p'}
    ts.states['s1']['ap'] = set()
    ts.states['s2']['ap'] = set()
    ts.states['s3']['ap'] = set()
    assert(ts.states['s0']['ap'] == {'p'})

    for state in {'s1', 's2', 's3'}:
        assert(ts.states[state]['ap'] == set() )

    logger.debug(ts)
    return ts


def ba_test():
    ba = trs.BA()

    aps = ['p']
    ba.atomic_propositions |= {'p'}
    assert('p' in ba.atomic_propositions)
    assert(ba.atomic_propositions == MathSet(aps) )
    assert(ba.alphabet == PowerSet(aps) )


    ba.states.add_from({'q0', 'q1'})
    assert(set(ba.states) == {'q0', 'q1'})

    ba.states.initial.add('q0')
    assert(set(ba.states.initial) == {'q0'})

    ba.states.accepting.add('q1')
    assert(set(ba.states.accepting) == {'q1'})

    ba.transitions.add('q0', 'q1', letter={'p'})
    ba.transitions.add('q1', 'q1', letter={'p'})
    ba.transitions.add('q1', 'q0', letter=set() )
    ba.transitions.add('q0', 'q0', letter=set() )

    logger.debug(ba)
    ba.save('ba.pdf')
    return ba


def ba_ts_prod_test():
    ts = ts_test()
    ba = ba_test()
    ba_ts = trs.products.ba_ts_sync_prod(ba, ts)

    check_prodba(ba_ts)
    (ts_ba, persistent) = trs.products.ts_ba_sync_prod(ts, ba)


    states = {('s0', 'q1'), ('s1', 'q0'),
              ('s2', 'q0'), ('s3', 'q0')}
    assert(set(ts_ba.states) == states)
    assert(persistent == {('s0', 'q1')} )


    ba_ts.save('prod.pdf')
    return ba_ts


def check_prodba(ba_ts):
    states = {('s0', 'q1'), ('s1', 'q0'),
              ('s2', 'q0'), ('s3', 'q0')}
    assert(set(ba_ts.states) == states)

    assert(set(ba_ts.states.initial) == {('s0', 'q1'), ('s1', 'q0')})

    assert(

        ba_ts.transitions.find(
            [('s0', 'q1')], [('s1', 'q0')]
        )[0][2]['letter'] == set()
    )

    for si, sj in [('s1', 's2'), ('s2', 's3')]:
        assert(
            ba_ts.transitions.find(
                [(si, 'q0')], [(sj, 'q0')]
            )[0][2]['letter'] == set()
        )

    assert(
        ba_ts.transitions.find(
            [('s3', 'q0')], [('s0', 'q1')]
        )[0][2]['letter'] == {'p'}
    )

def on_the_fly_test():
    ba = ba_test()
    ts = ts_test()
    prodba = trs.OnTheFlyProductAutomaton(ba, ts)
    assert(set(prodba.states) == {('s0', 'q1'), ('s1', 'q0')})

    prodba.save('prodba_initialized.pdf')
    prodba.add_all_states()
    check_prodba(prodba)
    prodba.save('prodba_full.pdf')
