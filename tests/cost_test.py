# This test includes unit tests for the tulip.transys.cost module.
#
# * vectorcost_test(): a simple unit test for tulip.transys.cost module,
#   ensuring that the addition, multiplication and comparison work correctly.
# * vectorcost_zero_test(): a unit test for tulip.transys.cost module,
#   focusing on the case where one cost is zero.
# * inf_test(): a unit test for tulip.transys.cost module,
#   focusing on the case where one cost is infinite.


from tulip.transys.cost import VectorCost


def vectorcost_test():
    num_item = 10
    a = VectorCost([2 * i for i in range(num_item)])
    assert len(a) == num_item
    for i in range(num_item):
        assert a[i] == 2 * i

    b = a + 1
    c = 1 + a
    assert b >= a
    assert b > a
    assert a < b
    assert a <= b
    assert b == c
    assert a != c
    assert len(b) == num_item
    assert len(c) == num_item

    i = 0
    for b_item in b:
        assert b_item == 2 * i + 1
        i += 1

    d = VectorCost([2 * i if i != 2 else 2 * i + 1 for i in range(num_item)])

    assert d > a

    e = b * a
    for i in range(len(e)):
        assert e[i] == 2 * i * (2 * i + 1)


def vectorcost_zero_test():
    num_item = 10
    a = VectorCost([2 * i for i in range(num_item)])

    assert a + 0 == a
    assert 0 + a == a
    assert a * 0 == 0
    assert 0 * a == 0


def inf_test():
    num_item = 10
    a = VectorCost([2 * i + 1 for i in range(num_item)])

    inf = float("inf")

    assert a + inf == inf
    assert inf + a == inf
    print(a * inf)
    assert a * inf == inf
    assert inf * a == inf
