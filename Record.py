import unittest


class Record(object):
    '''Record with fields, include _name'''
    def __init__(self, name):
        self._name = name

    def __str__(self):
        s = ''
        for kv in sorted(self.__dict__.items()):
            k, v = kv
            s = s + ('%s.%s = %s\n' % (self._name, k, v))
        return s


class Test(unittest.TestCase):
    def test_1(self):
        verbose = False

        class R(Record):
            def __init__(self, value):
                Record.__init__(self, 'r')
                self.a = 10

        r = R(10)
        self.assertEqual(r.a, 10)
        if verbose:
            print r

if __name__ == '__main__':
    unittest.main()
