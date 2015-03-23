'''Records that can be nested, as in r.a.b

BROKEN

ref: stackover flow at accessing python dict with multiple key lookup string
'''

import unittest
import pdb

orig_dict = dict


class NestedRecord(orig_dict):
    def __init__(self, *args, **kwargs):
        pdb.set_trace()
        super(NestedRecord, self).__init__(*args, **kwargs)
        # the statement just below fails
        for k, v in self.iteritems():
            if type(v) == orig_dict and not isinstance(v, NestedRecord):
                super(NestedRecord, self).__setitem__(k, NestedRecord(v))

    def __getattribute__(self, k):
        try:
            return orig_dict.__getattribute__(k)
        except:
            return self.__getitem__(k)

    def __setattribute__(self, k, v):
        if k in self:
            self.__setitem__(k, v)
        else:
            return orig_dict.__setattr__(k, v)

    def __delattr__(self, k):
        try:
            self.__delitem__(k)
        except:
            orig_dict.__delattr__(k)

    def __setitem__(self, k, v):
        toconvert = type(v) == orig_dict and not isinstance(v, NestedRecord)
        orig_dict.__setitem__(k, NestedRecord(v) if toconvert else v)


class Test(unittest.TestCase):
    def setUp(self):
        self.a = NestedRecord(c=2)
        self.b = NestedRecord(a=1, b=NestedRecord(c=2, d=3))

    def test_b(self):
        b = self.b
        b.a = b.b
        b.b = 1
        print 'b', b


if __name__ == '__main__':
    unittest.main()
