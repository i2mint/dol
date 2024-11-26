"""Test trans.py functionality."""

from dol.trans import filt_iter, redirect_getattr_to_getitem


def test_filt_iter():
    # Demo regex filter on a class
    contains_a = filt_iter.regex(r"a")
    # wrap the dict type with this
    filtered_dict = contains_a(dict)
    # now make a filtered_dict
    d = filtered_dict(apple=1, banana=2, cherry=3)
    # and see that keys not containing "a" are filtered out
    assert dict(d) == {"apple": 1, "banana": 2}

    # With this regex filt_iter, we made two specialized versions:
    # One filtering prefixes, and one filtering suffixes
    is_test = filt_iter.prefixes("test")  # Note, you can also pass a list of prefixes
    d = {"test.txt": 1, "report.doc": 2, "test_image.jpg": 3}
    dd = is_test(d)
    assert dict(dd) == {"test.txt": 1, "test_image.jpg": 3}

    is_text = filt_iter.suffixes([".txt", ".doc", ".pdf"])
    d = {"test.txt": 1, "report.doc": 2, "image.jpg": 3}
    dd = is_text(d)
    assert dict(dd) == {"test.txt": 1, "report.doc": 2}


def test_redirect_getattr_to_getitem():

    # Applying it to a class

    ## ... with the @decorator syntax
    @redirect_getattr_to_getitem
    class MyDict(dict):
        pass

    d1 = MyDict(a=1, b=2)
    assert d1.a == 1
    assert d1.b == 2
    assert list(d1) == ["a", "b"]

    ## ... as a decorator factory
    D = redirect_getattr_to_getitem()(dict)
    d2 = D(a=1, b=2)
    assert d2.a == 1
    assert d2.b == 2
    assert list(d2) == ["a", "b"]

    # Applying it to an instance

    ## ... as a decorator
    backend_d = dict(a=1, b=2)

    d3 = redirect_getattr_to_getitem(backend_d)
    assert d3.a == 1
    assert d3.b == 2
    assert list(d3) == ["a", "b"]

    ## ... as a decorator factory
    d4 = redirect_getattr_to_getitem()(backend_d)
    assert d4.a == 1
    assert d4.b == 2
    assert list(d4) == ["a", "b"]
