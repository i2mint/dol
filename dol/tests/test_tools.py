"""Test the tools module."""


from dol.tools import filt_iter


def test_filt_iter():
    # Demo regex filter on a class
    contains_a = filt_iter.regex(r'a')
    # wrap the dict type with this
    filtered_dict = contains_a(dict)
    # now make a filtered_dict
    d = filtered_dict(apple=1, banana=2, cherry=3)
    # and see that keys not containing "a" are filtered out
    assert dict(d) == {'apple': 1, 'banana': 2}

    # With this regex filt_iter, we made two specialized versions:
    # One filtering prefixes, and one filtering suffixes
    is_test = filt_iter.prefixes('test')  # Note, you can also pass a list of prefixes
    d = {'test.txt': 1, 'report.doc': 2, 'test_image.jpg': 3}
    dd = is_test(d)
    assert dict(dd) == {'test.txt': 1, 'test_image.jpg': 3}

    is_text = filt_iter.suffixes(['.txt', '.doc', '.pdf'])
    d = {'test.txt': 1, 'report.doc': 2, 'image.jpg': 3}
    dd = is_text(d)
    assert dict(dd) == {'test.txt': 1, 'report.doc': 2}
