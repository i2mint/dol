# dol.misc

Functions to read from and write to misc sources, choosing the codec from the key.

`get_obj`/`set_obj` read and write a file with the codec picked from its extension
(`.json`, `.csv`, `.pkl`, …); `MiscReaderMixin`/`MiscStoreMixin` add the same
key-conditioned (de)serialization to any store.

```pycon
>>> from dol.misc import MiscStoreMixin
>>> class M(MiscStoreMixin, dict):
...     pass
>>> m = M()
>>> m['a.json'] = {'x': 1}
>>> dict.__getitem__(m, 'a.json'), m['a.json']
(b'{"x": 1}', {'x': 1})
```

### Functions

| `csv_fileobj`(csv_data, \*args, \*\*kwargs)                                  |                                                                                                                                 |
|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `dflt_dflt_incoming_val_trans`(x)                                            |                                                                                                                                 |
| `dflt_func_key`(self, k)                                                     |                                                                                                                                 |
| [`get_obj`](#dol.misc.get_obj)(k[, store, ...])    | A quick way to get an object, with default.                                                                                     |
| `identity_method`(x)                                                         |                                                                                                                                 |
| [`set_obj`](#dol.misc.set_obj)(k, v[, store, ...]) | A quick way to set an object, with defaults for everything (but the key and value, you know, a clue of what you want to store). |
| `url_to_bytes`(url)                                                          |                                                                                                                                 |

### Classes

| [`MiscGetter`](#dol.misc.MiscGetter)([store, ...])                          | An object to write (and only write) to a store (default local files) with automatic deserialization according to a property of the key (default: file extension).        |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`MiscGetterAndSetter`](#dol.misc.MiscGetterAndSetter)([store, ...])                 | An object to read and write (and nothing else) to a store (default local) with automatic (de)serialization according to a property of the key (default: file extension). |
| [`MiscReaderMixin`](#dol.misc.MiscReaderMixin)([...])                            | Mixin to transform incoming vals according to the key their under.                                                                                                       |
| [`MiscStoreMixin`](#dol.misc.MiscStoreMixin)([incoming_val_trans_for_key, ...]) | Mixin to transform incoming and outgoing vals according to the key their under.                                                                                          |

### *class* dol.misc.MiscGetter(store=Files(rootdir='', subpath='', pattern_for_field=None, max_levels=None, include_hidden=False, assert_rootdir_existence=False), incoming_val_trans_for_key={'.bin': <function identity_method>, '.csv': <function <lambda>>, '.gz': <function decompress>, '.gzip': <function decompress>, '.json': <function <lambda>>, '.pickle': <function <lambda>>, '.pkl': <function <lambda>>, '.txt': <function <lambda>>, '.zip': <class 'dol.zipfiledol.FilesOfZip'>}, dflt_incoming_val_trans=<function identity_method>, func_key=<function MiscGetter.<lambda>>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

An object to write (and only write) to a store (default local files) with automatic deserialization
according to a property of the key (default: file extension).

```pycon
>>> from dol.misc import get_obj, misc_objs_get
>>> import os
>>> import json
>>>
>>> pjoin = lambda *p: os.path.join(os.path.expanduser('~'), *p)
>>> path = pjoin('tmp.json')
>>> d = {'a': {'b': {'c': [1, 2, 3]}}}
>>> json.dump(d, open(path, 'w'))  # putting a json file there, the normal way, so we can use it later
>>>
>>> k = path
>>> t = get_obj(k)  # if you'd like to use a function
>>> assert t == d
>>> tt = misc_objs_get[k]  # if you'd like to use an object (note: can get, but nothing else (no list, set, del, etc))
>>> assert tt == d
>>> t
{'a': {'b': {'c': [1, 2, 3]}}}
```

### *class* dol.misc.MiscGetterAndSetter(store=Files(rootdir='', subpath='', pattern_for_field=None, max_levels=None, include_hidden=False, assert_rootdir_existence=False), incoming_val_trans_for_key={'.bin': <function identity_method>, '.csv': <function <lambda>>, '.gz': <function decompress>, '.gzip': <function decompress>, '.json': <function <lambda>>, '.pickle': <function <lambda>>, '.pkl': <function <lambda>>, '.txt': <function <lambda>>, '.zip': <class 'dol.zipfiledol.FilesOfZip'>}, outgoing_val_trans_for_key={'.bin': <function identity_method>, '.cnf': <function <lambda>>, '.conf': <function <lambda>>, '.config': <function <lambda>>, '.csv': <function csv_fileobj>, '.gz': <function compress>, '.gzip': <function compress>, '.ini': <function <lambda>>, '.json': <function <lambda>>, '.pickle': <function <lambda>>, '.pkl': <function <lambda>>, '.txt': <function <lambda>>}, dflt_incoming_val_trans=<function identity_method>, func_key=<function MiscGetterAndSetter.<lambda>>)

Bases: [`MiscGetter`](#dol.misc.MiscGetter)

An object to read and write (and nothing else) to a store (default local) with automatic (de)serialization
according to a property of the key (default: file extension).

```pycon
>>> from dol.misc import set_obj, misc_objs  # the function and the object
>>> import json
>>> import os
>>>
>>> pjoin = lambda *p: os.path.join(os.path.expanduser('~'), *p)
>>>
>>> d = {'a': {'b': {'c': [1, 2, 3]}}}
>>> misc_objs[pjoin('tmp.json')] = d
>>> filepath = os.path.expanduser('~/tmp.json')
>>> assert misc_objs[filepath] == d  # yep, it's there, and can be retrieved
>>> assert json.load(open(filepath)) == d  # in case you don't believe it's an actual json file
>>>
>>> # using pickle
>>> misc_objs[pjoin('tmp.pkl')] = d
>>> assert misc_objs[pjoin('tmp.pkl')] == d
>>>
>>> # using txt
>>> misc_objs[pjoin('tmp.txt')] = 'hello world!'
>>> assert misc_objs[pjoin('tmp.txt')] == 'hello world!'
>>>
>>> # using csv
>>> misc_objs[pjoin('tmp.csv')] = [[1,2,3], ['a','b','c']]
>>> assert misc_objs[pjoin('tmp.csv')] == [['1','2','3'], ['a','b','c']]  # yeah, well, not numbers, but you deal with it
>>>
>>> # using bin
... misc_objs[pjoin('tmp.bin')] = b'let us pretend these are bytes of an audio waveform'
>>> assert misc_objs[pjoin('tmp.bin')] == b'let us pretend these are bytes of an audio waveform'
```

### *class* dol.misc.MiscReaderMixin(incoming_val_trans_for_key=None, dflt_incoming_val_trans=None, func_key=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Mixin to transform incoming vals according to the key their under.

#### WARNING
If used as a subclass, this mixin should (in general) be placed before the store

```pycon
>>> # make a reader that will wrap a dict
>>> class MiscReader(MiscReaderMixin, dict):
...     def __init__(self, d,
...                         incoming_val_trans_for_key=None,
...                         dflt_incoming_val_trans=None,
...                         func_key=None):
...         dict.__init__(self, d)
...         MiscReaderMixin.__init__(self, incoming_val_trans_for_key, dflt_incoming_val_trans, func_key)
...
>>>
>>> incoming_val_trans_for_key = dict(
...     MiscReaderMixin._incoming_val_trans_for_key,  # take the existing defaults...
...     **{'.bin': lambda v: [ord(x) for x in v.decode()], # ... override how to handle the .bin extension
...      '.reverse_this': lambda v: v[::-1]  # add a new extension (and how to handle it)
...     })
>>>
>>> import pickle
>>> d = {
...     'a.bin': b'abc123',
...     'a.reverse_this': b'abc123',
...     'a.csv': b'event,year\n Magna Carta,1215\n Guido,1956',
...     'a.txt': b'this is not a text',
...     'a.pkl': pickle.dumps(['text', [str, map], {'a list': [1, 2, 3]}]),
...     'a.json': '{"str": "field", "int": 42, "float": 3.14, "array": [1, 2], "nested": {"a": 1, "b": 2}}',
... }
>>>
>>> s = MiscReader(d=d, incoming_val_trans_for_key=incoming_val_trans_for_key)
>>> list(s)
['a.bin', 'a.reverse_this', 'a.csv', 'a.txt', 'a.pkl', 'a.json']
>>> s['a.bin']
[97, 98, 99, 49, 50, 51]
>>> s['a.reverse_this']
b'321cba'
>>> s['a.csv']
[['event', 'year'], [' Magna Carta', '1215'], [' Guido', '1956']]
>>> s['a.pkl']
['text', [<class 'str'>, <class 'map'>], {'a list': [1, 2, 3]}]
>>> s['a.json']
{'str': 'field', 'int': 42, 'float': 3.14, 'array': [1, 2], 'nested': {'a': 1, 'b': 2}}
```

### *class* dol.misc.MiscStoreMixin(incoming_val_trans_for_key=None, outgoing_val_trans_for_key=None, dflt_incoming_val_trans=None, dflt_outgoing_val_trans=None, func_key=None)

Bases: [`MiscReaderMixin`](#dol.misc.MiscReaderMixin)

Mixin to transform incoming and outgoing vals according to the key their under.

#### WARNING
If used as a subclass, this mixin should (in general) be placed before the store

#### SEE ALSO
preset and postget args from wrap_kvs decorator from dol.trans.

```pycon
>>> # Make a class to wrap a dict with a layer that transforms written and read values
>>> class MiscStore(MiscStoreMixin, dict):
...     def __init__(self, d,
...                         incoming_val_trans_for_key=None, outgoing_val_trans_for_key=None,
...                         dflt_incoming_val_trans=None, dflt_outgoing_val_trans=None,
...                         func_key=None):
...         dict.__init__(self, d)
...         MiscStoreMixin.__init__(self, incoming_val_trans_for_key, outgoing_val_trans_for_key,
...                                 dflt_incoming_val_trans, dflt_outgoing_val_trans, func_key)
...
>>>
>>> outgoing_val_trans_for_key = dict(
...     MiscStoreMixin._outgoing_val_trans_for_key,  # take the existing defaults...
...     **{'.bin': lambda v: ''.join([chr(x) for x in v]).encode(), # ... override how to handle the .bin extension
...        '.reverse_this': lambda v: v[::-1]  # add a new extension (and how to handle it)
...     })
>>> ss = MiscStore(d={},  # store starts empty
...                incoming_val_trans_for_key={},  # overriding incoming trans so we can see the raw data later
...                outgoing_val_trans_for_key=outgoing_val_trans_for_key)
...
>>> # here's what we're going to write in the store
>>> data_to_write = {
...      'a.bin': [97, 98, 99, 49, 50, 51],
...      'a.reverse_this': b'321cba',
...      'a.csv': [['event', 'year'], [' Magna Carta', '1215'], [' Guido', '1956']],
...      'a.txt': 'this is not a text',
...      'a.pkl': ['text', [str, map], {'a list': [1, 2, 3]}],
...      'a.json': {'str': 'field', 'int': 42, 'float': 3.14, 'array': [1, 2], 'nested': {'a': 1, 'b': 2}}}
>>> # write this data in our store
>>> for k, v in data_to_write.items():
...     ss[k] = v
>>> list(ss)
['a.bin', 'a.reverse_this', 'a.csv', 'a.txt', 'a.pkl', 'a.json']
>>> # Looking at the contents (what was actually stored/written)
>>> for k, v in ss.items():
...     if k != 'a.pkl':
...         print(f"{k}: {v}")
...     else:  # need to verify pickle data differently, since printing contents is problematic in doctest
...         assert pickle.loads(v) == data_to_write['a.pkl']
a.bin: b'abc123'
a.reverse_this: b'abc123'
a.csv: b'event,year\r\n Magna Carta,1215\r\n Guido,1956\r\n'
a.txt: b'this is not a text'
a.json: b'{"str": "field", "int": 42, "float": 3.14, "array": [1, 2], "nested": {"a": 1, "b": 2}}'
```

### dol.misc.get_obj(k, store=Files(rootdir='', subpath='', pattern_for_field=None, max_levels=None, include_hidden=False, assert_rootdir_existence=False), incoming_val_trans_for_key={'.bin': <function identity_method>, '.csv': <function <lambda>>, '.gz': <function decompress>, '.gzip': <function decompress>, '.json': <function <lambda>>, '.pickle': <function <lambda>>, '.pkl': <function <lambda>>, '.txt': <function <lambda>>, '.zip': <class 'dol.zipfiledol.FilesOfZip'>}, dflt_incoming_val_trans=<function identity_method>, func_key=<function <lambda>>)

A quick way to get an object, with default… everything (but the key, you know, a clue of what you want)

### dol.misc.set_obj(k, v, store=Files(rootdir='', subpath='', pattern_for_field=None, max_levels=None, include_hidden=False, assert_rootdir_existence=False), outgoing_val_trans_for_key={'.bin': <function identity_method>, '.cnf': <function <lambda>>, '.conf': <function <lambda>>, '.config': <function <lambda>>, '.csv': <function csv_fileobj>, '.gz': <function compress>, '.gzip': <function compress>, '.ini': <function <lambda>>, '.json': <function <lambda>>, '.pickle': <function <lambda>>, '.pkl': <function <lambda>>, '.txt': <function <lambda>>}, func_key=<function <lambda>>)

A quick way to set an object, with defaults for everything
(but the key and value, you know, a clue of what you want to store).
