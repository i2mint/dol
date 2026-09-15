# dol

Core tools to build simple interfaces to complex data sources and bend the interface to your will (and need).

`dol` wraps any storage backend (files, S3, databases, dicts) behind a dict-like
interface, and transforms that interface with composable layers. Start with
`wrap_kvs` (key/value transforms), the file stores (`Files`, `TextFiles`,
`JsonFiles`, `PickleFiles`), the ready-made codecs (`ValueCodecs`, `KeyCodecs`),
`filt_iter` (key filtering) and `cache_this` (caching).

```pycon
>>> from dol import wrap_kvs
>>> import json
>>> s = wrap_kvs({}, obj_of_data=json.loads, data_of_obj=json.dumps)
>>> s['a'] = {'x': 1}
>>> s['a'], s.store
({'x': 1}, {'a': '{"x": 1}'})
```

### Functions

| [`ihead`](#dol.ihead)(store[, n])   | Get the first item of an iterable, or a list of the first n items   |
|----------------------------------------------------------------------|---------------------------------------------------------------------|
| [`kvhead`](#dol.kvhead)(store[, n])  | Get the first item of a kv store, or a list of the first n items    |

### dol.ihead(store, n=1)

Get the first item of an iterable, or a list of the first n items

### dol.kvhead(store, n=1)

Get the first item of a kv store, or a list of the first n items

### Modules

| [`appendable`](dol.appendable.html.md#dol.appendable)([store_cls, return_keys, ...])   | Makes a new class with append (and consequential extend) methods                                                |
|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| [`base`](dol.base.html.md#module-dol.base)                                       | Base classes for making stores.                                                                                 |
| [`caching`](dol.caching.html.md#module-dol.caching)                                 | Tools to add caching layers to stores and methods.                                                              |
| [`content`](dol.content.html.md#module-dol.content)                                 | Content references and content-addressed storage — the flat "blob" layer.                                       |
| [`dig`](dol.dig.html.md#module-dol.dig)                                         | Layers introspection: walk the layers of a wrapped store and trace a key through them.                          |
| [`errors`](dol.errors.html.md#module-dol.errors)                                   | Error objects and utils.                                                                                        |
| [`explicit`](dol.explicit.html.md#module-dol.explicit)                               | Stores whose keys are given explicitly, with values fetched lazily from a source.                               |
| [`filesys`](dol.filesys.html.md#module-dol.filesys)                                 | File system access: dict-like stores over folders and files.                                                    |
| [`kv_codecs`](dol.kv_codecs.html.md#module-dol.kv_codecs)                             | Tools to make Key-Value Codecs (encoder-decoder pairs) from standard library tools.                             |
| [`misc`](dol.misc.html.md#module-dol.misc)                                       | Functions to read from and write to misc sources, choosing the codec from the key.                              |
| [`mixins`](dol.mixins.html.md#module-dol.mixins)                                   | Mixins that add or restrict store behaviours.                                                                   |
| [`naming`](dol.naming.html.md#module-dol.naming)                                   | This module is about generating, validating, and operating on (parametrized) fields (i.e. strings, e.g. paths). |
| [`paths`](dol.paths.html.md#module-dol.paths)                                     | Module for path (and path-like) object manipulation                                                             |
| [`recipes`](dol.recipes.html.md#module-dol.recipes)                                 | Recipes using dol                                                                                               |
| [`signatures`](dol.signatures.html.md#module-dol.signatures)                           | Signature calculus: Tools to make it easier to work with function's signatures.                                 |
| [`sources`](dol.sources.html.md#module-dol.sources)                                 | Key-value views of disparate sources.                                                                           |
| [`tools`](dol.tools.html.md#module-dol.tools)                                     | Various tools to add functionality to stores.                                                                   |
| [`trans`](dol.trans.html.md#module-dol.trans)                                     | Tools to wrap stores with key/value transforms, filters, caches and other layers.                               |
| [`trash`](dol.trash.html.md#module-dol.trash)                                     | Cross-platform file trash/recycle bin functionality for dol.                                                    |
| [`util`](dol.util.html.md#module-dol.util)                                       | General util objects: function composition, grouping, partial classes, file helpers.                            |
| [`zipfiledol`](dol.zipfiledol.html.md#module-dol.zipfiledol)                           | Data object layers and other utils to work with zip files.                                                      |
