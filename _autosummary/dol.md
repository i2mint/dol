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

| [`appendable`](dol.appendable.md#dol.appendable)([store_cls, return_keys, ...])   | Makes a new class with append (and consequential extend) methods                                                |
|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| [`base`](dol.base.md#module-dol.base)                                       | Base classes for making stores.                                                                                 |
| [`caching`](dol.caching.md#module-dol.caching)                                 | Tools to add caching layers to stores and methods.                                                              |
| [`content`](dol.content.md#module-dol.content)                                 | Content references and content-addressed storage — the flat "blob" layer.                                       |
| [`dig`](dol.dig.md#module-dol.dig)                                         | Layers introspection: walk the layers of a wrapped store and trace a key through them.                          |
| [`errors`](dol.errors.md#module-dol.errors)                                   | Error objects and utils.                                                                                        |
| [`explicit`](dol.explicit.md#module-dol.explicit)                               | Stores whose keys are given explicitly, with values fetched lazily from a source.                               |
| [`filesys`](dol.filesys.md#module-dol.filesys)                                 | File system access: dict-like stores over folders and files.                                                    |
| [`kv_codecs`](dol.kv_codecs.md#module-dol.kv_codecs)                             | Tools to make Key-Value Codecs (encoder-decoder pairs) from standard library tools.                             |
| [`misc`](dol.misc.md#module-dol.misc)                                       | Functions to read from and write to misc sources, choosing the codec from the key.                              |
| [`mixins`](dol.mixins.md#module-dol.mixins)                                   | Mixins that add or restrict store behaviours.                                                                   |
| [`naming`](dol.naming.md#module-dol.naming)                                   | This module is about generating, validating, and operating on (parametrized) fields (i.e. strings, e.g. paths). |
| [`paths`](dol.paths.md#module-dol.paths)                                     | Module for path (and path-like) object manipulation                                                             |
| [`recipes`](dol.recipes.md#module-dol.recipes)                                 | Recipes using dol                                                                                               |
| [`signatures`](dol.signatures.md#module-dol.signatures)                           | Signature calculus: Tools to make it easier to work with function's signatures.                                 |
| [`sources`](dol.sources.md#module-dol.sources)                                 | Key-value views of disparate sources.                                                                           |
| [`tools`](dol.tools.md#module-dol.tools)                                     | Various tools to add functionality to stores.                                                                   |
| [`trans`](dol.trans.md#module-dol.trans)                                     | Tools to wrap stores with key/value transforms, filters, caches and other layers.                               |
| [`trash`](dol.trash.md#module-dol.trash)                                     | Cross-platform file trash/recycle bin functionality for dol.                                                    |
| [`util`](dol.util.md#module-dol.util)                                       | General util objects: function composition, grouping, partial classes, file helpers.                            |
| [`zipfiledol`](dol.zipfiledol.md#module-dol.zipfiledol)                           | Data object layers and other utils to work with zip files.                                                      |
