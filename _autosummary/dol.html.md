# dol

Core tools to build simple interfaces to complex data sources and bend the interface to your will (and need)

### Functions

| [`ihead`](#dol.ihead)(store[, n])   | Get the first item of an iterable, or a list of the first n items   |
|----------------------------------------------------------------------|---------------------------------------------------------------------|
| [`kvhead`](#dol.kvhead)(store[, n])  | Get the first item of a kv store, or a list of the first n items    |

### dol.ihead(store, n=1)

Get the first item of an iterable, or a list of the first n items

### dol.kvhead(store, n=1)

Get the first item of a kv store, or a list of the first n items

### Modules

| [`appendable`](dol.appendable.html.md#dol.appendable)([store_cls, return_keys, ...])   | Makes a new class with append (and consequential extend) methods                                               |
|-------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| [`base`](dol.base.html.md#module-dol.base)                                       | Base classes for making stores.                                                                                |
| [`caching`](dol.caching.html.md#module-dol.caching)                                 | Tools to add caching layers to stores and methods.                                                             |
| [`content`](dol.content.html.md#module-dol.content)                                 | Content references and content-addressed storage — the flat "blob" layer.                                      |
| [`dig`](dol.dig.html.md#module-dol.dig)                                         | Layers introspection                                                                                           |
| [`errors`](dol.errors.html.md#module-dol.errors)                                   | Error objects and utils                                                                                        |
| [`explicit`](dol.explicit.html.md#module-dol.explicit)                               | utils to make stores based on a the input data itself                                                          |
| [`filesys`](dol.filesys.html.md#module-dol.filesys)                                 | File system access                                                                                             |
| [`kv_codecs`](dol.kv_codecs.html.md#module-dol.kv_codecs)                             | Tools to make Key-Value Codecs (encoder-decoder pairs) from standard library tools.                            |
| [`misc`](dol.misc.html.md#module-dol.misc)                                       | Functions to read from and write to misc sources                                                               |
| [`mixins`](dol.mixins.html.md#module-dol.mixins)                                   | Mixins                                                                                                         |
| [`naming`](dol.naming.html.md#module-dol.naming)                                   | This module is about generating, validating, and operating on (parametrized) fields (i.e. stings, e.g. paths). |
| [`paths`](dol.paths.html.md#module-dol.paths)                                     | Module for path (and path-like) object manipulation                                                            |
| [`recipes`](dol.recipes.html.md#module-dol.recipes)                                 | Recipes using dol                                                                                              |
| [`signatures`](dol.signatures.html.md#module-dol.signatures)                           | Signature calculus: Tools to make it easier to work with function's signatures.                                |
| [`sources`](dol.sources.html.md#module-dol.sources)                                 | This module contains key-value views of disparate sources.                                                     |
| [`tools`](dol.tools.html.md#module-dol.tools)                                     | Various tools to add functionality to stores                                                                   |
| [`trans`](dol.trans.html.md#module-dol.trans)                                     | Transformation/wrapping tools                                                                                  |
| [`trash`](dol.trash.html.md#module-dol.trash)                                     | Cross-platform file trash/recycle bin functionality for dol.                                                   |
| [`util`](dol.util.html.md#module-dol.util)                                       | General util objects                                                                                           |
| [`zipfiledol`](dol.zipfiledol.html.md#module-dol.zipfiledol)                           | Data object layers and other utils to work with zip files.                                                     |
