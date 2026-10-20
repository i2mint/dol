# dol.cache_keys

Key strategies: how `cache_this` turns an instance and a call into a cache key.

A key strategy (the `KeyStrategy` protocol) resolves a key from the method name when
the class is defined (`resolve_at_definition`) or, if it needs the instance, when the
cached attribute is accessed (`resolve_at_runtime`). The ready-made ones:

- `ExplicitKey`: a fixed key
- `ApplyToMethodName`: a function of the method name (e.g. add an extension)
- `InstanceProp`: the value of an attribute of the instance
- `ApplyToInstance`: a function of the instance
- `FromMethodArgs`: a function of the method’s arguments
- `CompositeKey`: several strategies combined

`cache_this` accepts a strategy, or anything `_resolve_key_for_cached_prop` /
`_resolve_key_for_cached_method` can turn into one (a string, a callable, …).
The strategies are also attributes of `cache_this` (`cache_this.ExplicitKey`).

These used to live in `dol.caching`, which still re-exports them (i2mint/dol#70).

```pycon
>>> from dol.cache_keys import ExplicitKey, ApplyToMethodName
>>> ExplicitKey('data.json').resolve_at_definition('load')
'data.json'
>>> ApplyToMethodName(lambda name: name + '.pkl').resolve_at_definition('load')
'load.pkl'
```

### Functions

| [`register_key_strategy`](#dol.cache_keys.register_key_strategy)(cls)   | Register a class as a KeyStrategy.   |
|-------------------------------------------------------------------------------|--------------------------------------|

### Classes

| [`ApplyToInstance`](#dol.cache_keys.ApplyToInstance)(func)                   | Apply a function to the instance to generate the key.        |
|------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| [`ApplyToMethodName`](#dol.cache_keys.ApplyToMethodName)(func)                 | Apply a function to the method name to generate the key.     |
| [`CompositeKey`](#dol.cache_keys.CompositeKey)(\*strategies[, separator]) | Combine multiple key strategies into a single composite key. |
| [`ExplicitKey`](#dol.cache_keys.ExplicitKey)(key)                        | Use an explicitly provided key value.                        |
| [`FromMethodArgs`](#dol.cache_keys.FromMethodArgs)(func)                    | Apply a function to method arguments to generate the key.    |
| [`InstanceProp`](#dol.cache_keys.InstanceProp)(prop_name)                 | Get a key from an instance property.                         |
| [`KeyStrategy`](#dol.cache_keys.KeyStrategy)(\*args, \*\*kwargs)         | Protocol defining how a key strategy should behave.          |

### *class* dol.cache_keys.ApplyToInstance(func)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Apply a function to the instance to generate the key.

#### resolve_at_definition(method_name)

Cannot resolve at definition time, need the instance.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### resolve_at_runtime(instance, method_name)

Apply the function to the instance at runtime.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* dol.cache_keys.ApplyToMethodName(func)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Apply a function to the method name to generate the key.

```pycon
>>> strategy = ApplyToMethodName(lambda name: f"{name}.cache")
>>> strategy.resolve_at_definition("my_method")
'my_method.cache'
```

#### resolve_at_definition(method_name)

Apply the function to the method name at definition time.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* dol.cache_keys.CompositeKey(\*strategies, separator='_')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Combine multiple key strategies into a single composite key.

Useful for creating keys that depend on both instance properties and method arguments.

#### resolve_at_definition(method_name)

Try to resolve all strategies at definition time.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### resolve_at_runtime(instance, method_name, \*args, \*\*kwargs)

Resolve all strategies at runtime and combine them.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* dol.cache_keys.ExplicitKey(key)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Use an explicitly provided key value.

```pycon
>>> strategy = ExplicitKey("my_key")
>>> strategy.resolve_at_definition("method_name")
'my_key'
```

#### resolve_at_definition(method_name)

Return the explicit key value at definition time.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* dol.cache_keys.FromMethodArgs(func)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Apply a function to method arguments to generate the key.

The function receives `(self, *args, **kwargs)` and should return a cache key.

#### resolve_at_definition(method_name)

Cannot resolve at definition time, need the arguments.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### resolve_at_runtime(instance, method_name, \*args, \*\*kwargs)

Apply the function to the instance and method arguments at runtime.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* dol.cache_keys.InstanceProp(prop_name)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Get a key from an instance property.

#### resolve_at_definition(method_name)

Cannot resolve at definition time, need the instance.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### resolve_at_runtime(instance, method_name)

Get the property value from the instance at runtime.

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *class* dol.cache_keys.KeyStrategy(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Protocol defining how a key strategy should behave.

#### resolve_at_definition(method_name)

Attempt to resolve the key at class definition time.

* **Parameters:**
  **method_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The name of the method being decorated.
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any) | [`None`](https://docs.python.org/3/builtins/constants.html#None)
* **Returns:**
  The resolved key or None if it can’t be resolved at definition time.

#### resolve_at_runtime(instance, method_name)

Resolve the key at runtime.
By default, this will call resolve_at_definition on method_name.

* **Parameters:**
  * **instance** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – The instance the property is being accessed on.
  * **method_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The name of the method being decorated.
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  The resolved key.

### dol.cache_keys.register_key_strategy(cls)

Register a class as a KeyStrategy.
