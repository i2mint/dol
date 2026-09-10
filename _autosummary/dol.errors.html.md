# dol.errors

Error objects and utils

### Functions

| [`items_with_caught_exceptions`](#dol.errors.items_with_caught_exceptions)(d[, callback, ...])   | Do what Mapping.items() does, but catching exceptions when getting the values for a key.   |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|

### Exceptions

| [`AlreadyExists`](#dol.errors.AlreadyExists)             | To use if an object already exists (and shouldn't; for example, to protect overwrites)           |
|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| [`DeletionsNotAllowed`](#dol.errors.DeletionsNotAllowed)       | Delete OperationNotAllowed                                                                       |
| [`IterationNotAllowed`](#dol.errors.IterationNotAllowed)       | Iteration OperationNotAllowed                                                                    |
| [`KeyValidationError`](#dol.errors.KeyValidationError)        | Error to raise when a key is not valid                                                           |
| [`MethodFuncNotValid`](#dol.errors.MethodFuncNotValid)        | Use when method function is not valid                                                            |
| [`MethodNameAlreadyExists`](#dol.errors.MethodNameAlreadyExists)   | To use when a method name already exists (and shouldn't)                                         |
| [`NoSuchKeyError`](#dol.errors.NoSuchKeyError)            | When a requested key doesn't exist                                                               |
| [`NotAllowed`](#dol.errors.NotAllowed)                | To use to indicate that something is not allowed                                                 |
| [`NotValid`](#dol.errors.NotValid)                  | To use to indicate when an object doesn't fit expected properties                                |
| [`OperationNotAllowed`](#dol.errors.OperationNotAllowed)       | When a given operation is not allowed (through being disabled, conditioned, or just implemented) |
| [`OverWritesNotAllowedError`](#dol.errors.OverWritesNotAllowedError) | Error to raise when a writes to existing keys are not allowed                                    |
| [`ReadsNotAllowed`](#dol.errors.ReadsNotAllowed)           | Read OperationNotAllowed                                                                         |
| [`SetattrNotAllowed`](#dol.errors.SetattrNotAllowed)         | An attribute was requested to be set, but some conditions didn't apply                           |
| [`WritesNotAllowed`](#dol.errors.WritesNotAllowed)          | Write OperationNotAllowed                                                                        |

### *exception* dol.errors.AlreadyExists

Bases: [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)

To use if an object already exists (and shouldn’t; for example, to protect overwrites)

### *exception* dol.errors.DeletionsNotAllowed

Bases: [`OperationNotAllowed`](#dol.errors.OperationNotAllowed)

Delete OperationNotAllowed

### *exception* dol.errors.IterationNotAllowed

Bases: [`OperationNotAllowed`](#dol.errors.OperationNotAllowed)

Iteration OperationNotAllowed

### *exception* dol.errors.KeyValidationError

Bases: [`NotValid`](#dol.errors.NotValid)

Error to raise when a key is not valid

### *exception* dol.errors.MethodFuncNotValid

Bases: [`NotValid`](#dol.errors.NotValid)

Use when method function is not valid

### *exception* dol.errors.MethodNameAlreadyExists

Bases: [`AlreadyExists`](#dol.errors.AlreadyExists)

To use when a method name already exists (and shouldn’t)

### *exception* dol.errors.NoSuchKeyError

Bases: [`KeyError`](https://docs.python.org/3/library/exceptions.html#KeyError)

When a requested key doesn’t exist

### *exception* dol.errors.NotAllowed

Bases: [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception)

To use to indicate that something is not allowed

### *exception* dol.errors.NotValid

Bases: [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError), [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError)

To use to indicate when an object doesn’t fit expected properties

### *exception* dol.errors.OperationNotAllowed

Bases: [`NotAllowed`](#dol.errors.NotAllowed), [`NotImplementedError`](https://docs.python.org/3/library/exceptions.html#NotImplementedError)

When a given operation is not allowed (through being disabled, conditioned, or just implemented)

### *exception* dol.errors.OverWritesNotAllowedError

Bases: [`OperationNotAllowed`](#dol.errors.OperationNotAllowed)

Error to raise when a writes to existing keys are not allowed

### *exception* dol.errors.ReadsNotAllowed

Bases: [`OperationNotAllowed`](#dol.errors.OperationNotAllowed)

Read OperationNotAllowed

### *exception* dol.errors.SetattrNotAllowed

Bases: [`NotAllowed`](#dol.errors.NotAllowed)

An attribute was requested to be set, but some conditions didn’t apply

### *exception* dol.errors.WritesNotAllowed

Bases: [`OperationNotAllowed`](#dol.errors.OperationNotAllowed)

Write OperationNotAllowed

### dol.errors.items_with_caught_exceptions(d, callback=None, catch_exceptions=(<class 'Exception'>, ), yield_callback_output=False)

Do what Mapping.items() does, but catching exceptions when getting the values for a key.

Some time your `store.items()` is annoying because of some exceptions that happen
when you’re retrieving some value for some of the keys.

Yes, if that happens, it’s that something is wrong with your store, and yes,
if it’s a store that’s going to be used a lot, you really should build the right store
that doesn’t have that problem.

But now that we appeased the paranoid naysayers with that warning, let’s get to business:
Sometimes you just want to get through the hurdle to get the job done. Sometimes your store is good enough,
except for a few exceptions. Sometimes your store gets it’s keys from a large pool of possible keys
(e.g. github stores or kaggle stores, or any store created by a free-form search seed),
so you can’t really depend on the fact that all the keys given by your key iterator
will give you a value without exception
– especially if you slapped on a bunch of post-processing on the out-coming values.

So you can right a for loop to iterate over your keys, catch the exceptions, do something with it…

Or, in many cases, you can just use `items_with_caught_exceptions`.

* **Parameters:**
  * **d** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – Any Mapping
  * **catch_exceptions** – A tuple of exceptions that should be caught
  * **callback** – 

    A function that will be called every time an exception is caught.
    The signature of the callback function is required to be:
    > k (key), e (error obj), d (mapping), i (index)

    but
* **Returns:**
  An (key, val) generator with exceptions caught

```pycon
>>> from collections.abc import Mapping
>>> class Test(Mapping):  # a Mapping class that has keys 0..9, but raises of KeyError if the key is not even
...     n = 10
...     def __iter__(self):
...         yield from range(2, self.n)
...     def __len__(self):
...         return self.n
...     def __getitem__(self, k):
...         if k % 2 == 0:
...             return k
...         else:
...             raise KeyError('Not even')
>>>
>>> list(items_with_caught_exceptions(Test()))
[(2, 2), (4, 4), (6, 6), (8, 8)]
>>>
>>> def my_log(k, e):
...     print(k, e)
>>> list(items_with_caught_exceptions(Test(), callback=my_log))
3 'Not even'
5 'Not even'
7 'Not even'
9 'Not even'
[(2, 2), (4, 4), (6, 6), (8, 8)]
>>> def my_other_log(i):
...     print(i)
>>> list(items_with_caught_exceptions(Test(), callback=my_other_log))
1
3
5
7
[(2, 2), (4, 4), (6, 6), (8, 8)]
```
