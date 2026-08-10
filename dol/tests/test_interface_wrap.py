"""Tests for the Option G prototype: dol/_interface_wrap.py.

Covers the #83/#86 census shapes (scalar key methods, iterable-of-keys,
key-value iterators), flat stacking (wrap-of-wrap extends the stack, never
nests), the no-double-apply guarantees (internal self-calls; prefix-owning
leaves), the pickle matrix that today's delegation machinery fails, the
undeclared-surface loudness policy, and lazy iterator mapping.
"""

import pickle
import pytest
from typing import Callable, Iterable, Iterator, Optional, Protocol, TypeVar

from dol._interface_wrap import (
    Codec,
    InterfaceSpec,
    InterfaceProxy,
    InterfaceWrapError,
    UndeclaredAttributeError,
    UnsupportedSpecShape,
    interface_wrap,
)

KT = TypeVar('KT')
VT = TypeVar('VT')


# --- module-level (picklable) codec functions -------------------------------


def add_json(k):
    return k + '.json'


def strip_json(k):
    return k[:-5]


def prefix_x(k):
    return 'x/' + k


def strip_x(k):
    return k[2:]


def int_to_str(v):
    return str(v)


def str_to_int(v):
    return int(v)


json_key_codec = Codec(encoder=add_json, decoder=strip_json)
x_key_codec = Codec(encoder=prefix_x, decoder=strip_x)
value_codec = Codec(encoder=int_to_str, decoder=str_to_int)


# --- a leaf in the shape of the #83 census ----------------------------------


class Bucket:
    """A backend with keyed non-Mapping methods — the #83 census shape.

    Owns its own prefix arithmetic (like s3dol's S3BucketReader): public keys
    are relative; wire keys are prefixed. Methods use the leaf's own public
    interface internally, so they are correct on the bare leaf — the property
    the boundary model must preserve.
    """

    def __init__(self, wire=None, *, prefix='logs/'):
        self.wire = wire if wire is not None else {}
        self.prefix = prefix

    # internal (wire-domain) helpers
    def _wire_key(self, k):
        return self.prefix + k

    # the Mapping-ish surface
    def __getitem__(self, k):
        return self.wire[self._wire_key(k)]

    def __setitem__(self, k, v):
        self.wire[self._wire_key(k)] = v

    def __delitem__(self, k):
        del self.wire[self._wire_key(k)]

    def __iter__(self):
        p = self.prefix
        return (w[len(p):] for w in self.wire if w.startswith(p))

    def __contains__(self, k):
        return self._wire_key(k) in self.wire

    # the census shapes
    def url_for(self, k, *, expires_in=3600):
        return f'https://x.example/{self._wire_key(k)}?e={expires_in}'

    def delete_many(self, keys):
        for k in keys:
            del self[k]  # internal self-call: stays below the boundary

    def items_page(self):
        for k in self:
            yield (k, self[k])

    def copy_key(self, src, dst):
        self[dst] = self[src]


class BucketInterface(Protocol[KT, VT]):
    def __getitem__(self, k: KT) -> VT: ...
    def __setitem__(self, k: KT, v: VT) -> None: ...
    def __delitem__(self, k: KT) -> None: ...
    def __iter__(self) -> Iterator[KT]: ...
    def __contains__(self, k: KT) -> bool: ...
    def url_for(self, k: KT, *, expires_in: int = 3600) -> str: ...
    def delete_many(self, keys: Iterable[KT]) -> None: ...
    def items_page(self) -> Iterator[tuple[KT, VT]]: ...
    def copy_key(self, src: KT, dst: KT) -> None: ...


def mk_bucket():
    return Bucket({'logs/a.json': '1', 'logs/b.json': '2'})


def wrap_bucket(bucket=None, **kwargs):
    # `wire` and `prefix` are public data attributes of the leaf; the loudness
    # policy makes forwarding them an explicit choice (this is the designed
    # gesture, not a workaround).
    kwargs.setdefault('passthrough', {'wire', 'prefix'})
    return interface_wrap(
        bucket if bucket is not None else mk_bucket(),
        spec=BucketInterface,
        codecs=dict(KT=json_key_codec, VT=value_codec),
        **kwargs,
    )


# --- the boundary invariant on the census shapes ----------------------------


def test_mapping_surface():
    s = wrap_bucket()
    assert s['a'] == 1
    assert sorted(s) == ['a', 'b']
    assert 'a' in s and 'zzz' not in s
    s['c'] = 3
    assert s.__wrapped__.wire['logs/c.json'] == '3'
    del s['c']
    assert 'c' not in s


def test_scalar_key_method_gets_mapped_key():
    """#83's minimal repro shape: url_for must see the leaf-domain key."""
    s = wrap_bucket()
    assert s.url_for('a') == 'https://x.example/logs/a.json?e=3600'
    # non-role kwargs pass through untouched
    assert s.url_for('a', expires_in=60).endswith('?e=60')


def test_iterable_of_keys_arg():
    s = wrap_bucket()
    s.delete_many(['a'])
    assert sorted(s) == ['b']


def test_iterator_of_pairs_return_is_lazy_and_mapped():
    s = wrap_bucket()
    pages = s.items_page()
    assert iter(pages) is pages  # stayed an iterator (lazy)
    assert sorted(pages) == [('a', 1), ('b', 2)]


def test_two_key_params():
    s = wrap_bucket()
    s.copy_key('a', 'target')
    assert s['target'] == 1
    assert 'logs/target.json' in s.__wrapped__.wire


def test_internal_self_calls_do_not_double_apply():
    """delete_many calls del self[k] internally; that call must stay below
    the boundary (single application of the key codec)."""
    s = wrap_bucket()
    s.delete_many(['a', 'b'])
    assert s.__wrapped__.wire == {}


def test_prefix_owning_leaf_composes():
    """The leaf's own prefix arithmetic composes with the stack: encoder maps
    outer->leaf-public, leaf maps leaf-public->wire. No double-apply."""
    s = wrap_bucket()
    # outer 'a' -> leaf 'a.json' -> wire 'logs/a.json'
    assert s.url_for('a').split('/', 3)[-1] == 'logs/a.json?e=3600'


# --- flat stacking ----------------------------------------------------------


def test_wrap_of_wrap_extends_stack_not_nests():
    s1 = wrap_bucket()
    s2 = interface_wrap(
        s1, spec=BucketInterface, codecs=dict(KT=x_key_codec)
    )
    assert isinstance(s2, InterfaceProxy)
    assert s2.__wrapped__ is s1.__wrapped__  # SAME leaf: no nesting
    assert len(s2._self_stack) == 2
    # outer key 'a' -> +'x/' is OUTER-most? No: second wrap is outer.
    # Encoders run outer->inner: x_key first? Layer order: stack is
    # innermost-first, so layer0=json, layer1=x. Encode: x then json.
    assert s2._encode_role('KT', 'a') == 'x/a.json'
    # inverse mapping (the missing primitive of #83 §5.4): decoder walk
    assert s2._decode_role('KT', 'x/a.json') == 'a'


def test_stacked_read_write_roundtrip():
    leaf = Bucket({})
    s1 = wrap_bucket(leaf)
    s2 = interface_wrap(s1, spec=BucketInterface, codecs=dict(KT=x_key_codec))
    s2['a'] = 7
    assert leaf.wire == {'logs/x/a.json': '7'}
    assert s2['a'] == 7
    assert list(s2) == ['a']
    assert s2.url_for('a').split('/', 3)[-1].startswith('logs/x/a.json')


def test_deep_stack_flat_cost_object_graph():
    """Six layers: still ONE proxy, one leaf, six stack entries."""
    s = wrap_bucket()
    for _ in range(5):
        s = interface_wrap(s, spec=BucketInterface, codecs=dict(KT=x_key_codec))
    assert isinstance(s.__wrapped__, Bucket)  # not a proxy: no nesting
    assert len(s._self_stack) == 6
    assert s._encode_role('KT', 'a') == 'x/x/x/x/x/a.json'


# --- pickling ---------------------------------------------------------------


def test_pickle_roundtrip_instance():
    s = wrap_bucket()
    s2 = pickle.loads(pickle.dumps(s))
    assert s2['a'] == 1
    assert sorted(s2) == ['a', 'b']
    assert s2.url_for('a') == 'https://x.example/logs/a.json?e=3600'


def test_pickle_roundtrip_stacked():
    """Stacked wraps pickle — the case today's delegation machinery fails
    (anonymous intermediate classes in the reduce payload)."""
    s = interface_wrap(
        wrap_bucket(), spec=BucketInterface, codecs=dict(KT=x_key_codec)
    )
    s2 = pickle.loads(pickle.dumps(s))
    assert len(s2._self_stack) == 2
    assert s2._encode_role('KT', 'a') == 'x/a.json'


def test_pickle_preserves_transform_behavior_after_write():
    s = pickle.loads(pickle.dumps(wrap_bucket()))
    s['new'] = 9
    assert s.__wrapped__.wire['logs/new.json'] == '9'


def test_pickle_with_lambda_codec_fails_loudly():
    s = interface_wrap(
        mk_bucket(),
        spec=BucketInterface,
        codecs=dict(KT=Codec(encoder=lambda k: k, decoder=lambda k: k)),
        passthrough={'wire', 'prefix'},
    )
    with pytest.raises(Exception):  # PicklingError or AttributeError
        pickle.dumps(s)


# --- loudness policies ------------------------------------------------------


class Leaky:
    def __getitem__(self, k):
        return 42

    def surprise_delete(self, k):
        """A public keyed method the spec forgot."""


class MinimalGet(Protocol[KT, VT]):
    def __getitem__(self, k: KT) -> VT: ...


def test_undeclared_public_method_raises_at_wrap_time():
    with pytest.raises(UndeclaredAttributeError) as exc:
        interface_wrap(
            Leaky(), spec=MinimalGet, codecs=dict(KT=json_key_codec)
        )
    assert 'surprise_delete' in str(exc.value)


def test_undeclared_passthrough_is_explicit_and_works():
    s = interface_wrap(
        Leaky(),
        spec=MinimalGet,
        codecs=dict(KT=json_key_codec),
        undeclared='passthrough',
    )
    assert s.surprise_delete is not None  # forwarded verbatim, by choice


def test_undeclared_exclude_hides_and_raises_on_use():
    s = interface_wrap(
        Leaky(),
        spec=MinimalGet,
        codecs=dict(KT=json_key_codec),
        undeclared='exclude',
    )
    with pytest.raises(UndeclaredAttributeError):
        _ = s.surprise_delete


def test_unknown_codec_role_raises():
    with pytest.raises(InterfaceWrapError):
        interface_wrap(
            Leaky(),
            spec=MinimalGet,
            codecs=dict(QT=json_key_codec),  # QT occurs nowhere in the spec
            undeclared='exclude',
        )


def test_unsupported_shape_refuses_at_wrap_time():
    class Bad(Protocol[KT, VT]):
        def weird(self, k: Callable[[KT], int]) -> None: ...

    with pytest.raises(UnsupportedSpecShape):
        InterfaceSpec.from_annotated(Bad)


# --- capability mirroring ---------------------------------------------------


def test_missing_leaf_method_not_resurrected():
    """A spec'd method the leaf lacks must NOT appear on the proxy (contrast
    _filt_iter's __len__-resurrection bug)."""

    class NoLen:
        def __getitem__(self, k):
            return 1

    class SpecWithLen(Protocol[KT, VT]):
        def __getitem__(self, k: KT) -> VT: ...
        def __len__(self) -> int: ...

    s = interface_wrap(
        NoLen(), spec=SpecWithLen, codecs=dict(KT=json_key_codec),
        undeclared='exclude',
    )
    with pytest.raises(TypeError):
        len(s)


# --- generalization beyond KT/VT --------------------------------------------


QT = TypeVar('QT')


def test_arbitrary_role_lane():
    """The mechanism is role-generic: any TypeVar name is a codec lane."""

    class Queryable(Protocol[QT]):
        def search(self, q: QT) -> list: ...

    class Engine:
        def search(self, q):
            return [q]

    s = interface_wrap(
        Engine(),
        spec=Queryable,
        codecs=dict(QT=Codec(encoder=str.upper, decoder=str.lower)),
    )
    assert s.search('hello') == ['HELLO']


# --- dict-form spec (no typing required) ------------------------------------


def test_dict_form_spec():
    # Integer keys = positional parameter index (no typing, no name coupling
    # to the leaf's own parameter names — dict names its params key/value).
    spec = {
        '__getitem__': {0: 'KT', 'return': 'VT'},
        '__setitem__': {0: 'KT', 1: 'VT'},
    }
    d = {}
    s = interface_wrap(
        d,
        spec=spec,
        codecs=dict(KT=json_key_codec, VT=value_codec),
        undeclared='exclude',
    )
    s['a'] = 5
    assert d == {'a.json': '5'}
    assert s['a'] == 5


# --- optional / laziness edge cases ----------------------------------------


def test_optional_key_param():
    class OptGet(Protocol[KT, VT]):
        def find(self, k: Optional[KT]) -> Optional[VT]: ...

    class L:
        def find(self, k):
            return None if k is None else '7'

    s = interface_wrap(
        L(), spec=OptGet, codecs=dict(KT=json_key_codec, VT=value_codec),
        undeclared='exclude',
    )
    assert s.find(None) is None
    assert s.find('a') == 7


# --- panel-driven hardening tests (refute round 1) ---------------------------


from dol._interface_wrap import UnderAnnotatedSpecError


def test_var_positional_role_maps_elementwise():
    """*keys: KT must encode each element, not the tuple (was a silent bug)."""

    class VarSpec(Protocol[KT]):
        def delete(self, *keys: KT) -> None: ...

    class L:
        def delete(self, *keys):
            self.got = keys

    s = interface_wrap(L(), spec=VarSpec, codecs=dict(KT=json_key_codec),
                       undeclared='exclude')
    s.delete('a', 'b')
    assert s.__wrapped__.got == ('a.json', 'b.json')


def test_var_keyword_role_refuses():
    class KwSpec(Protocol[VT]):
        def update_all(self, **kv: VT) -> None: ...

    class L:
        def update_all(self, **kv): ...

    with pytest.raises(UnsupportedSpecShape):
        interface_wrap(L(), spec=KwSpec, codecs=dict(VT=value_codec),
                       undeclared='exclude')


def test_unannotated_param_in_spec_method_refuses():
    """A spec method with an unannotated param is silence-by-omission one
    level down — refuse at compile time."""

    class Sloppy(Protocol[KT]):
        def url_for(self, k) -> str: ...  # forgot the annotation

    with pytest.raises(UnderAnnotatedSpecError):
        InterfaceSpec.from_annotated(Sloppy)


def test_property_in_spec_refuses_not_vanishes():
    class WithProp(Protocol[KT]):
        def __getitem__(self, k: KT) -> str: ...
        rootdir = property(lambda self: '/')

    with pytest.raises(UnsupportedSpecShape):
        InterfaceSpec.from_annotated(WithProp)


def test_none_default_never_encoded():
    """`k: KT = None`: the None default is leaf-domain; never encode it.
    Also converges the 3.10 (implicit-Optional) vs 3.11+ compilation."""

    class DefSpec(Protocol[KT]):
        def latest(self, k: KT = None) -> str: ...

    class L:
        def latest(self, k=None):
            return f'got:{k}'

    s = interface_wrap(L(), spec=DefSpec, codecs=dict(KT=json_key_codec),
                       undeclared='exclude')
    assert s.latest() == 'got:None'       # default: untouched
    assert s.latest(None) == 'got:None'   # explicit None: untouched
    assert s.latest('a') == 'got:a.json'  # real key: encoded


def test_in_flight_iterator_survives_stack_extension():
    """Wrapping is copy-not-mutate: an iterator obtained before a new wrap
    keeps the pipelines it was compiled with."""
    s1 = wrap_bucket()
    it = iter(sorted(s1))
    first = it if isinstance(it, str) else None  # noqa: just consume below
    got_first = next(iter(sorted(s1)))
    s2 = interface_wrap(s1, spec=BucketInterface, codecs=dict(KT=x_key_codec))
    # s1's own iteration is unaffected by s2's existence
    assert sorted(s1) == ['a', 'b']
    assert len(s1._self_stack) == 1 and len(s2._self_stack) == 2
    assert got_first == 'a'


def test_unspecced_dunder_absent_not_leaked():
    """dict's __or__ must NOT be silently mirrored raw (the basepy-verified
    transform-bypass leak): a dunder outside the spec simply doesn't exist
    on the proxy — loud TypeError, no silent raw data."""
    d = {'a.json': '1'}
    s = interface_wrap(
        d,
        spec={'__getitem__': {0: 'KT', 'return': 'VT'}},
        codecs=dict(KT=json_key_codec, VT=value_codec),
        undeclared='exclude',
    )
    with pytest.raises(TypeError):
        s | {'b': 2}


def test_nested_iterator_of_iterators():
    class NestSpec(Protocol[KT]):
        def batches(self) -> Iterator[Iterator[KT]]: ...

    class L:
        def batches(self):
            yield iter(['a.json'])
            yield iter(['b.json'])

    s = interface_wrap(L(), spec=NestSpec, codecs=dict(KT=json_key_codec),
                       undeclared='exclude')
    assert [list(b) for b in s.batches()] == [['a'], ['b']]


def test_wrapping_legacy_store_warns():
    from dol import wrap_kvs

    legacy = wrap_kvs({'a.json': '1'}, obj_of_data=str)
    with pytest.warns(UserWarning, match='legacy dol Store'):
        interface_wrap(
            legacy,
            spec={'__getitem__': {0: 'KT'}},
            codecs=dict(KT=json_key_codec),
            undeclared='passthrough',
        )


# --- independent code-review round (findings 1-9) ----------------------------


import copy


def test_dict_form_spec_pickles_and_copies():
    """Finding 2: __reduce__ round-trips the normalized 3-tuple form through
    from_dict; copy.copy rides the same path."""
    spec = {'__getitem__': {0: 'KT', 'return': 'VT'}}
    d = {'a.json': '1'}
    s = interface_wrap(d, spec=spec,
                       codecs=dict(KT=json_key_codec, VT=value_codec),
                       undeclared='exclude')
    s2 = pickle.loads(pickle.dumps(s))
    assert s2['a'] == 1
    s3 = copy.copy(s)
    assert s3['a'] == 1
    assert s3.__wrapped__ is not None  # shallow copy shares nothing broken


def test_explicit_dunder_access_is_loud():
    """Finding 3: s.__contains__ / s.__or__ must not silently hand out the
    leaf's raw bound method — AttributeError keeps duck typing honest."""
    s = wrap_bucket()
    with pytest.raises(AttributeError):
        s.__or__
    # spec'd dunders remain accessible (they live on the class)
    assert s.__contains__('a') is True


def test_no_sequence_protocol_iteration_leak():
    """A __getitem__-only spec must not let iter() invent integer-key
    iteration via the legacy sequence protocol."""
    s = interface_wrap(
        {'a.json': '1'},
        spec={'__getitem__': {0: 'KT', 'return': 'VT'}},
        codecs=dict(KT=json_key_codec, VT=value_codec),
        undeclared='exclude',
    )
    with pytest.raises(TypeError):
        iter(s)


def test_keyword_call_of_specd_method():
    """Finding 4: POSITIONAL_OR_KEYWORD contracts include keyword calls;
    the spec's name is the contract even when the leaf names it differently."""

    class KwSpec(Protocol[KT]):
        def url_for(self, key: KT) -> str: ...

    class L:
        def url_for(self, target):  # leaf uses a DIFFERENT param name
            return 'u/' + target

    s = interface_wrap(L(), spec=KwSpec, codecs=dict(KT=json_key_codec),
                       undeclared='exclude')
    assert s.url_for('a') == 'u/a.json'
    assert s.url_for(key='a') == 'u/a.json'


def test_iterable_arg_survives_reiteration():
    """Finding 5: Iterable (re-iterable contract) args are materialized;
    a leaf that iterates twice sees both passes."""

    class TwoPass(Protocol[KT]):
        def pairs(self, keys: Iterable[KT]) -> list: ...

    class L:
        def pairs(self, keys):
            return [list(keys), list(keys)]

    s = interface_wrap(L(), spec=TwoPass, codecs=dict(KT=json_key_codec),
                       undeclared='exclude')
    assert s.pairs(['a']) == [['a.json'], ['a.json']]


def test_invalid_undeclared_policy_refuses():
    with pytest.raises(ValueError):
        wrap_bucket(undeclared='riase')  # typo must not silently mean exclude


def test_dict_kv_return_shape():
    """Review gap 9a: dict[KT, VT] returns map both keys and values."""

    class BulkSpec(Protocol[KT, VT]):
        def bulk(self, ks: list[KT]) -> dict[KT, VT]: ...

    class L:
        def bulk(self, ks):
            return {k: '7' for k in ks}

    s = interface_wrap(L(), spec=BulkSpec,
                       codecs=dict(KT=json_key_codec, VT=value_codec),
                       undeclared='exclude')
    assert s.bulk(['a']) == {'a': 7}
