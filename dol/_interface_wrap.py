"""Spec-carried boundary codecs on a flat proxy (Option G prototype).

EXPERIMENTAL — private module, not exported from ``dol``. Design study:
``misc/docs/dol_issue86_design.md`` (companion to discussions #86 and the
#83/#18 design docs).

The idea, in one paragraph: a wrap is ``(leaf, spec, stack)`` where *spec*
declares, per method, at which argument/return paths the "types of interest"
(KT, VT, ... — any TypeVar "role") occur, and *stack* is a flat sequence of
codec layers, each mapping ``role -> Codec(encoder, decoder)``. Wrapping an
already-wrapped object **extends the stack over the same leaf** — there is
never a wrapper-of-wrapper. At wrap time the stack is compiled: per role, the
encoder pipeline (outer→inner) and decoder pipeline (inner→outer) are fused
into single callables; per method, a boundary plan binds parameters and return
paths to them. Method bodies always run against the **leaf's own public
interface** (``self`` is the leaf), so a method that is correct on the bare
leaf stays correct under any codec stack, and internal ``self.x()`` calls stay
below the boundary — transforms are applied exactly once, at the boundary.

What this deliberately does NOT do: it does not change what ``self`` is inside
leaf methods (Issue #18's outer-domain methods are the other half of the
problem, served by ``wrapped_self`` today), it does not rebind methods (the
rejected option D), and it is not (yet) ``wrap_kvs``.

>>> from typing import Protocol, TypeVar, Iterator, Iterable
>>> KT, VT = TypeVar('KT'), TypeVar('VT')
>>> class KvInterface(Protocol[KT, VT]):
...     def __getitem__(self, k: KT) -> VT: ...
...     def __setitem__(self, k: KT, v: VT) -> None: ...
...     def __iter__(self) -> Iterator[KT]: ...
...     def __len__(self) -> int: ...
...     def __contains__(self, k: KT) -> bool: ...
>>> d = {'a.json': '1'}
>>> codecs = dict(
...     KT=Codec(encoder=lambda k: k + '.json', decoder=lambda k: k[:-5]),
...     VT=Codec(encoder=str, decoder=int),
... )
>>> s = interface_wrap(d, spec=KvInterface, codecs=codecs)
>>> s['a']
1
>>> s['b'] = 2
>>> d
{'a.json': '1', 'b.json': '2'}
>>> list(s)
['a', 'b']
>>> 'a' in s
True

Loudness: under the default ``undeclared='exclude'`` policy, a public leaf
attribute the spec doesn't cover is hidden — USING it refuses with guidance
(it would be served with unmapped keys/values, the #83 bug class):

>>> s.get('a')  # doctest: +ELLIPSIS
Traceback (most recent call last):
...
dol._interface_wrap.UndeclaredAttributeError: 'get' is not in the interface...

Strict mode refuses at WRAP time instead (``undeclared='raise'``); and
``undeclared='passthrough'`` / ``passthrough={...}`` forward verbatim — every
escape is explicit.

For the common Mapping-shaped case there is a built-in spec and a
``wrap_kvs``-shaped facade, so simple things stay simple:

>>> t = kv_interface_wrap({}, id_of_key=lambda k: k + '.txt',
...                       key_of_id=lambda k: k[:-4])
>>> t['a'] = 'hello'
>>> list(t)
['a']
>>> t == {'a': 'hello'}
True
"""

from collections.abc import Iterable as _IterableABC, Iterator as _IteratorABC
from dataclasses import dataclass, field
from functools import cached_property
import inspect
import typing
from typing import (
    Any,
    Callable,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

__all__ = [
    "Codec",
    "InterfaceSpec",
    "InterfaceProxy",
    "MappingInterface",
    "interface_wrap",
    "kv_interface_wrap",
    "InterfaceWrapError",
    "UnsupportedSpecShape",
    "UnderAnnotatedSpecError",
    "UndeclaredAttributeError",
]


# ---------------------------------------------------------------------------
# Errors — loud by default (No Silent Failures)


class InterfaceWrapError(Exception):
    """Base for all errors raised by this module."""


class UnsupportedSpecShape(InterfaceWrapError, TypeError):
    """An annotation contains a role TypeVar at a path we cannot map.

    Raised at wrap (compile) time, never at call time: refusing early beats
    guessing (dol_issue83_design.md §5.7).
    """


class UnderAnnotatedSpecError(InterfaceWrapError, TypeError):
    """A spec'd method has a parameter with no annotation at all.

    An unannotated parameter is indistinguishable from a deliberately
    non-role parameter, which is exactly the silence-by-omission failure mode
    this mechanism exists to kill — one level down (a key parameter the spec
    author forgot to annotate would silently receive OUTER keys). Annotate
    every named parameter of a spec method: with a role TypeVar if it carries
    a role, with a concrete type (or ``Any``) to state it does not.
    """


class UndeclaredAttributeError(InterfaceWrapError, AttributeError):
    """A public attribute of the leaf is not covered by the spec.

    The census's failure mode is silence-by-omission (s3dol ADR-0011 D5), so
    an attribute that would be served with unmapped keys/values must be
    explicitly passed through or added to the spec. Under the default
    ``undeclared='exclude'`` policy this raises at USE time; under the strict
    ``'raise'`` policy, at wrap time.
    """


# ---------------------------------------------------------------------------
# Codec


@dataclass(frozen=True)
class Codec:
    """An encoder/decoder pair for one role (type of interest).

    ``encoder`` maps outer→inner (the direction of arguments going in);
    ``decoder`` maps inner→outer (the direction of results coming out).
    Same field semantics as ``dol.trans.Codec``; redefined here to keep this
    leaf module dependency-free within dol.

    ``decoded_type``/``encoded_type`` are OPTIONAL type tags (design decision
    2026-08-10, question 3): ``decoded_type`` is the outer-facing domain,
    ``encoded_type`` the inner-facing (leafward) one. When two adjacent stack
    layers both declare the facing types, stack compilation validates the
    seam (the outer layer's ``encoded_type`` must be the inner layer's
    ``decoded_type``) and refuses loudly on mismatch. ``None`` = untagged =
    unchecked, so plain ``Codec(f, g)`` keeps working.
    """

    encoder: Callable[[Any], Any]
    decoder: Callable[[Any], Any]
    decoded_type: Optional[type] = None
    encoded_type: Optional[type] = None

    def __iter__(self):
        return iter((self.encoder, self.decoder))


def _identity(x):
    return x


def _fuse(funcs):
    """Compose single-argument functions left-to-right into one callable."""
    funcs = [f for f in funcs if f is not _identity]
    if not funcs:
        return _identity
    if len(funcs) == 1:
        return funcs[0]

    def fused(x, _funcs=tuple(funcs)):
        for f in _funcs:
            x = f(x)
        return x

    return fused


# ---------------------------------------------------------------------------
# Spec introspection: find role TypeVars at paths inside annotations

# A path is a tuple of steps; each step is (origin, arg_index). The empty path
# means the annotation IS the role ("bare").


class _RoleSite(NamedTuple):
    role: str  # TypeVar name
    path: tuple  # ((origin, arg_index), ...)


def _find_role_sites(ann, roles, path=()):
    """Yield ``_RoleSite`` for each occurrence of a role TypeVar in ``ann``.

    ``roles`` maps TypeVar *name* -> TypeVar. Matching is by name, not
    identity, so a user's ``KT = TypeVar('KT')`` matches a spec authored with
    a different-but-same-named TypeVar (e.g. ``typing.KT`` re-exported by
    dol). Name collisions across genuinely different roles are the user's
    responsibility — roles are names here.
    """
    if isinstance(ann, TypeVar):
        if ann.__name__ in roles:
            yield _RoleSite(ann.__name__, path)
        return
    origin = get_origin(ann)
    if origin is None:
        return
    for i, arg in enumerate(get_args(ann)):
        if isinstance(arg, list):
            # Callable[[X, Y], R]: the parameter list arrives as a list.
            # Descend so a role inside it is SEEN (and then refused by the
            # path transformer — origin Callable is unmappable) rather than
            # silently ignored.
            for el in arg:
                yield from _find_role_sites(el, roles, path + ((origin, i),))
        else:
            yield from _find_role_sites(arg, roles, path + ((origin, i),))


def _transformer_for_path(ann, path, role_func, *, where):
    """Build f(value) applying ``role_func`` at ``path`` inside ``value``.

    ``ann`` is the (sub-)annotation the path descends into — carried along so
    each container case can inspect its own type arguments. Supports the
    container shapes we can map faithfully; anything else raises
    ``UnsupportedSpecShape`` at compile time.
    """
    if not path:
        return role_func
    (origin, index), rest = path[0], path[1:]
    sub_ann = get_args(ann)[index] if get_args(ann) else Any
    inner = _transformer_for_path(sub_ann, rest, role_func, where=where)

    if origin is list:
        return lambda v: [inner(x) for x in v]
    if origin in (set, frozenset):
        return lambda v, _o=origin: _o(inner(x) for x in v)
    if origin is tuple:
        args = get_args(ann)
        if len(args) == 2 and args[1] is Ellipsis:
            # Variadic tuple[X, ...]: map every element.
            return lambda v: tuple(inner(x) for x in v)

        def map_tuple(v, _i=index, _inner=inner):
            return tuple(_inner(x) if j == _i else x for j, x in enumerate(v))

        return map_tuple
    if origin is dict:
        if index == 0:
            return lambda v: {inner(k): x for k, x in v.items()}
        return lambda v: {k: inner(x) for k, x in v.items()}
    if origin is _IteratorABC:
        # Iterators are one-shot by contract: map lazily (streaming preserved).
        return lambda v: map(inner, v)
    if origin is _IterableABC:
        # Iterable implies RE-iterable: materialize, because handing a
        # one-shot map to code that iterates twice (or len()s) silently
        # yields an empty second pass.
        return lambda v: [inner(x) for x in v]
    if origin is Union:
        args = get_args(ann)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[X]: map non-None, pass None through.
            return lambda v: v if v is None else inner(v)
        raise UnsupportedSpecShape(
            f"Cannot map a role inside a non-Optional Union (in {where}: "
            f"{ann!r}): there is no reliable runtime discrimination between "
            f"union arms. Declare separate methods or use an explicit spec."
        )
    raise UnsupportedSpecShape(
        f"Cannot map role inside {origin!r} (in {where}: {ann!r}). "
        f"Supported containers: list, set, frozenset, tuple, dict, "
        f"Iterable, Iterator, Optional. "
        f"Add an explicit method override or exclude the method."
    )


@dataclass(frozen=True)
class InterfaceSpec:
    """Per-method role placement, compiled from an annotated class.

    ``source`` is typically a ``Protocol`` class whose methods are annotated
    with role TypeVars (KT, VT, ...). ``methods`` maps method name to a dict:
    ``{param_name_or_'return': [(role, path), ...]}``. You can also build an
    ``InterfaceSpec`` directly from that dict form (``from_dict``) when
    annotations are unavailable — same mechanism, no typing required.
    """

    methods: Mapping[str, Mapping[str, tuple]]
    signatures: Mapping[str, inspect.Signature]
    source: Any = None

    @classmethod
    def from_annotated(cls, source, *, roles=None):
        """Compile a spec from an annotated (Protocol) class.

        ``roles``: iterable of role names to look for; default = names of the
        TypeVars in ``source.__parameters__``, else {'KT', 'VT'}.
        """
        if roles is None:
            params = getattr(source, "__parameters__", ())
            roles = {p.__name__: p for p in params if isinstance(p, TypeVar)} or {
                "KT": None,
                "VT": None,
            }
        else:
            roles = {name: None for name in roles}
        methods = {}
        signatures = {}
        for name, func in _spec_functions(source):
            hints = _resolved_hints(func, source)
            sig = inspect.signature(func)
            # Loudness one level down (s3dol ADR-0011 D5): every named param
            # of a spec method must be annotated, or a forgotten role is
            # silent. (*args/**kwargs stay conventional passthrough.)
            for p in sig.parameters.values():
                if p.name in ("self", "cls"):
                    continue
                if p.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if p.annotation is inspect.Parameter.empty:
                    raise UnderAnnotatedSpecError(
                        f"{source.__name__}.{name}: parameter {p.name!r} has "
                        f"no annotation. Annotate it with a role TypeVar if "
                        f"it carries keys/values, or with a concrete type "
                        f"(or Any) to declare it role-free."
                    )
            sites = {}
            for pname, ann in hints.items():
                found = list(_find_role_sites(ann, roles))
                if found:
                    # Validate each path is mappable now (refuse early).
                    for site in found:
                        _transformer_for_path(
                            ann,
                            site.path,
                            _identity,
                            where=f"{source.__name__}.{name}({pname})",
                        )
                    sites[pname] = tuple(((s.role, s.path, ann) for s in found))
            methods[name] = sites
            signatures[name] = sig
        return cls(methods=methods, signatures=signatures, source=source)

    @classmethod
    def from_dict(cls, methods, *, signatures=None, source=None):
        """Build a spec from the explicit dict form.

        ``methods``: ``{method_name: {param_or_'return': [(role, path)] | role_str}}``
        where a bare role string means "the whole value has this role".
        """

        def norm_occurrence(occ):
            # Accept the user 2-tuple (role, path), the normalized 3-tuple
            # (role, path, ann) — __reduce__ round-trips the normalized form
            # back through here — and a bare role string.
            if isinstance(occ, str):
                return (occ, (), Any)
            if len(occ) == 2:
                role, path = occ
                return (role, tuple(path), Any)
            role, path, ann = occ
            return (role, tuple(path), ann)

        norm = {}
        for mname, params in methods.items():
            norm[mname] = {
                p: (
                    (norm_occurrence(v),)
                    if isinstance(v, str)
                    else tuple(norm_occurrence(occ) for occ in v)
                )
                for p, v in params.items()
            }
        return cls(methods=norm, signatures=signatures or {}, source=source)


def _spec_functions(source):
    """Yield (name, function) for the spec's declared methods (incl. dunders).

    Only the spec class's OWN plain functions count (inherited Protocol
    methods carry the base's TypeVars, whose substitution is future work).
    Members a spec cannot host refuse loudly instead of vanishing silently.
    """
    for name, member in vars(source).items():
        if name in (
            "__init__",
            "__subclasshook__",
            "__init_subclass__",
            "__class_getitem__",
        ):
            continue
        if isinstance(member, (property, staticmethod, classmethod)):
            raise UnsupportedSpecShape(
                f"{source.__name__}.{name}: {type(member).__name__} members "
                f"are not supported in interface specs (yet) — they would "
                f"be silently skipped otherwise. Remove it or use a plain "
                f"method."
            )
        if inspect.isfunction(member):
            yield name, member


def _resolved_hints(func, owner):
    """``get_type_hints`` with the owner's module globals, resolving strings."""
    module = inspect.getmodule(owner)
    globalns = getattr(module, "__dict__", {})
    return get_type_hints(func, globalns=globalns)


# ---------------------------------------------------------------------------
# The flat codec stack


def _fused_role_funcs(stack, *, direction):
    """Fuse a stack of ``{role: Codec}`` layers into ``{role: callable}``.

    ``stack`` is ordered innermost-first (append order: the first layer
    applied to the leaf is index 0). Encoders run outer->inner (reversed
    stack order); decoders run inner->outer (stack order).
    """
    roles = set()
    for layer in stack:
        roles.update(layer)
    out = {}
    for role in roles:
        if direction == "encode":
            funcs = [layer[role].encoder for layer in reversed(stack) if role in layer]
        else:
            funcs = [layer[role].decoder for layer in stack if role in layer]
        out[role] = _fuse(funcs)
    return out


def _validate_stack_seams(stack):
    """Validate typed-codec seams between adjacent layers (per role).

    Layers are innermost-first. For a role present in layers i < j (adjacent
    among the layers that carry that role), the OUTER layer's leafward face
    (``encoded_type``) meets the INNER layer's outer face (``decoded_type``).
    When both are declared and differ, refuse loudly; ``None`` = untagged =
    unchecked (design decision 2026-08-10, question 3).
    """
    roles = {role for layer in stack for role in layer}
    for role in roles:
        carriers = [(i, layer[role]) for i, layer in enumerate(stack) if role in layer]
        for (i, inner_c), (j, outer_c) in zip(carriers, carriers[1:]):
            inner_face = inner_c.decoded_type
            outer_face = outer_c.encoded_type
            if (
                inner_face is not None
                and outer_face is not None
                and inner_face is not outer_face
            ):
                raise InterfaceWrapError(
                    f"Typed-codec seam mismatch for role {role!r}: layer {i} "
                    f"decodes to {inner_face.__name__} but layer {j} encodes "
                    f"to {outer_face.__name__}. Adjacent codec layers must "
                    f"agree at their seam (outer encoded_type == inner "
                    f"decoded_type)."
                )


# ---------------------------------------------------------------------------
# Method-plan compilation


def _compile_method_plan(name, sites, leaf_method, sig, encoders, decoders):
    """Compile one boundary method: encode role args, call leaf, decode result.

    ``sites``: {param_name_or_'return': ((role, path, ann), ...)}.
    Returns a callable(*args, **kwargs) with the leaf method baked in.
    """
    # Build per-parameter transformers (outer -> inner). Integer site keys
    # mean positional index (dict-form specs), resolved against the outer
    # signature when one exists; when none does (builtin slots like
    # dict.__getitem__ have no text signature on 3.10), compile a purely
    # positional plan instead.
    positional_only_plan = False
    if any(isinstance(p, int) for p in sites):
        if sig is None:
            positional_only_plan = True
        else:
            param_names_by_index = list(sig.parameters)
            sites = {
                (param_names_by_index[p] if isinstance(p, int) else p): v
                for p, v in sites.items()
            }

    param_transforms = {}  # pname -> callable
    return_transform = None
    for pname, occurrences in sites.items():
        funcs = []
        for role, path, ann in occurrences:
            role_func = (encoders if pname != "return" else decoders).get(
                role, _identity
            )
            if role_func is _identity:
                continue
            funcs.append(_transformer_for_path(ann, path, role_func, where=name))
        if not funcs:
            continue
        fused = _fuse(funcs)
        if pname == "return":
            return_transform = fused
            continue
        param = sig.parameters.get(pname) if sig is not None else None
        if param is not None:
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                raise UnsupportedSpecShape(
                    f"{name}: role on a **kwargs parameter ({pname!r}) is "
                    f"not supported — keyword names as keys have no "
                    f"annotation channel."
                )
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                # bound.arguments holds a TUPLE for *args: map elementwise
                # (a bare-role transform applied to the tuple itself would
                # silently corrupt).
                fused = (lambda f: lambda tup: tuple(f(x) for x in tup))(fused)
            if param.default is None:
                # A None default lives in the LEAF's domain and, on 3.10,
                # get_type_hints implicitly wraps `x: KT = None` in Optional.
                # Normalize both: never transform None.
                fused = (lambda f: lambda v: v if v is None else f(v))(fused)
        param_transforms[pname] = fused

    if not param_transforms and return_transform is None:
        # Spec'd but no active roles in this stack: plain passthrough.
        return leaf_method

    if positional_only_plan:
        # No signature to resolve against (3.10 builtin slots): transform by
        # positional index; keyword calls of role'd params are refused loudly.
        idx_transforms = {
            p: t for p, t in param_transforms.items() if isinstance(p, int)
        }

        def plan(*args, **kwargs):
            args = tuple(
                idx_transforms[i](a) if i in idx_transforms else a
                for i, a in enumerate(args)
            )
            if len(args) <= max(idx_transforms, default=-1):
                raise TypeError(
                    f"{name}: role-bearing positional argument(s) "
                    f"{sorted(idx_transforms)} must be passed positionally "
                    f"(no signature is available to resolve keyword calls)."
                )
            result = leaf_method(*args, **kwargs)
            if return_transform is not None:
                result = return_transform(result)
            return result

        return plan

    # Fast path: single role'd parameter, first positional slot with no
    # default. Keyword calls of that param are handled by name (POSITIONAL_
    # OR_KEYWORD contracts include them), without a Signature.bind per call.
    param_names = list(sig.parameters) if sig is not None else []
    _first = sig.parameters[param_names[0]] if param_names else None
    if (
        sig is not None
        and param_names
        and set(param_transforms) <= {param_names[0]}
        and _first.default is inspect.Parameter.empty
        and _first.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ):
        first_transform = param_transforms.get(param_names[0])
        first_name = param_names[0]

        if first_transform is None:

            def plan(*args, **kwargs):
                return return_transform(leaf_method(*args, **kwargs))

        else:

            def plan(*args, **kwargs):
                if args:
                    args = (first_transform(args[0]),) + args[1:]
                elif first_name in kwargs:
                    # Re-emit positionally: the caller used the SPEC's name,
                    # which the leaf's own parameter may not share.
                    kwargs = dict(kwargs)
                    args = (first_transform(kwargs.pop(first_name)),)
                result = leaf_method(*args, **kwargs)
                if return_transform is not None:
                    result = return_transform(result)
                return result

        return plan

    if sig is None:
        raise UnsupportedSpecShape(
            f"Method {name!r} has role-bearing named parameters but no "
            f"inspectable signature to bind against."
        )

    def plan(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        for pname, transform in param_transforms.items():
            if pname in bound.arguments:
                bound.arguments[pname] = transform(bound.arguments[pname])
        result = leaf_method(*bound.args, **bound.kwargs)
        if return_transform is not None:
            result = return_transform(result)
        return result

    return plan


def _outer_signature(leaf, name, spec):
    """The signature calls are bound against — the OUTER contract.

    The spec's signature is authoritative when it exists: the caller talks to
    the interface the spec declares, and the leaf's own parameter *names* are
    irrelevant (dict names its key ``key``; the spec may say ``k``). Calls are
    re-emitted positionally/keyword exactly as bound, so the leaf receives
    them as the caller sent them. Falls back to the leaf's signature for
    dict-form specs without signatures.
    """
    sig = spec.signatures.get(name)
    if sig is None:
        func = getattr(type(leaf), name, None) or getattr(leaf, name, None)
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return None
    params = list(sig.parameters.values())
    if params and params[0].name in ("self", "cls"):
        params = params[1:]
        sig = sig.replace(parameters=params)
    return sig


# ---------------------------------------------------------------------------
# The proxy


class InterfaceProxy:
    """Base class for generated flat proxies. Instances hold the whole wrap.

    State (all under proxy-private names, wrapt's ``_self_`` lesson):
    ``_self_leaf`` (strong ref, the innermost object — also ``__wrapped__``),
    ``_self_spec``, ``_self_stack`` (tuple of {role: Codec}, innermost-first),
    ``_self_plans`` (compiled boundary callables), plus the policy fields.
    """

    _self_passthrough = frozenset()

    def __init__(self, *args, **kwargs):
        # Two construction modes: wrapping an existing instance (internal,
        # via interface_wrap) or constructing the leaf (class-wrap mode).
        raise TypeError("InterfaceProxy subclasses are built via interface_wrap(...)")

    @property
    def __wrapped__(self):
        return self._self_leaf

    def _encode_role(self, role, value):
        """Map an outer-domain value of ``role`` to the leaf domain (total)."""
        return self._self_encoders.get(role, _identity)(value)

    def _decode_role(self, role, value):
        """Map a leaf-domain value of ``role`` outward (the inverse walk)."""
        return self._self_decoders.get(role, _identity)(value)

    def __repr__(self):
        return (
            f"<{type(self).__name__} of {self._self_leaf!r} "
            f"with {len(self._self_stack)} codec layer(s)>"
        )

    def __reduce__(self):
        spec_ref = self._self_spec.source or dict(self._self_spec.methods)
        return (
            _rebuild_interface_wrap,
            (
                self._self_leaf,
                spec_ref,
                tuple(self._self_stack),
                self._self_undeclared,
                tuple(sorted(self._self_passthrough)),
            ),
        )

    def __getattr__(self, name):
        # Only reached when normal lookup fails: plans and passthroughs first.
        if name.startswith("__") and name.endswith("__"):
            # Explicit dunder access must NOT escape to the leaf: forwarding
            # would hand out raw leaf-bound methods (s.__contains__('outer_k')
            # silently answering in the wrong key domain). Plain
            # AttributeError keeps hasattr-style duck typing honest.
            raise AttributeError(name)
        if name.startswith("_"):
            return getattr(object.__getattribute__(self, "_self_leaf"), name)
        if name in object.__getattribute__(self, "_self_passthrough"):
            return getattr(object.__getattribute__(self, "_self_leaf"), name)
        raise UndeclaredAttributeError(
            f"{name!r} is not in the interface spec of this wrap. "
            f"Add it to the spec, or pass passthrough={{'{name}'}} to "
            f"interface_wrap to forward it verbatim (unmapped keys/values!)."
        )


def _rebuild_interface_wrap(leaf, spec_ref, stack, undeclared, passthrough):
    """Pickle reconstructor: recompile the wrap from values (no dynamic class)."""
    return interface_wrap(
        leaf,
        spec=spec_ref,
        _stack=stack,
        undeclared=undeclared,
        passthrough=passthrough,
    )


_DUNDER_METHOD_TEMPLATE = """
def {name}(self, *args, **kwargs):
    return self._self_plans[{name!r}](*args, **kwargs)
"""

_proxy_class_cache = {}


def _build_proxy_class(leaf_type, spec, method_names, class_name=None):
    """Generate (and cache) the proxy class for (leaf_type, spec, surface).

    The class namespace holds one dispatching method per spec'd method the
    leaf actually has — dunders included, so implicit special-method lookup
    works. Capability mirroring: a method the leaf lacks is NOT given to the
    proxy class.
    """
    # Cache only source-backed specs: the source class is a stable, hashable
    # key the cache holds strongly. Dict-form specs (no source) build a fresh
    # class — caching them by id() risks collisions after GC id-reuse and
    # unbounded growth otherwise.
    key = None
    if spec.source is not None:
        key = (leaf_type, spec.source, tuple(method_names))
        cached = _proxy_class_cache.get(key)
        if cached is not None:
            return cached
    ns = {}
    for name in method_names:
        exec(_DUNDER_METHOD_TEMPLATE.format(name=name), {}, ns)
    if "__getitem__" in ns and "__iter__" not in ns:
        # Without this, Python's legacy sequence protocol would invent
        # iteration from __getitem__(0), __getitem__(1), ... — feeding int
        # keys through the key encoder, silently. Louder to refuse.
        def __iter__(self):
            raise TypeError(
                f"{type(self).__name__} is not iterable: __iter__ is not in "
                f"its interface spec (and sequence-protocol fallback over "
                f"__getitem__ would silently feed integer keys through the "
                f"key codec)."
            )

        ns["__iter__"] = __iter__
    if "__getitem__" in method_names and "__iter__" in method_names:
        # Design decision (2026-08-10, question 4): when the spec'd surface
        # supports Mapping-style traversal, equality compares OUTER views —
        # a wrapped store equals a dict holding its outer items. Defining
        # __eq__ sets __hash__ to None (mutable-mapping convention), which is
        # the decided no-hash policy.
        def __eq__(self, other):
            if other is self:
                return True
            try:
                other_items = {k: other[k] for k in other}
            except (TypeError, KeyError):
                return NotImplemented
            return {k: self[k] for k in self} == other_items

        ns["__eq__"] = __eq__
    ns["__module__"] = __name__
    cls_name = class_name or f"{leaf_type.__name__}InterfaceProxy"
    cls = type(cls_name, (InterfaceProxy,), ns)
    if key is not None:
        _proxy_class_cache[key] = cls
    return cls


# ---------------------------------------------------------------------------
# Public entry point


def interface_wrap(
    obj,
    *,
    spec,
    codecs: Optional[Mapping[str, Codec]] = None,
    undeclared: str = "exclude",
    passthrough: _IterableABC = (),
    _stack: Optional[tuple] = None,
):
    """Wrap ``obj`` (an instance) with boundary codecs per an interface spec.

    :param obj: the object to wrap, or an existing ``InterfaceProxy`` (in
        which case the new codec layer extends the flat stack over the SAME
        leaf — wrapping never nests).
    :param spec: an annotated (Protocol) class, an ``InterfaceSpec``, or the
        explicit dict form accepted by ``InterfaceSpec.from_dict``.
    :param codecs: one codec layer: ``{role_name: Codec(encoder, decoder)}``.
    :param undeclared: policy for public leaf attributes absent from the
        spec. Default ``'exclude'`` (design decision 2026-08-10, question 2):
        the wrap succeeds, and every USE of an undeclared attribute raises
        ``UndeclaredAttributeError`` with guidance — refusal at the moment of
        danger, with no habit-forming escape. ``'raise'`` is the strict mode
        (refuse at wrap time, listing names); ``'passthrough'`` forwards
        verbatim (explicitly unmapped keys/values).
    :param passthrough: explicit names to forward verbatim regardless.
    """
    if undeclared not in ("raise", "passthrough", "exclude"):
        raise ValueError(
            f"undeclared must be 'raise', 'passthrough' or 'exclude', "
            f"got {undeclared!r}"
        )
    # --- normalize the spec
    if isinstance(spec, InterfaceSpec):
        spec_obj = spec
    elif isinstance(spec, type):
        spec_obj = InterfaceSpec.from_annotated(spec)
    elif isinstance(spec, Mapping):
        spec_obj = InterfaceSpec.from_dict(spec)
    else:
        raise TypeError(f"Cannot interpret spec: {spec!r}")

    # --- normalize the stack
    if isinstance(obj, InterfaceProxy):
        leaf = obj._self_leaf
        base_stack = tuple(obj._self_stack)
        passthrough = frozenset(passthrough) | obj._self_passthrough
    else:
        leaf = obj
        base_stack = tuple(_stack or ())
    if isinstance(leaf, type):
        raise TypeError(
            "interface_wrap wraps instances in this prototype; "
            "class-wrapping is future work (see the design doc)."
        )
    # Mixed-architecture stacks: flat-model guarantees (encode/decode
    # totality, __wrapped__ = raw backend, pickle uniformity) are scoped to
    # pure interface_wrap stacks. Wrapping a legacy dol Store is allowed but
    # the Store chain below is opaque to us — say so.
    try:
        from dol.base import Store as _LegacyStore

        if isinstance(leaf, _LegacyStore):
            import warnings

            warnings.warn(
                "interface_wrap over a legacy dol Store: the Store (and its "
                ".store chain) is treated as an opaque leaf — __wrapped__ "
                "is the Store, not the raw backend, and flat-stack "
                "guarantees apply only to the layers above it.",
                stacklevel=2,
            )
    except ImportError:  # pragma: no cover - dol.base always importable here
        pass
    stack = base_stack + ((dict(codecs),) if codecs else ())

    # --- role sanity (loudness): every codec role must appear in the spec
    used_roles = {
        role
        for sites in spec_obj.methods.values()
        for occurrences in sites.values()
        for role, _path, _ann in occurrences
    }
    for layer in stack:
        unknown = set(layer) - used_roles
        if unknown:
            raise InterfaceWrapError(
                f"Codec layer names role(s) {sorted(unknown)} that occur "
                f"nowhere in the spec (spec roles: {sorted(used_roles)}). "
                f"A codec that can never apply is almost certainly a mistake."
            )
    _validate_stack_seams(stack)

    # --- undeclared-surface policy (wrap time, loud by default)
    passthrough = frozenset(passthrough)
    public_attrs = {n for n in dir(leaf) if not n.startswith("_")}
    undeclared_names = public_attrs - set(spec_obj.methods) - passthrough
    if undeclared_names and undeclared == "raise":
        raise UndeclaredAttributeError(
            f"The leaf exposes public attributes not covered by the spec: "
            f"{sorted(undeclared_names)}. Methods among these would receive "
            f"UNMAPPED keys/values through this wrap (the #83 bug class). "
            f"Add them to the spec, or pass undeclared='passthrough' / "
            f"'exclude', or list them in passthrough=... explicitly."
        )
    if undeclared == "passthrough":
        passthrough = passthrough | undeclared_names

    # --- compile
    encoders = _fused_role_funcs(stack, direction="encode")
    decoders = _fused_role_funcs(stack, direction="decode")
    present = [name for name in spec_obj.methods if hasattr(leaf, name)]
    plans = {}
    for name in present:
        sig = _outer_signature(leaf, name, spec_obj)
        plans[name] = _compile_method_plan(
            name,
            spec_obj.methods[name],
            getattr(leaf, name),
            sig,
            encoders,
            decoders,
        )

    cls = _build_proxy_class(type(leaf), spec_obj, tuple(present))
    proxy = object.__new__(cls)
    object.__setattr__(proxy, "_self_leaf", leaf)
    object.__setattr__(proxy, "_self_spec", spec_obj)
    object.__setattr__(proxy, "_self_stack", stack)
    object.__setattr__(proxy, "_self_plans", plans)
    object.__setattr__(proxy, "_self_encoders", encoders)
    object.__setattr__(proxy, "_self_decoders", decoders)
    object.__setattr__(proxy, "_self_undeclared", undeclared)
    object.__setattr__(proxy, "_self_passthrough", passthrough)
    return proxy


# ---------------------------------------------------------------------------
# Built-in Mapping spec and the wrap_kvs-shaped facade
# (design decision 2026-08-10, question 1: simple things stay simple)


KT = TypeVar("KT")
VT = TypeVar("VT")


class MappingInterface(Protocol[KT, VT]):
    """The built-in Mapping-shaped spec: the six methods dol's Store routes.

    The MutableMapping mixin surface (``get``, ``keys``, ``items``,
    ``update``, ...) is deliberately NOT declared: each of those needs its
    own vocabulary decision (``KeysView[KT]``, ``update(**kw)``, ``get``'s
    default), so under the default ``undeclared='exclude'`` policy they are
    hidden-and-loud rather than silently unmapped.
    """

    def __getitem__(self, k: KT) -> VT: ...

    def __setitem__(self, k: KT, v: VT) -> None: ...

    def __delitem__(self, k: KT) -> None: ...

    def __iter__(self) -> Iterator[KT]: ...

    def __len__(self) -> int: ...

    def __contains__(self, k: KT) -> bool: ...


def _as_codec(codec_or_pair):
    """Coerce anything with .encoder/.decoder (or a pair) to our Codec."""
    if isinstance(codec_or_pair, Codec):
        return codec_or_pair
    if hasattr(codec_or_pair, "encoder") and hasattr(codec_or_pair, "decoder"):
        return Codec(
            encoder=codec_or_pair.encoder,
            decoder=codec_or_pair.decoder,
            decoded_type=getattr(codec_or_pair, "decoded_type", None),
            encoded_type=getattr(codec_or_pair, "encoded_type", None),
        )
    encoder, decoder = codec_or_pair
    return Codec(encoder=encoder, decoder=decoder)


def kv_interface_wrap(
    store,
    *,
    obj_of_data: Optional[Callable] = None,
    data_of_obj: Optional[Callable] = None,
    key_of_id: Optional[Callable] = None,
    id_of_key: Optional[Callable] = None,
    key_codec=None,
    value_codec=None,
    undeclared: str = "exclude",
    passthrough: _IterableABC = (),
):
    """``wrap_kvs``-shaped kwargs facade over the interface engine.

    Same transform-naming conventions as ``dol.wrap_kvs`` (``X_of_Y``:
    ``id_of_key`` encodes keys going in, ``key_of_id`` decodes keys coming
    out, ``data_of_obj`` encodes values going in, ``obj_of_data`` decodes
    values coming out), compiled onto a flat proxy with the built-in
    ``MappingInterface`` spec.

    Unlike ``wrap_kvs``, transforms here are plain unary callables — there is
    no self-convention (``f(self, x)``) inference, because there are no
    wrapper layers for a transform to receive.

    >>> import json
    >>> s = kv_interface_wrap({}, data_of_obj=json.dumps, obj_of_data=json.loads)
    >>> s['a'] = {'x': 1}
    >>> s['a']
    {'x': 1}
    >>> s.__wrapped__
    {'a': '{"x": 1}'}
    """
    if key_codec is not None and (key_of_id is not None or id_of_key is not None):
        raise ValueError("Pass key_codec OR key_of_id/id_of_key, not both.")
    if value_codec is not None and (obj_of_data is not None or data_of_obj is not None):
        raise ValueError("Pass value_codec OR obj_of_data/data_of_obj, not both.")
    layer = {}
    if key_codec is not None:
        layer["KT"] = _as_codec(key_codec)
    elif key_of_id is not None or id_of_key is not None:
        layer["KT"] = Codec(
            encoder=id_of_key or _identity, decoder=key_of_id or _identity
        )
    if value_codec is not None:
        layer["VT"] = _as_codec(value_codec)
    elif obj_of_data is not None or data_of_obj is not None:
        layer["VT"] = Codec(
            encoder=data_of_obj or _identity, decoder=obj_of_data or _identity
        )
    return interface_wrap(
        store,
        spec=MappingInterface,
        codecs=layer or None,
        undeclared=undeclared,
        passthrough=passthrough,
    )
