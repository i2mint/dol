# Command: add-codec

Add a codec (encoder/decoder pair) to an existing store or create a new codec for `ValueCodecs`/`KeyCodecs`.

## Usage

```
/add-codec <description of what to encode/decode>
```

## Examples

```
/add-codec add msgpack serialization to my store
/add-codec add base64 encoding to keys
/add-codec create a Zod-validated value codec for UserSchema
```

## What this command does

1. Reads `dol/kv_codecs.py` to understand existing codecs and patterns
2. Reads `dol/trans.py` for `Codec`, `ValueCodec`, `KeyCodec` definitions
3. Implements the codec as:
   - A `ValueCodec(encoder=..., decoder=...)` or `KeyCodec(encoder=..., decoder=...)` if it's a reusable pair
   - A `wrap_kvs(store, obj_of_data=..., data_of_obj=...)` if it's one-off
4. Shows how to apply it to a store and how to compose it with existing codecs

## Output pattern

```python
from dol.trans import ValueCodec
import msgpack

msgpack_codec = ValueCodec(
    encoder=msgpack.dumps,
    decoder=msgpack.loads,
)

# Apply to any store
MsgpackStore = msgpack_codec(dict)

# Compose: msgpack + gzip
from dol import ValueCodecs
compressed_msgpack = msgpack_codec + ValueCodecs.gzip()
```

## Notes

- Use `Codec.compose_with` / `+` operator to chain codecs
- See `dol/kv_codecs.py` for real examples of `ValueCodecs.*` factories
- See `misc/docs/python_design.md` section "Codec Abstraction" for details
