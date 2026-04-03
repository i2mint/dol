# Command: explain-store

Explain how a given dol store works, tracing the key/value transform pipeline.

## Usage

```
/explain-store <store_expression_or_class_name>
```

## What this command does

1. Finds the store definition in the codebase
2. Traces the MRO (method resolution order) to identify all transform layers
3. For each `wrap_kvs` layer, shows what transforms are applied
4. Draws the data flow diagram for read and write operations
5. Shows a concrete example: what happens when you do `store['some_key']` and `store['some_key'] = value`

## Example output format

```
Store: JsonFileStore (wrap_kvs applied to Files)

Read pipeline:
  key='report'
  → id_of_key: 'report' + '.json' = 'report.json'
  → Files.__getitem__('report.json')
  → raw_data: b'{"x": 1}'
  → obj_of_data: json.loads(raw_data)
  → returns: {'x': 1}

Write pipeline:
  key='report', obj={'x': 1}
  → id_of_key: 'report.json'
  → data_of_obj: json.dumps({'x': 1}) = '{"x": 1}'
  → Files.__setitem__('report.json', '{"x": 1}')

Iteration:
  Files.__iter__() → ['report.json', 'data.json', ...]
  → key_of_id: strip '.json' → ['report', 'data', ...]
```

## Notes

- See `dol/base.py:Store.__getitem__` for the actual implementation
- See `misc/docs/dol_design.md` for the full class hierarchy
