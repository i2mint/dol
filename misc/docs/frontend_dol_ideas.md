# Frontend dol Ideas: Toward `zoddal`

This document explores how the design principles of `dol` translate to the frontend (TypeScript/React) ecosystem, and proposes the architecture for `zoddal` — a **ZOD-Data-Access-Layer** for frontend applications.

For background on dol's general design, see [general_design.md](general_design.md).

---

## The Core Analogy

Python's `dol` is built around two key ideas:
1. **A minimal, language-native KV interface** (`Mapping`/`MutableMapping` ≈ Python's `dict`)
2. **Composable transform layers** that adapt any backend to that interface

The frontend has an analogous "language-native" KV concept: JavaScript's `Map<K, V>` and the object-access pattern `record[key]`. But unlike Python, the frontend also has a second critical concern: **asynchrony** — all real storage operations are `async`.

The zoddal design builds on this: a **type-safe, async KV interface for the frontend**, with the same composable transform philosophy as dol.

---

## Two Layers of Frontend Affordances

Frontend applications have two distinct "collection affordance" layers that need to be bridged:

### Layer 1: Storage Layer (Backend-Facing)

Talking to REST APIs, IndexedDB, localStorage, cloud storage:
- `GET /users/42` → fetch a user by ID
- `POST /users` → create a new user
- `PUT /users/42` → update
- `DELETE /users/42` → delete
- `GET /users?filter=...` → list with query

This maps naturally to a KV interface:
```typescript
interface KvStore<K, V> {
  get(key: K): Promise<V>;
  set(key: K, value: V): Promise<void>;
  delete(key: K): Promise<void>;
  keys(): AsyncIterable<K>;  // or Promise<K[]>
  has(key: K): Promise<boolean>;
}
```

### Layer 2: UI Layer (Component-Facing)

Presenting collections in the UI: tables, grids, forms, CRUD dialogs. This is what `zod-collection-ui` addresses with its `defineCollection` / `DataProvider` interface.

The `DataProvider` interface (from zod-collection-ui):
```typescript
interface DataProvider<T> {
  getList(params: { sort?, filter?, search?, pagination? }): Promise<{ data: T[]; total: number }>;
  getOne(id: string): Promise<T>;
  create(data: Partial<T>): Promise<T>;
  update(id: string, data: Partial<T>): Promise<T>;
  delete(id: string): Promise<void>;
}
```

### The Gap: Bridging Storage and UI

Currently these two layers are often implemented independently, creating duplication. `zoddal` aims to be **DRY across both**: a `KvStore` at the storage layer can be adapted to a `DataProvider` at the UI layer through a standard bridge.

---

## The zoddal Architecture

### Core Interfaces

```typescript
// Read-only key-value store (analog to dol's KvReader)
interface KvReader<K, V> {
  get(key: K): Promise<V | undefined>;
  has(key: K): Promise<boolean>;
  keys(): Promise<K[]>;  // or AsyncIterable<K>
  values(): Promise<V[]>;
  entries(): Promise<[K, V][]>;
  head(): Promise<[K, V] | undefined>;
}

// Read-write store (analog to dol's KvPersister)
interface KvStore<K, V> extends KvReader<K, V> {
  set(key: K, value: V): Promise<void>;
  delete(key: K): Promise<void>;
}

// Mutable store with update semantics (for REST PATCH-style updates)
interface MutableKvStore<K, V> extends KvStore<K, V> {
  update(key: K, patch: Partial<V>): Promise<V>;
}
```

### The Transform Pipeline

Mirroring dol's `wrap_kvs`, zoddal provides `wrapKvs`:

```typescript
function wrapKvs<K1, V1, K2, V2>(
  store: KvStore<K1, V1>,
  transforms: {
    // Key transforms
    keyOfId?: (id: K1) => K2;     // outgoing: storage key → interface key
    idOfKey?: (key: K2) => K1;    // incoming: interface key → storage key
    // Value transforms
    objOfData?: (data: V1) => V2;  // outgoing: raw data → domain object
    dataOfObj?: (obj: V2) => V1;   // incoming: domain object → raw data
    // Key-conditioned transforms (analog to preset/postget)
    postget?: (key: K2, data: V1) => V2;
    preset?: (key: K2, obj: V2) => V1;
  }
): KvStore<K2, V2>
```

**Example**: Adapting a REST API to a typed KV store:

```typescript
const rawApiStore: KvStore<string, unknown> = restAdapter('/api/users');

const userStore = wrapKvs(rawApiStore, {
  // Keys: external IDs stay as strings
  // Values: validate with Zod schema
  objOfData: (raw) => UserSchema.parse(raw),
  dataOfObj: (user) => UserSchema.partial().parse(user),
});
```

---

## Codec Pattern in TypeScript

The `Codec` pattern from dol maps cleanly:

```typescript
interface Codec<Decoded, Encoded> {
  encode: (decoded: Decoded) => Encoded;
  decode: (encoded: Encoded) => Decoded;
}

// Compose codecs (like dol's Codec + operator)
function composeCodecs<A, B, C>(
  first: Codec<A, B>,
  second: Codec<B, C>
): Codec<A, C> {
  return {
    encode: (a) => second.encode(first.encode(a)),
    decode: (c) => first.decode(second.decode(c)),
  };
}

// ValueCodec: wraps a store with a value codec
class ValueCodec<Decoded, Encoded> implements Codec<Decoded, Encoded> {
  constructor(public encode: (d: Decoded) => Encoded,
              public decode: (e: Encoded) => Decoded) {}

  wrap<K>(store: KvStore<K, Encoded>): KvStore<K, Decoded> {
    return wrapKvs(store, { objOfData: this.decode, dataOfObj: this.encode });
  }
}

// KeyCodec: wraps a store with a key codec
class KeyCodec<OuterKey, InnerKey> implements Codec<OuterKey, InnerKey> {
  constructor(public encode: (k: OuterKey) => InnerKey,
              public decode: (id: InnerKey) => OuterKey) {}

  wrap<V>(store: KvStore<InnerKey, V>): KvStore<OuterKey, V> {
    return wrapKvs(store, { idOfKey: this.encode, keyOfId: this.decode });
  }
}
```

**Ready-made codecs:**
```typescript
const Codecs = {
  json: new ValueCodec<unknown, string>(JSON.stringify, JSON.parse),
  zodValidated: <T>(schema: z.ZodType<T>) =>
    new ValueCodec<T, unknown>(
      (v) => v,                   // no encoding on write (validated upstream)
      (raw) => schema.parse(raw)  // decode = validate + parse
    ),
  urlEncoded: new KeyCodec<string, string>(encodeURIComponent, decodeURIComponent),
  pathPrefixed: (prefix: string) =>
    new KeyCodec<string, string>(
      (k) => `${prefix}/${k}`,
      (id) => id.slice(prefix.length + 1),
    ),
};
```

---

## Built-In Adapters

### In-Memory Adapter (for testing)

```typescript
function memStore<K, V>(initial?: Map<K, V>): KvStore<K, V> {
  const map = initial ?? new Map<K, V>();
  return {
    get: async (k) => map.get(k),
    set: async (k, v) => { map.set(k, v); },
    delete: async (k) => { map.delete(k); },
    has: async (k) => map.has(k),
    keys: async () => [...map.keys()],
    values: async () => [...map.values()],
    entries: async () => [...map.entries()],
    head: async () => map.size > 0 ? [map.keys().next().value, map.values().next().value] : undefined,
  };
}
```

### REST Adapter

```typescript
function restAdapter<T>(baseUrl: string): MutableKvStore<string, T> {
  return {
    get: async (id) => {
      const res = await fetch(`${baseUrl}/${id}`);
      if (res.status === 404) return undefined;
      return res.json();
    },
    set: async (id, value) => {
      await fetch(`${baseUrl}/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(value),
      });
    },
    delete: async (id) => {
      await fetch(`${baseUrl}/${id}`, { method: 'DELETE' });
    },
    has: async (id) => {
      const res = await fetch(`${baseUrl}/${id}`, { method: 'HEAD' });
      return res.ok;
    },
    keys: async () => {
      const res = await fetch(baseUrl);
      const items: T[] = await res.json();
      return items.map((item: any) => item.id);
    },
    values: async () => {
      const res = await fetch(baseUrl);
      return res.json();
    },
    entries: async () => {
      const res = await fetch(baseUrl);
      const items: T[] = await res.json();
      return items.map((item: any) => [item.id, item] as [string, T]);
    },
    head: async () => {
      const res = await fetch(`${baseUrl}?limit=1`);
      const items: T[] = await res.json();
      if (!items.length) return undefined;
      const item = items[0] as any;
      return [item.id, item] as [string, T];
    },
    update: async (id, patch) => {
      const res = await fetch(`${baseUrl}/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      });
      return res.json();
    },
  };
}
```

### IndexedDB Adapter

```typescript
function indexedDbStore<T>(dbName: string, storeName: string): KvStore<string, T> {
  // ... opens IDB connection, implements get/set/delete/keys via IDB transactions
}
```

---

## Bridge to `zod-collection-ui`'s `DataProvider`

The `DataProvider` interface (used by `zod-collection-ui`) can be derived from a `KvStore` + a Zod schema:

```typescript
function kvStoreToDataProvider<T extends { id: string }>(
  store: MutableKvStore<string, T>,
  schema: z.ZodType<T>,
): DataProvider<T> {
  return {
    getList: async ({ sort, filter, search, pagination }) => {
      const entries = await store.entries();
      let items = entries.map(([, v]) => v);

      // Apply filtering, sorting, search in-memory
      // (or delegate to store if it supports server-side queries)
      if (filter) items = applyFilter(items, filter);
      if (search) items = applySearch(items, search);
      if (sort) items = applySort(items, sort);

      const total = items.length;
      if (pagination) {
        const { page, pageSize } = pagination;
        items = items.slice((page - 1) * pageSize, page * pageSize);
      }
      return { data: items, total };
    },
    getOne: async (id) => {
      const item = await store.get(id);
      if (!item) throw new Error(`Not found: ${id}`);
      return item;
    },
    create: async (data) => {
      const item = schema.parse({ ...data, id: crypto.randomUUID() });
      await store.set(item.id, item);
      return item;
    },
    update: async (id, data) => {
      const existing = await store.get(id);
      if (!existing) throw new Error(`Not found: ${id}`);
      const updated = schema.parse({ ...existing, ...data });
      await store.set(id, updated);
      return updated;
    },
    delete: async (id) => {
      await store.delete(id);
    },
  };
}
```

This bridge means: **define your storage once as a `KvStore`, get a `DataProvider` for free**.

---

## Zod as the Interface Definition Language

In Python, `dol` relies on type hints and `collections.abc` protocols to define interfaces. In the frontend, **Zod schemas** play this role:

- Zod schemas describe the *shape* of domain objects (what Python type hints do)
- Zod `.meta()` annotations describe *affordances* (what Python docstrings/comments do)
- `defineCollection(schema)` derives a full UI + storage configuration (analogous to `wrap_kvs` deriving a full store from transforms)

```typescript
// Python dol analogy:
//   schema   ≈ Python class with type hints
//   .meta()  ≈ docstring with behavioral hints
//   wrapKvs  ≈ wrap_kvs

const UserSchema = z.object({
  id: z.string().uuid().meta({ editable: false }),
  name: z.string().meta({ sortable: true, searchable: true }),
  email: z.string().email().meta({ filterable: 'exact' }),
  role: z.enum(['admin', 'user']).meta({ filterable: 'select', groupable: true }),
});

// Storage layer: derive a KV store
const userStore = wrapKvs(
  restAdapter('/api/users'),
  { objOfData: (raw) => UserSchema.parse(raw) }
);

// UI layer: derive a collection definition
const userCollection = defineCollection({
  schema: UserSchema,
  store: userStore,   // zoddal bridge: store IS the data provider
});
```

---

## The DRY Principle Across Layers

The key insight is that both the storage layer and the UI layer need the same fundamental operations — list, get, set, delete — just with different ergonomics:

| Operation | Storage KV | UI DataProvider |
|-----------|-----------|----------------|
| List all  | `store.keys()` | `getList({ pagination })` |
| Get one   | `store.get(id)` | `getOne(id)` |
| Create    | `store.set(id, value)` | `create(data)` |
| Update    | `store.set(id, {...existing, ...patch})` | `update(id, patch)` |
| Delete    | `store.delete(id)` | `delete(id)` |

The differences (pagination, sorting, filtering at the UI level; raw bytes/URLs at the storage level) are handled by the transform layers and the bridge function — **not by duplicating the core CRUD logic**.

---

## Capability Discovery

Mirroring the `CollectionCapabilities` concept from the Cosmograph resource design:

```typescript
interface StoreCapabilities {
  canCreate: boolean;
  canUpdate: boolean;
  canDelete: boolean;
  canList: boolean;
  supportsServerSideFilter: boolean;
  supportsServerSideSort: boolean;
  supportsServerSidePagination: boolean;
  maxPageSize?: number;
}

// Store can declare its capabilities
interface CapableKvStore<K, V> extends KvStore<K, V> {
  capabilities(): Promise<StoreCapabilities>;
}
```

The `kvStoreToDataProvider` bridge reads capabilities to decide whether to do server-side or client-side filtering/sorting.

---

## Proposed `zoddal` Package Architecture

```
zoddal/
├── core/
│   ├── types.ts           KvReader, KvStore, MutableKvStore, Codec, StoreCapabilities
│   ├── wrap.ts            wrapKvs(), ValueCodec, KeyCodec, composeCodecs()
│   └── codecs.ts          Codecs.json, Codecs.zodValidated, Codecs.urlEncoded, etc.
│
├── adapters/
│   ├── memory.ts          memStore() — in-memory, for testing
│   ├── rest.ts            restAdapter() — REST/HTTP with fetch
│   ├── indexeddb.ts       indexedDbStore() — browser IndexedDB
│   └── localStorage.ts    localStorageStore() — simple key-value
│
├── bridge/
│   └── dataProvider.ts    kvStoreToDataProvider() — bridge to zod-collection-ui
│
└── index.ts               public exports
```

**Usage at a glance:**

```typescript
import { restAdapter, wrapKvs, Codecs, kvStoreToDataProvider } from 'zoddal';
import { defineCollection } from 'zod-collection-ui';

// 1. Define schema
const PostSchema = z.object({
  id: z.string(),
  title: z.string().meta({ sortable: true, searchable: true }),
  content: z.string().meta({ editable: true }),
  status: z.enum(['draft', 'published']).meta({ filterable: 'select' }),
});

// 2. Build storage store (one line)
const postStore = wrapKvs(restAdapter('/api/posts'), {
  objOfData: (raw) => PostSchema.parse(raw),
});

// 3. Bridge to UI (one line)
const postDataProvider = kvStoreToDataProvider(postStore, PostSchema);

// 4. Define collection (plugs into any zod-collection-ui renderer)
const postCollection = defineCollection({
  schema: PostSchema,
  dataProvider: postDataProvider,
  affordances: { create: true, delete: true, search: true },
});
```

---

## Differences from Python dol

| Aspect | Python dol | zoddal |
|--------|-----------|--------|
| All operations | Synchronous | `async`/`Promise`-based |
| Key type | Any hashable | Typically `string` (URL path segments) |
| Interface definition | `collections.abc` ABCs | TypeScript interfaces |
| Schema language | Type hints + ABCs | Zod schemas |
| Codec composition | `+` operator on `Codec` | `composeCodecs()` function |
| Test backend | `dict` | `memStore()` |
| "Russian dolls" | `wrap_kvs()` stacking | `wrapKvs()` stacking |
| UI layer bridge | N/A (separate concern) | `kvStoreToDataProvider()` |

---

## Open Questions for zoddal Design

1. **Async iteration**: Should `keys()` return `Promise<K[]>` (all at once) or `AsyncIterable<K>` (streaming)? The latter is more general but more complex to use in React.

2. **Optimistic updates**: The KV model is pull-based (fetch → display). For optimistic UI updates, a separate in-memory overlay store (like dol's `WriteBackChainMap`) could be used before the async write completes.

3. **Reactivity**: A `KvStore` is not reactive by itself. A thin reactive wrapper (using `zustand` or signals) around the store would allow React components to subscribe to changes.

4. **Error handling**: Should `get(key)` throw on missing (like Python's `KeyError`) or return `undefined`? The `undefined` model is more idiomatic in TypeScript. The `zoddal` proposal uses `undefined` for missing keys.

5. **Bulk operations**: `updateMany`, `deleteMany` are common in UI contexts but not in the minimal KV interface. Should these be optional methods on `MutableKvStore`?

6. **Server-side queries**: The `DataProvider.getList` interface supports server-side filtering, but the base `KvStore.keys()` doesn't. The bridge uses capability discovery to decide. Alternatively, add an optional `query(params)` method to stores that support it.
