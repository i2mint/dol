import pytest
from dol.sources import FanoutPersister
from dol.trans import wrap_kvs


@pytest.mark.parametrize(
    "value_encoder,value_decoder,persister_args,persister_kwargs,data",
    [
        (
            # Tuple fan-out store (separate values)
            lambda v: {k: v for k, v in enumerate(v)},
            lambda v: tuple(vv for vv in v.values()),
            (dict(), dict(), dict()),
            dict(get_existing_values_only=True),
            [
                ("k1", (1, 2, 3), None),
                ("k2", (1, 2), None),
                ("k3", (1, 2, 3, 4), ValueError),
            ],
        ),
        (
            # Salary fan-out store (computed values)
            lambda v: dict(net=v * 0.8, tax=v * 0.2),  # Gross to breakdown
            lambda v: sum(v.values()),  # Breakdown to gross
            (),
            dict(net=dict(), tax=dict()),
            [
                ("Peter", 3000, None),
                ("Paul", 5000, None),
                ("Jack", 10000, None),
            ],
        ),
    ],
)
def test_mk_custom_fanout_store(
    value_encoder, value_decoder, persister_args, persister_kwargs, data
):
    store = wrap_kvs(
        FanoutPersister.from_variadics(
            *persister_args,
            **persister_kwargs,
        ),
        data_of_obj=value_encoder,
        obj_of_data=value_decoder,
    )
    for k, v, error in data:

        def test_data():
            store[k] = v
            assert store[k] == v
            del store[k]
            assert k not in store

        if error is None:
            test_data()
        else:
            with pytest.raises(error):
                test_data()
