from __future__ import annotations

import common as C
import data as D
from mesh_hierarchy import load_hierarchy


def test_prepared_split_counts_and_isolation():
    splits = {name: D.load_split(name) for name in C.SPLITS}
    D.verify_split_isolation(splits)
    assert {name: len(value.scan_ids) for name, value in splits.items()} == {
        "train": 2037,
        "val": 269,
        "test": 277,
    }
    assert {name: len(set(value.subject_ids)) for name, value in splits.items()} == {
        "train": 475,
        "val": 61,
        "test": 61,
    }


def test_hierarchy_contract():
    hierarchy = load_hierarchy()
    assert hierarchy.sizes == [2746, 1373, 344, 86]
    assert [tuple(value.shape) for value in hierarchy.down] == [
        (1373, 2746),
        (344, 1373),
        (86, 344),
    ]
    assert [tuple(value.shape) for value in hierarchy.up] == [
        (2746, 1373),
        (1373, 344),
        (344, 86),
    ]


def test_pair_contract():
    split = D.load_split("train")
    rows = C.load_pairs("train", split.scan_ids, split.subject_ids, split.labels.numpy())
    assert len(rows) == 4208
    assert {(row.diagnosis, row.pair_type) for row in rows} == {
        ("CN", "adjacent"),
        ("CN", "nonadjacent"),
        ("AD", "adjacent"),
        ("AD", "nonadjacent"),
    }

