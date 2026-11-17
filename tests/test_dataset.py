import io
import json
import zipfile

import pytest

from core import dataset_loader as dl
from core.data_validator import DatasetValidationError, validate_dataset
from tests.conftest import write_raw

FEATURES = {0: [1, 2], 1: [2, 3], 2: [], 3: [1]}
EDGES = [(0, 1), (1, 0), (1, 2), (2, 2), (2, 3), (0, 9)]  # dup (1,0), self-loop (2,2), unknown user 9


def test_validation_counts_problems_and_cleans(tmp_path):
    raw = write_raw(tmp_path / "raw", EDGES, FEATURES, {0: 5, 1: 5, 2: 6, 3: 6})
    edges, feats, targets = dl.load_raw_tables(dl.locate_raw_files(raw))
    clean, rep = validate_dataset(edges, feats, targets)
    assert rep.total_users == 4
    assert rep.duplicate_edges == 1 and rep.self_loops == 1 and rep.edges_with_unknown_users == 1
    assert rep.total_edges_clean == 3
    assert rep.users_without_features == 1          # user 2 has no features
    assert rep.users_missing_from_features == 1     # user 9 appears only in an edge
    assert rep.connected_components == 1 and rep.isolated_nodes == 0
    assert rep.feature_count == 3 and rep.max_features_per_user == 2
    assert rep.target_ids_consistent
    assert not rep.matches_documented_counts


def test_empty_dataset_raises(tmp_path):
    raw = write_raw(tmp_path / "raw", [], {})
    edges, feats, _ = dl.load_raw_tables(dl.locate_raw_files(raw))
    with pytest.raises(DatasetValidationError):
        validate_dataset(edges, feats, None)


def test_malformed_features_file(tmp_path):
    raw = write_raw(tmp_path / "raw", [(0, 1)], {0: [1], 1: [2]})
    (raw / "lastfm_asia_features.json").write_text("{not json")
    with pytest.raises(DatasetValidationError):
        dl.load_raw_tables(dl.locate_raw_files(raw))


def test_feature_matrix_uses_original_ids(tmp_path):
    import numpy as np
    node_ids = np.array([0, 1, 2, 3])
    X, fids = dl.build_feature_matrix(FEATURES, node_ids)
    assert list(fids) == [1, 2, 3] and X.shape == (4, 3)
    assert X[0].toarray().tolist() == [[1, 1, 0]]      # user 0 likes features 1,2
    assert X[2].nnz == 0                               # missing features stay empty, nothing invented


def test_process_save_and_reload(tmp_path):
    raw = write_raw(tmp_path / "raw", [(0, 1), (1, 2), (2, 3)], FEATURES, {0: 1, 1: 1, 2: 2, 3: 2})
    files = dl.locate_raw_files(raw)
    out = tmp_path / "processed"
    ds = dl.process_dataset(files, out)
    assert dl.processed_is_current(out, files.sha256())
    again = dl.load_processed(out)
    assert list(again.node_ids) == [0, 1, 2, 3] and again.edges.shape[0] == 3
    assert (again.features != ds.features).nnz == 0
    assert not dl.processed_is_current(out, "different-hash")


def test_manual_zip_is_used_without_network(tmp_path):
    raw = write_raw(tmp_path / "src", [(0, 1)], {0: [1], 1: [2]}, {0: 1, 1: 1})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in raw.iterdir():
            z.write(f, f"nested/{f.name}")
    target = tmp_path / "raw"
    target.mkdir()
    (target / "lastfm_asia.zip").write_bytes(buf.getvalue())
    files = dl.ensure_raw_dataset(target, urls=[])      # no URLs: must not touch the network
    assert files.edges.exists() and files.features.exists()


def test_download_failure_gives_manual_instructions(tmp_path):
    with pytest.raises(dl.DatasetDownloadError) as exc:
        dl.download_dataset(tmp_path / "raw", urls=["http://127.0.0.1:9/none.zip"], timeout=2)
    msg = str(exc.value)
    assert "Could not download" in msg and str(tmp_path / "raw") in msg and "lastfm_asia.zip" in msg
