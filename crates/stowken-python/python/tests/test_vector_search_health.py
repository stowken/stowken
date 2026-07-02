"""
Unit tests for the Vector Search index-health helpers.

These exercise pure classification + the ``ensure_index_healthy`` control
flow with a fake WorkspaceClient and monkeypatched setup functions, so they
run without Databricks or the ``databricks-vectorsearch`` package.

Run:
    pytest python/tests/test_vector_search_health.py
"""

import pytest

from stowken.databricks import vector_search as vs


# ── Pure classifiers ──────────────────────────────────────────────────────


def test_classify_status_ready():
    assert vs._classify_index_status({"status": {"ready": True}}) == "ready"


@pytest.mark.parametrize(
    "body",
    [
        {"status": {"ready": False, "detailed_state": "PROVISIONING_ENDPOINT"}},
        {"status": {}},
        {},
        None,
    ],
)
def test_classify_status_syncing(body):
    assert vs._classify_index_status(body) == "syncing"


class _FakeErr(Exception):
    def __init__(self, msg, error_code="", status_code=None):
        super().__init__(msg)
        self.error_code = error_code
        self.status_code = status_code


class NotFound(Exception):
    pass


@pytest.mark.parametrize(
    "exc",
    [
        _FakeErr("nope", status_code=404),
        _FakeErr("RESOURCE_DOES_NOT_EXIST: x", error_code="RESOURCE_DOES_NOT_EXIST"),
        _FakeErr("Index foo does not exist"),
        NotFound("gone"),
    ],
)
def test_classify_error_missing(exc):
    assert vs._classify_index_error(exc) == "missing"


@pytest.mark.parametrize(
    "exc",
    [
        _FakeErr(
            "Something went wrong unexpectedly. ErrorTraceId: abc", status_code=500
        ),
        _FakeErr("AISearchException", status_code=503),
        _FakeErr("timeout"),
    ],
)
def test_classify_error_broken(exc):
    assert vs._classify_index_error(exc) == "broken"


# ── Fake client + control-flow tests ──────────────────────────────────────


class _FakeApiClient:
    """Returns queued responses (dict) or raises queued exceptions per GET/DELETE."""

    def __init__(self, get_results):
        # get_results: list of either dict (body) or Exception
        self._get = list(get_results)
        self.deleted = []

    def do(self, method, path):
        if method == "DELETE":
            self.deleted.append(path)
            return {}
        # GET
        nxt = self._get.pop(0) if self._get else {"status": {"ready": True}}
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _FakeWorkspace:
    def __init__(self, api_client):
        self.api_client = api_client


def _client(get_results):
    return _FakeWorkspace(_FakeApiClient(get_results))


def test_ready_returns_ready_no_recreate(monkeypatch):
    called = []
    monkeypatch.setattr(vs, "setup_index", lambda *a, **k: called.append("setup"))
    state = vs.ensure_index_healthy(
        backend=None,
        endpoint_name="ep",
        index_name="cat.sch.conversations_idx",
        _client=_client([{"status": {"ready": True}}]),
        poll_seconds=0,
    )
    assert state == "ready"
    assert called == []


def test_missing_without_recreate_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="is 'missing'"):
        vs.ensure_index_healthy(
            backend=None,
            endpoint_name="ep",
            index_name="cat.sch.conversations_idx",
            recreate=False,
            _client=_client([NotFound("gone")]),
            poll_seconds=0,
        )


def test_missing_without_recreate_no_raise_returns_state():
    state = vs.ensure_index_healthy(
        backend=None,
        endpoint_name="ep",
        index_name="cat.sch.conversations_idx",
        recreate=False,
        require_ready=False,
        _client=_client([NotFound("gone")]),
        poll_seconds=0,
    )
    assert state == "missing"


def test_missing_with_recreate_calls_setup_index(monkeypatch):
    calls = []
    monkeypatch.setattr(
        vs, "setup_index", lambda backend, **k: calls.append(("conv", k))
    )
    # GET sequence: missing (initial probe) → ready (post-recreate wait)
    state = vs.ensure_index_healthy(
        backend=None,
        endpoint_name="ep",
        index_name="cat.sch.conversations_idx",
        recreate=True,
        _client=_client([NotFound("gone"), {"status": {"ready": True}}]),
        poll_seconds=0,
    )
    assert state == "ready"
    assert len(calls) == 1 and calls[0][0] == "conv"


def test_broken_with_recreate_deletes_then_recreates(monkeypatch):
    calls = []
    monkeypatch.setattr(
        vs, "setup_segment_index", lambda backend, **k: calls.append(("seg", k))
    )
    fake = _client(
        [
            _FakeErr("AISearchException", status_code=500),  # initial probe → broken
            NotFound("gone"),  # post-delete poll → missing
            {"status": {"ready": True}},  # post-recreate wait → ready
        ]
    )
    state = vs.ensure_index_healthy(
        backend=None,
        endpoint_name="ep",
        index_name="cat.sch.segments_idx",  # → kind inferred as segments
        recreate=True,
        _client=fake,
        poll_seconds=0,
    )
    assert state == "ready"
    assert calls and calls[0][0] == "seg"
    assert fake.api_client.deleted == [
        "/api/2.0/vector-search/indexes/cat.sch.segments_idx"
    ]


def test_broken_without_recreate_raises():
    with pytest.raises(RuntimeError, match="is 'broken'"):
        vs.ensure_index_healthy(
            backend=None,
            endpoint_name="ep",
            index_name="cat.sch.conversations_idx",
            recreate=False,
            _client=_client([_FakeErr("AISearchException", status_code=500)]),
            poll_seconds=0,
        )


def test_syncing_then_ready_waits():
    state = vs.ensure_index_healthy(
        backend=None,
        endpoint_name="ep",
        index_name="cat.sch.conversations_idx",
        _client=_client(
            [
                {"status": {"ready": False}},  # probe → syncing
                {"status": {"ready": False}},  # wait poll 1
                {"status": {"ready": True}},  # wait poll 2 → ready
            ]
        ),
        poll_seconds=0,
    )
    assert state == "ready"


def test_kind_inference():
    # Indirectly: a segments_idx recreate must call setup_segment_index.

    seen = {}

    def fake_seg(backend, **k):
        seen["seg"] = True

    def fake_conv(backend, **k):
        seen["conv"] = True

    import stowken.databricks.vector_search as mod

    orig_seg, orig_conv = mod.setup_segment_index, mod.setup_index
    try:
        mod.setup_segment_index = fake_seg
        mod.setup_index = fake_conv
        mod.ensure_index_healthy(
            backend=None,
            endpoint_name="ep",
            index_name="cat.sch.segments_idx",
            recreate=True,
            _client=_client([NotFound("x"), {"status": {"ready": True}}]),
            poll_seconds=0,
        )
    finally:
        mod.setup_segment_index, mod.setup_index = orig_seg, orig_conv
    assert seen == {"seg": True}
