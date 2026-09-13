# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The compiled-plan cache's rules, without a GPU or the DSL: identity is the
whole manifest, an entry is only ever reused under its own embedded key, and
anything doubtful is a miss."""

import json
import pathlib

import pytest

import cudnn
from cudnn.frost import compiled_cache as cc

pytestmark = pytest.mark.L0


def test_manifest_names_every_dependency_and_is_deterministic():
    m = cc.environment_manifest()
    for field in (
        "schema",
        "cudnn_frontend",
        "cudnn_source",
        "cutlass_dsl",
        "tvm_ffi",
        "cuda_driver",
        "device_name",
        "compute_capability",
        "sm_count",
        "l2_bytes",
    ):
        assert field in m and isinstance(m[field], str) and m[field], field
    assert m["schema"] == cc._SCHEMA and m["cudnn_frontend"] == cudnn.__version__
    assert cc.environment_manifest() == m  # same process, same answer


def test_entry_location_is_the_content_hash_of_manifest_and_key(tmp_path):
    m = {"schema": cc._SCHEMA, "cudnn_frontend": "1.30.0", "cutlass_dsl": "4.7.1"}
    a = cc._entry_dir(tmp_path, m, "k1")
    b = cc._entry_dir(tmp_path, m, "k2")
    c = cc._entry_dir(tmp_path, dict(m, cutlass_dsl="4.8.0"), "k1")
    assert a.parent == b.parent and a != b  # same environment, different kernels
    assert c.parent != a.parent  # a dependency moved: a different directory outright
    assert a.parts[-3] == cc._SCHEMA


def test_cache_dir_override_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path / "from_env"))
    assert cc.get_cache_dir() == tmp_path / "from_env"
    cc.set_cache_dir(tmp_path / "explicit")
    try:
        assert cc.get_cache_dir() == tmp_path / "explicit"
    finally:
        cc.set_cache_dir(None)
    assert cc.get_cache_dir() == tmp_path / "from_env"


def test_disabled_or_uncacheable_compiles_as_before(monkeypatch):
    import cutlass.cute as cute

    calls = []
    monkeypatch.setattr(cute, "compile", lambda fn, *a, **k: calls.append((fn, a, k)) or "compiled")
    cc.reset_stats()
    monkeypatch.setenv(cc._ENV_DISABLE, "1")
    assert cc.compile_cached(lambda: None, 1, cache_key="k", options="--enable-tvm-ffi") == "compiled"
    monkeypatch.delenv(cc._ENV_DISABLE)
    assert cc.compile_cached(lambda: None, 1, cache_key=None, options="--enable-tvm-ffi") == "compiled"  # no key: not cacheable
    assert cc.compile_cached(lambda: None, 1, cache_key="k", options="") == "compiled"  # no tvm-ffi: not exportable
    assert len(calls) == 3 and cc.stats()["bypassed"] == 3


def test_an_environment_that_cannot_identify_itself_persists_nothing(monkeypatch, tmp_path):
    import cutlass.cute as cute

    monkeypatch.setattr(cute, "compile", lambda fn, *a, **k: "compiled")
    monkeypatch.setattr(cc, "environment_manifest", lambda device=None: {"schema": cc._SCHEMA, "cudnn_frontend": "1.30.0", "cuda_driver": "unknown"})
    cc.set_cache_dir(tmp_path)
    try:
        cc.reset_stats()
        assert cc.compile_cached(lambda x: x, 1, cache_key="k", options="--enable-tvm-ffi") == "compiled"
    finally:
        cc.set_cache_dir(None)
    assert cc.stats()["bypassed"] == 1 and not list(tmp_path.rglob("*"))


def test_a_foreign_or_corrupt_entry_is_a_miss(tmp_path):
    cc.reset_stats()
    entry = tmp_path / "e"
    entry.mkdir()
    (entry / cc._OBJECT).write_bytes(b"not an object")
    (entry / cc._ENTRY).write_text(json.dumps({"schema": cc._SCHEMA, "key": "other", "symbol": "frost_gemm"}))
    assert cc._try_load(entry, "mine", "frost_gemm") is None  # embedded key differs
    (entry / cc._ENTRY).write_text("{not json")
    assert cc._try_load(entry, "mine", "frost_gemm") is None  # unreadable record
    (entry / cc._ENTRY).unlink()
    assert cc._try_load(entry, "mine", "frost_gemm") is None  # no record at all
    assert cc.stats()["invalid"] == 1  # only the key mismatch is worth a warning; the others are plain misses


def test_the_record_carries_the_in_process_calling_convention():
    """A kernel's runtime signature is its Python signature MINUS constexpr and
    env-stream parameters; the record must say what the DSL's wrapper said."""
    import inspect
    from collections import namedtuple

    Spec = namedtuple("Spec", "arg_names arg_defaults kwonly_names kwonly_defaults")

    class ExecutionArgs:  # what jit_executor exposes
        def get_kwargs_wrapper_spec(self, exclude=()):
            names = [n for n in ("q", "flag", "count", "tail") if n not in exclude]
            defaults = {"count": 0, "tail": None}
            return Spec(names, tuple(defaults[n] for n in names if n in defaults), [n for n in ("stream",) if n not in exclude], {})

    def in_process_wrapper(q, count=0, tail=None, *, stream):  # 'flag' was a Constexpr: gone at run time
        return None

    class Compiled:
        execution_args = ExecutionArgs()
        _kwargs_wrapper = staticmethod(in_process_wrapper)

    spec = cc._wrapper_spec_of(Compiled(), (1, True, 0, None), {})
    assert spec == {"arg_names": ["q", "count", "tail"], "arg_defaults": [0, None], "kwonly_names": ["stream"], "kwonly_defaults": {}}
    calls = []
    rebuilt = cc._rebuild_wrapper(lambda *a: calls.append(a), spec)
    rebuilt("Q", stream="S")
    rebuilt("Q", 7, tail="T", stream="S")
    assert calls == [("Q", 0, None, "S"), ("Q", 7, "T", "S")]  # defaults filled, kwargs placed, positional-only underneath
    assert list(inspect.signature(rebuilt).parameters) == ["q", "count", "tail", "stream"]

    # not reproducible exactly -> no record
    class NoWrapper:
        execution_args = ExecutionArgs()

    assert cc._wrapper_spec_of(NoWrapper(), (), {}) is None
    from dataclasses import dataclass

    @dataclass
    class Op:
        k: int = 1

    assert cc._wrapper_spec_of(Compiled(), (Op(),), {}) is None  # a dataclass argument goes through a hook the record cannot describe
    assert cc._rebuild_wrapper(lambda *a: None, None) is None  # an entry from before the record carried a spec is a miss


def test_template_key_joins_the_digest_with_plain_arguments_only():
    g = {"FROST_SOURCE_DIGEST": "abc123"}
    key = cc.template_key(g, {"b": 2, "qh": 8, "lse_stride": (1, 2, 3), "flag": True, "opt": None}, "compile")
    assert key.startswith("abc123|compile|") and "('b', 2)" in key and "('lse_stride', (1, 2, 3))" in key
    assert cc.template_key(g, {"qh": 8, "b": 2}, "compile") == cc.template_key(g, {"b": 2, "qh": 8}, "compile")  # order-free
    assert cc.template_key(g, {"b": 2}, "compile") != cc.template_key(g, {"b": 2}, "other")  # the function is named
    assert cc.template_key({}, {"b": 2}) is None  # no digest: the loader did not produce this module
    assert cc.template_key(g, {"b": 2, "device": object()}) is None  # a repr that need not survive a process

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class P:
        d: int = 128
        thd: bool = False

    assert "P(d=128, thd=False)" in cc.template_key(g, {"p": P()}, "compile")  # a params record of plain fields is plain
    assert cc.template_key(g, {"p": P(d=object())}, "compile") is None  # unless one of its fields is not
    import cutlass

    assert "cutlass" in cc.template_key(g, {"io_dtype": cutlass.BFloat16}, "compile")  # a dtype CLASS names itself


def test_the_loader_digests_the_file_and_the_params(tmp_path):
    from dataclasses import dataclass

    from cudnn.frost import template_loader

    @dataclass(frozen=True)
    class Params:
        causal: bool = False

    src = tmp_path / "tiny_template.py"
    src.write_text("CFG = FROST_TEMPLATE_PARAMS\n")
    a = template_loader.load_template(str(src), Params(False), tag="tiny")
    b = template_loader.load_template(str(src), Params(True), tag="tiny")
    assert a.FROST_SOURCE_DIGEST and len(a.FROST_SOURCE_DIGEST) == 16
    assert a.FROST_SOURCE_DIGEST != b.FROST_SOURCE_DIGEST  # the params are part of it
    assert template_loader.load_template(str(src), Params(False), tag="tiny") is a  # the module cache still keys on (path, params)
    src.write_text("CFG = FROST_TEMPLATE_PARAMS  # edited\n")
    c = template_loader.load_template(str(src), Params(None), tag="tiny")  # a new params value forces a re-read
    assert c.FROST_SOURCE_DIGEST not in (a.FROST_SOURCE_DIGEST, b.FROST_SOURCE_DIGEST)


def test_the_manifest_carries_the_source_tree_not_a_commit(tmp_path):
    """An edit anywhere in the package -- committed or not -- lands in another
    environment directory; a wheel hashes the same every process."""
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y = 2\n")
    (tmp_path / "sub" / "__pycache__").mkdir()
    (tmp_path / "sub" / "__pycache__" / "b.cpython-312.pyc").write_bytes(b"ignored")
    before = cc.source_tree_digest(tmp_path)
    assert before == cc.source_tree_digest(tmp_path) and len(before) == 16
    (tmp_path / "sub" / "__pycache__" / "b.cpython-312.pyc").write_bytes(b"still ignored")
    assert cc.source_tree_digest(tmp_path) == before
    (tmp_path / "sub" / "b.py").write_text("y = 3\n")  # a shared helper edited, no version bump
    assert cc.source_tree_digest(tmp_path) != before
    import cudnn

    assert cc.environment_manifest()["cudnn_source"] == cc.source_tree_digest(pathlib.Path(cudnn.__file__).resolve().parent)
