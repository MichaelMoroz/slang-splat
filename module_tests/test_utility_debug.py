from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fake_slangpy = ModuleType("slangpy")
fake_slangpy.float3 = lambda *args: tuple(args)
fake_slangpy.Tensor = type("Tensor", (), {})
fake_slangpy.Buffer = type("Buffer", (), {})
fake_slangpy.ShaderCursor = object
fake_slangpy.BufferOffsetPair = lambda buffer, offset: (buffer, offset)
sys.modules.setdefault("slangpy", fake_slangpy)

import utility.utility as utility_module

from utility.debug import with_debug_group
from utility.utility import dispatch, dispatch_indirect


class _FakeCommandEncoder:
    def __init__(self) -> None:
        self.events: list[tuple[str, object, object]] = []
        self.compute_pass = _FakeComputePass()

    def push_debug_group(self, label: object, color: object) -> None:
        self.events.append(("push", label, color))

    def pop_debug_group(self) -> None:
        self.events.append(("pop", None, None))

    def begin_compute_pass(self) -> "_FakeComputePassContext":
        return _FakeComputePassContext(self.compute_pass)


class _FakeComputePassContext:
    def __init__(self, compute_pass: "_FakeComputePass") -> None:
        self.compute_pass = compute_pass

    def __enter__(self) -> "_FakeComputePass":
        return self.compute_pass

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeComputePass:
    def __init__(self) -> None:
        self.bound = None
        self.dispatch_pair = None

    def bind_pipeline(self, pipeline: object) -> dict[str, object]:
        self.bound = {"pipeline": pipeline}
        return self.bound

    def dispatch_compute_indirect(self, pair: object) -> None:
        self.dispatch_pair = pair


class _FakeKernel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def dispatch(self, *, thread_count: object, vars: dict[str, object], command_encoder: object) -> None:
        self.calls.append({"thread_count": thread_count, "vars": vars, "command_encoder": command_encoder})


class _FakeShaderCursor:
    def __init__(self, bound: dict[str, object]) -> None:
        object.__setattr__(self, "_bound", bound)

    def __setattr__(self, name: str, value: object) -> None:
        self._bound[name] = value


def test_with_debug_group_wraps_function_once() -> None:
    encoder = _FakeCommandEncoder()

    @with_debug_group("scope", "amber")
    def run(command_encoder: _FakeCommandEncoder, value: int) -> int:
        command_encoder.events.append(("body", value, None))
        return value + 1

    assert run(encoder, 4) == 5
    assert encoder.events == [("push", "scope", "amber"), ("body", 4, None), ("pop", None, None)]


def test_dispatch_wraps_kernel_call_in_optional_debug_group() -> None:
    encoder = _FakeCommandEncoder()
    kernel = _FakeKernel()
    vars = {"value": 7}

    dispatch(
        kernel=kernel,
        thread_count="threads",
        vars=vars,
        command_encoder=encoder,
        debug_label="dispatch.scope",
        debug_color="blue",
    )

    assert encoder.events == [("push", "dispatch.scope", "blue"), ("pop", None, None)]
    assert kernel.calls == [{"thread_count": "threads", "vars": vars, "command_encoder": encoder}]


def test_dispatch_indirect_binds_vars_and_applies_offset(monkeypatch) -> None:
    encoder = _FakeCommandEncoder()
    monkeypatch.setattr(utility_module.spy, "ShaderCursor", _FakeShaderCursor)
    monkeypatch.setattr(utility_module.spy, "BufferOffsetPair", lambda buffer, offset: (buffer, offset))

    dispatch_indirect(
        pipeline="pipeline",
        args_buffer="args",
        vars={"alpha": 1, "beta": 2},
        command_encoder=encoder,
        arg_offset=3,
        resource_binder=lambda value: f"bound:{value}",
        debug_label="indirect.scope",
        debug_color="cyan",
    )

    assert encoder.events == [("push", "indirect.scope", "cyan"), ("pop", None, None)]
    assert encoder.compute_pass.bound == {"pipeline": "pipeline", "alpha": "bound:1", "beta": "bound:2"}
    assert encoder.compute_pass.dispatch_pair == ("args", 12)
