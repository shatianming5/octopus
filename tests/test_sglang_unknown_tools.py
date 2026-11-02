"""read_thread is not in a request's tools until a read can succeed (docs/cross-branch-read.md), so a parser must forward
calls to tools the request does not list. SGLang's JSON-family detectors (0.5.10.post1 in the trainer venv, 0.5.19 on the
eval servers) drop them unless
SGLANG_FORWARD_UNKNOWN_TOOLS=1, which octopus/rl/config.set_derived_settings sets wherever AReaL's proxy parses
(train.py's workers, dev_eval and rollout_load)."""
import pytest

fcp = pytest.importorskip("sglang.srt.function_call.function_call_parser")
from sglang.srt.entrypoints.openai.protocol import Function, Tool  # noqa: E402

CALL = '{"name": "read_thread", "arguments": {"thread": "t1"}}'


@pytest.mark.parametrize("parser,reply", [("qwen25", f"<tool_call>\n{CALL}\n</tool_call>"),
                                          ("llama3", f"<|python_tag|>{CALL}")])
def test_unknown_tool_forwarded_only_with_flag(monkeypatch, parser, reply):
    tools = [Tool(type="function", function=Function(name="bash", parameters={"type": "object", "properties": {}}))]
    monkeypatch.delenv("SGLANG_FORWARD_UNKNOWN_TOOLS", raising=False)
    assert fcp.FunctionCallParser(tools, parser).parse_non_stream(reply)[1] == []
    monkeypatch.setenv("SGLANG_FORWARD_UNKNOWN_TOOLS", "1")
    (call,) = fcp.FunctionCallParser(tools, parser).parse_non_stream(reply)[1]
    assert call.name == "read_thread" and call.parameters == '{"thread": "t1"}'
