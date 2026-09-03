"""Third-party request loggers are quieted on import, unless DEBUG is on."""

import logging

from deep_research_agent.agent import _quiet_third_party_loggers

NOISY = ("httpx", "httpx2", "mcp.client.streamable_http")


def test_request_loggers_are_quieted_on_import():
    # Importing the agent module (done above) already ran the quieting.
    for name in NOISY:
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING, name


def test_debug_root_keeps_request_logs():
    root = logging.getLogger()
    previous = root.level
    for name in NOISY:
        logging.getLogger(name).setLevel(logging.NOTSET)
    root.setLevel(logging.DEBUG)
    try:
        _quiet_third_party_loggers()
        for name in NOISY:
            assert logging.getLogger(name).level == logging.NOTSET, name
    finally:
        root.setLevel(previous)
        _quiet_third_party_loggers()
