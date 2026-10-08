"""Chart placement is deliberate, never automatic — fetching a series is not showing it.

The spam this prevents (price, volume and market-cap charts for every entity a sub-agent
touched, on a question about sentiment) only shows up in a live session, so these pin the
wording that both roles and the per-result chart note carry."""

from deep_research_agent import prompts
from deep_research_agent.events import CHART_NOTE


def test_orchestrator_places_charts_deliberately():
    p = prompts.orchestrator_prompt("", "")
    assert "CHARTS (deliberate, never automatic)" in p
    assert "FETCHING data and SHOWING data are different acts" in p
    assert "The default is ZERO charts" in p
    assert "Never the same series twice" in p
    # Briefs ask sub-agents for a rendered chart only on the user's explicit request.
    assert "a brief that does not ask for one gets none" in p
    # The existing OUTPUT rule now defers to the CHARTS section instead of standing alone.
    assert "(and only then — see CHARTS)" in p


def test_subagent_treats_display_tools_as_display_not_data():
    p = prompts.subagent_prompt("", "")
    assert "FETCHING IS NOT SHOWING" in p
    assert "Call a display tool ONLY when your brief explicitly asks for a chart" in p
    assert "A brief that does not mention a chart wants none" in p
    assert "not every series you touched on the way" in p


def test_chart_note_offers_the_chart_without_claiming_it_is_shown():
    note = CHART_NOTE.format(id="1a2b3c4d")
    assert "chart: 1a2b3c4d" in note and "[chart:1a2b3c4d]" in note   # what the tests/UI key on
    assert "already displayed" not in note
    assert "leave it unplaced" in note
