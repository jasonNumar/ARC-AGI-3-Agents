from typing import Callable, Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .observer_arc_agent import ObserverArcAgent
from .recorder import Recorder
from .swarm import Swarm
from .templates.random_agent import Random

load_dotenv()


OPTIONAL_IMPORT_ERRORS: dict[str, str] = {}


def _try_optional(name: str, loader: Callable[[], None]) -> None:
    try:
        loader()
    except ImportError as exc:
        OPTIONAL_IMPORT_ERRORS[name] = str(exc)


def _load_langgraph_functional() -> None:
    global LangGraphFunc, LangGraphTextOnly
    from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly


def _load_langgraph_random() -> None:
    global LangGraphRandom
    from .templates.langgraph_random_agent import LangGraphRandom


def _load_langgraph_thinking() -> None:
    global LangGraphThinking
    from .templates.langgraph_thinking import LangGraphThinking


def _load_llm_agents() -> None:
    global LLM, FastLLM, GuidedLLM, ReasoningLLM
    from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM


def _load_multimodal() -> None:
    global MultiModalLLM
    from .templates.multimodal import MultiModalLLM


def _load_reasoning_agent() -> None:
    global ReasoningAgent
    from .templates.reasoning_agent import ReasoningAgent


def _load_smolagents() -> None:
    global SmolCodingAgent, SmolVisionAgent
    from .templates.smolagents import SmolCodingAgent, SmolVisionAgent


_try_optional("langgraph_functional", _load_langgraph_functional)
_try_optional("langgraph_random", _load_langgraph_random)
_try_optional("langgraph_thinking", _load_langgraph_thinking)
_try_optional("llm_agents", _load_llm_agents)
_try_optional("multimodal", _load_multimodal)
_try_optional("reasoning_agent", _load_reasoning_agent)
_try_optional("smolagents", _load_smolagents)

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
if "ReasoningAgent" in globals():
    AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent

__all__ = [name for name in [
    "Swarm",
    "Random",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphThinking",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
    "MultiModalLLM",
    "ObserverArcAgent",
    "OPTIONAL_IMPORT_ERRORS",
] if name in globals()]
