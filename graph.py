from src.state import State

from typing import Optional
from datetime import date

from components.router import router_node, route_next
from components.research import research_node
from components.orchestrator import orchestrator_node
from components.worker import worker_node, fanout
from reducer.reducer import reducer_subgraph

from langgraph.graph import StateGraph, START, END

g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("join_workers", lambda state: {}) # no-op barrier node
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {'research': "research", 'orchestrator': 'orchestrator'})
g.add_edge('research', 'orchestrator')

g.add_conditional_edges('orchestrator', fanout, ['worker'])
g.add_edge('worker', 'join_workers')
g.add_edge('join_workers', 'reducer')
g.add_edge('reducer', END)

app = g.compile()


def run(topic: str, as_of: Optional[str] = None):

    if as_of is None:
        as_of = date.today().isoformat()

    out = app.invoke(
        {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "sections": [],
            "as_of": as_of,
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "", 
        }
    )

    return out

if __name__ == "__main__":
    run("Future scope of GenAI and AgenticAI")