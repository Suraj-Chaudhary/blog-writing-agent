from src.state import State
from src.schema import Plan

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv

load_dotenv()
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5-9 sections (tasks) suitable for the topic and audience.
- Each task must include:
    1) goal (1 sentence)
    2) 3-6 bullets that are concrete, specific, and non-overlapping
    3) target word count (120-550)
    
Quality bar:
- Assume the reader is a developer; use correct technology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes atleast 2 of these somewhere:
    * minimal code sketch / MWE (set requires_code=True for that section)
    * edge cases / failure modes
    * performance/cost considerations
    * security/privacy considerations (if relevant)
    * debugging/observability tips
    
Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
    - Use evidence for up-to-date examples (models/tools/releases) in bullets.
    - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
    - Set blog_king = "news_roundup".
    - Every section is about summarizing events + implications.
    - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
    - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
      and include only what can be supported.
      
Output must strictly match the Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence[:16]]}"
                )
            ),
        ]
    )

    return {'plan': plan}