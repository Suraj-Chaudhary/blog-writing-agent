from src.state import State
from src.schema import Task, Plan, EvidenceItem

from langgraph.types import Send
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv

load_dotenv()
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state['topic'],
                "mode": state['mode'],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])]
            },
        )
        for task in state['plan'].tasks
    ]


WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (+-15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
    DO NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
    Focus on summarizing events and implications.
    
Grounding policy:
- If mode == open_book:
    - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URL.
    - For each event claim, attach a source as a Markdown link: ([Source](URL)).
    - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citation == true:
    - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citation is true.

Code:
- If requires_code == true, include atleast one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""

def worker_node(payload: dict) -> dict:

    task = Task(**payload["task"])
    plan = Plan(**payload['plan'])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n-" + "\n-".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requries_citations}\n"
                    f"requires_code: {task.required_code}\n"
                    f"Bullets: {bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {'sections': [(task.id, section_md)]}