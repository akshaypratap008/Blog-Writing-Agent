import os
import operator
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated
from pydantic import BaseModel, Field
import re
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from src.blog_writing_agent.utils import load_prompt

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ..., 
        description="One sentence describing what the reader should be able to understand/do after this section."
    )
    bullets: List[str] = Field(
        ...,
        min_length= 3,
        max_length=6,
        description="3-6 concrete, non-overlapping subpoints to cover in this section."
    )

    target_words: int = Field(
        ...,
        description= "Target word count for this section. (120-550)"
    )

    tags: List[str] = Field(
        default_factory= list
    )

    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparision", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory= list)
    tasks: List[Task]

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None    # keep if tavily provides else ignore
    snippet: Optional[str] = None
    source : Optional[str] = None

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal['closed_book', 'hybrid', 'open_book']
    queries : List[str] = Field(default_factory = list)

class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory= list)

class ImageSpec(BaseModel):
    placeholder: str = Field(..., description= "e.g. [[IMAGE_1]]")
    filename: str = Field(..., description = "save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description= "prompt to send to the image model")
    size : Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"

class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research : bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # worker
    sections: Annotated[List[tuple[int, str]], operator.add]

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    
    final: str

llm = ChatOpenAI(model = 'gpt-4.1-mini')

router_prompt = load_prompt("router_decision_prompt")

def router_node(state: State) -> State:
    topic = state['topic']
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content = router_prompt),
            HumanMessage(content = f"Topic: {topic}")
        ]
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries
    }

# conditional function to route to reach or orchestrator node
def route_next(state: State):
    return "research" if state['needs_research'] else "orchestrator"

# tavily search
def _tavily_search(query: str, max_results: int) -> List[dict]:
    tool = TavilySearchResults(max_results= max_results)
    results = tool.invoke({"query": query})

    normalised: List[dict] = []

    for r in results or []:
        normalised.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source")
            }
        )
    return normalised

def research_node(state: State) -> State:
    # take first 10 queries from the state
    queries = (state.get("queries", []) or [])[:10]
    max_results = 6
    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(_tavily_search(query=q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)

    pack = extractor.invoke(
        [
            SystemMessage(content = load_prompt("research_prompt")),
            HumanMessage(content = f"Raw Results : \n{raw_results}")
        ]
    )

    # deduplicate by url
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {
        "evidence": list(dedup.values())
    }

def orchestrator(state: State):
    planner = llm.with_structured_output(Plan)

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = planner.invoke(
        [
            SystemMessage(content= load_prompt("orchestrator_prompt")),
            HumanMessage(content= (
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY) use for fresh claims; may be empty: \n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            )
        ]
    )

    return {"plan": plan}

# fanout - this is required so that the output from orchestrator (plan) is sent to multiple worker to perform task parallely 
def fanout(state: State):
    
    return [Send(
        "worker",
        {
            "task": task.model_dump(),
            "topic": state['topic'],
            "mode": state['mode'],
            "plan": state['plan'].model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])]
        } 
    )for task in state['plan'].tasks]

def worker_node(payload: dict) -> dict:
    
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = llm.invoke(
        [
            SystemMessage(content=load_prompt("worker_system_prompt")),
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
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

# -- Subgraph for reducer --

# reducer
# merge content -> decide images -> generate and place images

def merge_content(state: State) -> State:
    plan = state['plan']

    ordered_sections = [md for _, md in sorted(state['sections'], key = lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()

    merged_md = f"# {plan.blog_title}\n\n{body}"
    return {"merged_md": merged_md}

def decide_images(state: State): 
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state['merged_md']
    plan = state['plan']
    assert plan is not None 

    image_plan = planner.invoke(
        [
            SystemMessage(
                content = load_prompt("decide_image_prompt")
            ),
            HumanMessage(
                content = (
                    f"Blog Kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompt. \n\n"
                    f"{merged_md}"
                )
            )
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images]
    }

def _gemini_generate_image_bytes(prompt: str) -> bytes:
    "Return raw image bytes generated by Gemini"

    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY Error")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model = "gemini-2.5-flash-image",
        contents = prompt,
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category= "HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH"
                )
            ]
        )
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")

def generate_and_place_images(state: State):
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    output_dir = Path("final_blogs")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    safe_title = re.sub(r'[<>:"/\\|?*]+', "_", plan.blog_title).strip(" .") or "blog"
    output_path = output_dir / f"{safe_title}.md"

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = Path(spec["filename"]).name
        out_path = images_dir / filename

        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption', '')}\n>\n"
                    f"> **Alt:** {spec.get('alt', '')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt', '')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        if placeholder in md:
            md = md.replace(placeholder, img_md)
        else:
            md = f"{md.rstrip()}\n\n{img_md}\n"

    output_path.write_text(md, encoding="utf-8")
    return {"final": md}

# --- Building Subgraph for reducer --- 

reducer_graph = StateGraph(State)

reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)

reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)

reducer_subgraph = reducer_graph.compile()

reducer_subgraph

# ---- Build Main Graph ---- 

g = StateGraph(State)

g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

from datetime import date

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
    run("How AI is revolutionising Software Engineering in 2026")
