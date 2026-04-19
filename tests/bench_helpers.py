"""Synthetic 5-year store generator for benchmarking.

Populates an encrypted MemoryStore with realistic data at scale:
- ~45,000 facts across all fact types
- ~2,000 entities (people, places, projects, etc.)
- ~8,000 relationships
- ~18,000 episodes

Also provides helpers for tier-based population (1k, 5k, 10k, 19k, 45k vectors).
"""

from __future__ import annotations

import math
import random
import struct
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sieve.config import StoreConfig
from sieve.store import MemoryStore

# --- Vocabulary pools for realistic content ---

_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Pete",
    "Quinn", "Ruby", "Sam", "Tara", "Uma", "Vince", "Wendy", "Xander",
    "Yara", "Zane", "Amir", "Bianca", "Carlos", "Dalia", "Emil", "Fatima",
    "Gustav", "Helena", "Ivan", "Julia", "Karim", "Layla", "Mikhail", "Nadia",
]

_CITIES = [
    "Dubai", "London", "Tokyo", "New York", "Paris", "Berlin", "Sydney",
    "Singapore", "Toronto", "Mumbai", "Seoul", "San Francisco", "Amsterdam",
    "Barcelona", "Hong Kong", "Istanbul", "Zurich", "Bangkok", "Moscow",
    "Cape Town", "Dubai Marina", "Abu Dhabi", "Riyadh", "Cairo",
]

_PROFESSIONS = [
    "pilot", "software engineer", "data scientist", "architect", "teacher",
    "doctor", "chef", "photographer", "journalist", "consultant",
    "designer", "researcher", "entrepreneur", "analyst", "manager",
]

_HOBBIES = [
    "photography", "cooking", "hiking", "reading", "gardening",
    "chess", "painting", "cycling", "swimming", "gaming",
    "yoga", "sailing", "woodworking", "astronomy", "writing",
]

_PROJECTS = [
    "Project Alpha", "Recall", "DataSync", "CloudBridge", "NeuralNet",
    "Quantum", "Fusion", "Catalyst", "Nexus", "Horizon",
    "Pinnacle", "Vertex", "Meridian", "Zenith", "Aurora",
    "Titan", "Phoenix", "Orion", "Eclipse", "Summit",
]

_FACT_TEMPLATES = [
    "User lives in {city}",
    "User is a {profession}",
    "User's hobby is {hobby}",
    "User works at {company}",
    "User is {age} years old",
    "User has a {relation} named {name}",
    "User's favorite food is {food}",
    "User speaks {language}",
    "User drives a {car}",
    "User studied at {university}",
    "User prefers {preference}",
    "User is allergic to {allergen}",
    "User's salary is ${amount}k per year",
    "User has been working for {years} years",
    "User's phone number is +{number}",
]

_COMPANIES = ["Google", "Microsoft", "Amazon", "Apple", "Meta", "OpenAI", "Anthropic",
              "Tesla", "SpaceX", "Stripe", "Airbnb", "Uber", "Netflix", "Spotify"]
_FOODS = ["sushi", "pizza", "tacos", "pasta", "curry", "steak", "ramen", "falafel"]
_LANGUAGES = ["English", "Arabic", "French", "Spanish", "Mandarin", "Japanese", "German"]
_CARS = ["Tesla Model 3", "BMW 3-Series", "Toyota Camry", "Honda Civic", "Mercedes C-Class"]
_UNIVERSITIES = ["MIT", "Stanford", "Oxford", "Cambridge", "ETH Zurich", "Caltech", "Harvard"]
_PREFERENCES = ["dark mode", "light mode", "vim keybindings", "early mornings", "late nights"]
_ALLERGENS = ["peanuts", "shellfish", "gluten", "dairy", "dust"]
_RELATIONS = ["brother", "sister", "mother", "father", "friend", "colleague", "partner"]

_EPISODE_TEMPLATES = [
    "Discussed {topic} with {name} during {event}",
    "Made a decision to {action} at {location}",
    "Had a meeting about {project} with {name}",
    "Traveled to {city} for {reason}",
    "Completed {task} for {project}",
    "Learned about {topic} from {source}",
    "Resolved conflict between {name1} and {name2} regarding {topic}",
    "Planned {event} scheduled for next {timeframe}",
    "Reviewed progress on {project} — {status}",
    "Debugged issue in {project} related to {component}",
]

_TOPICS = ["API design", "database migration", "performance", "security", "UX", "testing",
           "deployment", "architecture", "budgeting", "team structure", "roadmap"]
_EVENTS = ["standup", "sprint review", "dinner", "conference", "workshop", "hackathon"]
_ACTIONS = ["refactor the auth module", "switch to PostgreSQL", "hire a designer",
            "upgrade the infrastructure", "migrate to cloud", "adopt TDD"]
_TASKS = ["code review", "documentation", "unit tests", "integration tests", "deployment"]
_SOURCES = ["online course", "textbook", "colleague", "conference talk", "blog post"]
_COMPONENTS = ["auth module", "API gateway", "cache layer", "message queue", "scheduler"]
_STATUSES = ["on track", "behind schedule", "ahead of plan", "blocked", "completed"]
_TIMEFRAMES = ["week", "month", "quarter"]

_ENTITY_TYPES = ["person", "place", "project", "concept", "object", "asset"]
_RELATIONSHIP_TYPES = [
    "knows", "works_with", "lives_in", "works_on", "manages",
    "reports_to", "mentor_of", "friend_of", "sibling_of", "parent_of",
    "located_in", "part_of", "depends_on", "related_to", "nuanced_view",
    "temporal_update", "collaborates_with", "owns", "uses",
]

_PREFERENCE_CATEGORIES = ["communication", "retrieval", "behaviour", "query_pattern"]


def _random_embedding(dim: int = 768) -> list[float]:
    """Generate a random unit-normalized embedding vector."""
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def _themed_embedding(theme_idx: int, dim: int = 768, noise: float = 0.3) -> list[float]:
    """Generate an embedding biased toward a theme cluster.

    This creates clusters in embedding space so vector search benchmarks
    are more realistic than pure random.
    """
    random.seed(theme_idx * 31337)  # deterministic cluster center
    center = [random.gauss(0, 1) for _ in range(dim)]
    random.seed()  # re-randomize

    vec = [c + random.gauss(0, noise) for c in center]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def _fill_template(template: str) -> str:
    """Fill a template string with random vocabulary."""
    replacements = {
        "{city}": random.choice(_CITIES),
        "{profession}": random.choice(_PROFESSIONS),
        "{hobby}": random.choice(_HOBBIES),
        "{company}": random.choice(_COMPANIES),
        "{age}": str(random.randint(20, 65)),
        "{relation}": random.choice(_RELATIONS),
        "{name}": random.choice(_NAMES),
        "{name1}": random.choice(_NAMES),
        "{name2}": random.choice(_NAMES),
        "{food}": random.choice(_FOODS),
        "{language}": random.choice(_LANGUAGES),
        "{car}": random.choice(_CARS),
        "{university}": random.choice(_UNIVERSITIES),
        "{preference}": random.choice(_PREFERENCES),
        "{allergen}": random.choice(_ALLERGENS),
        "{amount}": str(random.randint(40, 300)),
        "{years}": str(random.randint(1, 30)),
        "{number}": str(random.randint(1000000000, 9999999999)),
        "{topic}": random.choice(_TOPICS),
        "{event}": random.choice(_EVENTS),
        "{action}": random.choice(_ACTIONS),
        "{location}": random.choice(_CITIES),
        "{project}": random.choice(_PROJECTS),
        "{task}": random.choice(_TASKS),
        "{source}": random.choice(_SOURCES),
        "{reason}": random.choice(_EVENTS),
        "{component}": random.choice(_COMPONENTS),
        "{status}": random.choice(_STATUSES),
        "{timeframe}": random.choice(_TIMEFRAMES),
    }
    result = template
    for key, val in replacements.items():
        result = result.replace(key, val)
    return result


@dataclass
class GenerationStats:
    """Statistics from a synthetic store generation run."""
    facts: int = 0
    entities: int = 0
    relationships: int = 0
    episodes: int = 0
    preferences: int = 0
    sessions: int = 0
    elapsed_s: float = 0.0

    def summary(self) -> str:
        return (
            f"Generated: {self.facts:,} facts, {self.entities:,} entities, "
            f"{self.relationships:,} relationships, {self.episodes:,} episodes, "
            f"{self.preferences:,} preferences, {self.sessions:,} sessions "
            f"in {self.elapsed_s:.1f}s"
        )


def _serialize_emb(emb: list[float] | None) -> bytes | None:
    """Serialize embedding to bytes for direct SQL insertion."""
    if emb is None:
        return None
    return struct.pack(f"<{len(emb)}f", *emb)


def populate_store(
    store: MemoryStore,
    *,
    num_facts: int = 45_000,
    num_entities: int = 2_000,
    num_relationships: int = 8_000,
    num_episodes: int = 18_000,
    num_preferences: int = 200,
    num_sessions: int = 500,
    embedding_dim: int = 768,
    with_embeddings: bool = True,
    batch_size: int = 500,
    seed: int = 42,
) -> GenerationStats:
    """Populate a MemoryStore with synthetic data.

    Uses raw SQL batch inserts in a single transaction for speed.
    All data is deterministic given the same seed.
    """
    random.seed(seed)
    stats = GenerationStats()
    t0 = time.monotonic()
    conn = store.conn
    now = "2024-01-01T00:00:00Z"

    fact_types = ["objective", "subjective", "conditional", "temporal"]
    fact_type_weights = [0.5, 0.2, 0.15, 0.15]

    # --- Entities (single transaction) ---
    entity_ids: list[str] = []
    conn.execute("BEGIN")
    for i in range(num_entities):
        etype = random.choice(_ENTITY_TYPES)
        if etype == "person":
            name = f"{random.choice(_NAMES)} {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis'])}"
        elif etype == "place":
            name = random.choice(_CITIES) + (f" {random.choice(['Office', 'Airport', 'Station', 'Mall', 'Park'])}" if random.random() > 0.5 else "")
        elif etype == "project":
            name = random.choice(_PROJECTS) + f" v{random.randint(1, 5)}.{random.randint(0, 9)}"
        else:
            name = f"{etype.capitalize()}_{i}"

        eid = uuid.uuid4().hex[:16]
        emb_blob = _serialize_emb(_themed_embedding(i % 50, embedding_dim)) if with_embeddings else None
        conn.execute(
            "INSERT INTO entities (id, name, type, description, embedding, created_at) VALUES (?,?,?,?,?,?)",
            (eid, name, etype, f"Auto-generated {etype}", emb_blob, now),
        )
        entity_ids.append(eid)
        stats.entities += 1
    conn.execute("COMMIT")

    # --- Facts (batched transactions) ---
    num_themes = min(100, max(1, num_facts // 100))
    conn.execute("BEGIN")
    for i in range(num_facts):
        template = random.choice(_FACT_TEMPLATES)
        content = _fill_template(template) + f" (ref-{i})"
        ft = random.choices(fact_types, fact_type_weights, k=1)[0]
        confidence = round(random.uniform(0.3, 1.0), 2)
        theme = i % num_themes
        emb_blob = _serialize_emb(_themed_embedding(theme, embedding_dim)) if with_embeddings else None

        linked = None
        if entity_ids and random.random() < 0.3:
            linked = random.sample(entity_ids, min(3, len(entity_ids)))

        fact_id = uuid.uuid4().hex[:16]
        conn.execute(
            """INSERT INTO facts
               (id, content, embedding, entity_ids, source, confidence,
                fact_type, status, status_detail, created_at, last_confirmed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (fact_id, content, emb_blob,
             str(linked) if linked else None,
             "synthetic", confidence, ft, "current", "provisional", now, now),
        )
        # Insert into vec_facts if embedding exists
        if emb_blob is not None:
            conn.execute(
                "INSERT INTO vec_facts (fact_id, embedding) VALUES (?, ?)",
                (fact_id, emb_blob),
            )
        stats.facts += 1

        if (i + 1) % batch_size == 0:
            conn.execute("COMMIT")
            conn.execute("BEGIN")

    conn.execute("COMMIT")

    # --- Relationships ---
    conn.execute("BEGIN")
    for i in range(num_relationships):
        if len(entity_ids) < 2:
            break
        src, tgt = random.sample(entity_ids, 2)
        rel_type = random.choice(_RELATIONSHIP_TYPES)
        confidence = round(random.uniform(0.4, 1.0), 2)
        rel_id = uuid.uuid4().hex[:16]
        conn.execute(
            """INSERT INTO relationships
               (id, source_entity, relationship, target_entity, confidence, status, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (rel_id, src, rel_type, tgt, confidence, "current", now),
        )
        stats.relationships += 1
        if (i + 1) % batch_size == 0:
            conn.execute("COMMIT")
            conn.execute("BEGIN")
    conn.execute("COMMIT")

    # --- Sessions ---
    session_ids: list[str] = []
    conn.execute("BEGIN")
    for i in range(num_sessions):
        sid = uuid.uuid4().hex[:16]
        coherence = round(random.uniform(0.3, 1.0), 2)
        conn.execute(
            "INSERT INTO sessions (id, coherence_score, message_count, started_at, ended_at) VALUES (?,?,?,?,?)",
            (sid, coherence, random.randint(5, 50), now, now),
        )
        session_ids.append(sid)
        stats.sessions += 1
    conn.execute("COMMIT")

    # --- Episodes ---
    conn.execute("BEGIN")
    for i in range(num_episodes):
        template = random.choice(_EPISODE_TEMPLATES)
        summary = _fill_template(template)
        theme = i % num_themes
        emb_blob = _serialize_emb(_themed_embedding(theme + 200, embedding_dim)) if with_embeddings else None

        involved = None
        if entity_ids and random.random() < 0.4:
            involved = random.sample(entity_ids, min(4, len(entity_ids)))

        decisions = None
        if random.random() < 0.3:
            decisions = [_fill_template("Decided to {action}")]

        sid = random.choice(session_ids) if session_ids else None
        ep_id = uuid.uuid4().hex[:16]
        conn.execute(
            """INSERT INTO episodes
               (id, summary, embedding, entities_involved, decisions_made, session_id, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (ep_id, summary, emb_blob,
             str(involved) if involved else None,
             str(decisions) if decisions else None,
             sid, now),
        )
        stats.episodes += 1
        if (i + 1) % batch_size == 0:
            conn.execute("COMMIT")
            conn.execute("BEGIN")
    conn.execute("COMMIT")

    # --- Preferences ---
    conn.execute("BEGIN")
    for i in range(num_preferences):
        cat = random.choice(_PREFERENCE_CATEGORIES)
        content = f"{cat}_pref_{i}: {random.choice(_PREFERENCES)}"
        strength = round(random.uniform(0.3, 1.0), 2)
        pref_id = uuid.uuid4().hex[:16]
        conn.execute(
            "INSERT INTO preferences (id, category, content, strength, observation_count, last_observed_at) VALUES (?,?,?,?,?,?)",
            (pref_id, cat, content, strength, 1, now),
        )
        stats.preferences += 1
    conn.execute("COMMIT")

    stats.elapsed_s = time.monotonic() - t0
    return stats


def create_populated_store(
    tmp_path: Path,
    *,
    num_facts: int = 45_000,
    num_entities: int = 2_000,
    num_relationships: int = 8_000,
    num_episodes: int = 18_000,
    embedding_dim: int = 768,
    with_embeddings: bool = True,
    seed: int = 42,
) -> tuple[MemoryStore, GenerationStats]:
    """Create a new MemoryStore at tmp_path and populate it.

    Returns (store, stats). Store is left open.
    """
    config = StoreConfig(
        path=str(tmp_path / "bench.db"),
        embedding_dimensions=embedding_dim,
    )
    store = MemoryStore(config, passphrase="bench-test-key")
    store.open()
    store.init_schema()

    gen_stats = populate_store(
        store,
        num_facts=num_facts,
        num_entities=num_entities,
        num_relationships=num_relationships,
        num_episodes=num_episodes,
        embedding_dim=embedding_dim,
        with_embeddings=with_embeddings,
        seed=seed,
    )
    return store, gen_stats


# --- Realistic payload generators ---

def make_bloated_payload(
    user_query: str = "Where do I live?",
    *,
    system_prompt_tokens: int = 9600,
    tool_schema_tokens: int = 8000,
    workspace_tokens: int = 3600,
    history_turns: int = 20,
) -> dict:
    """Generate a realistically bloated OpenClaw-style Ollama payload.

    Simulates ~47k tokens of context: system prompt, tool schemas,
    workspace files, and long conversation history.
    """
    # Build a big system prompt (~9600 tokens ≈ ~38k chars)
    system_prompt = "You are an advanced AI assistant.\n" * (system_prompt_tokens * 4 // 40)

    # Build tool schemas (~8000 tokens)
    tools = []
    for i in range(50):
        tools.append({
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": f"Tool {i} that does something very specific and important. " * 8,
                "parameters": {
                    "type": "object",
                    "properties": {
                        f"param_{j}": {
                            "type": "string",
                            "description": f"Parameter {j} for tool {i}. " * 4,
                        }
                        for j in range(8)
                    },
                },
            },
        })

    # Build conversation history
    messages = [{"role": "system", "content": system_prompt}]

    # Workspace files as system messages
    workspace_content = "// workspace file content\n" * (workspace_tokens * 4 // 25)
    messages.append({"role": "system", "content": f"<workspace>\n{workspace_content}\n</workspace>"})

    for i in range(history_turns):
        messages.append({
            "role": "user",
            "content": f"Question {i}: Tell me about {random.choice(_TOPICS)} in detail. "
                        f"Also consider {random.choice(_TOPICS)} and how it relates.",
        })
        messages.append({
            "role": "assistant",
            "content": f"Here is a detailed response about that topic. " * 20
                        + f"In conclusion, this relates to {random.choice(_TOPICS)}.",
        })

    messages.append({"role": "user", "content": user_query})

    return {
        "model": "qwen3.5:35b",
        "messages": messages,
        "tools": tools,
        "stream": True,
        "options": {"temperature": 0.7},
    }


def make_lean_payload(
    user_query: str = "Where do I live?",
    context: str = "",
) -> dict:
    """Generate a Recall-style lean payload (~500-1500 tokens)."""
    from sieve.pipeline import LEAN_SYSTEM_PROMPT, RECALL_TOOL

    messages = [{"role": "system", "content": LEAN_SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": user_query})

    return {
        "model": "qwen3.5:35b",
        "messages": messages,
        "tools": [RECALL_TOOL],
        "stream": True,
    }
