"""
Knowledge Graph Extraction Prompts - English Version

Prompts for extracting entity-relation triples, event-entity links,
and event-event relations from memory atomic facts.
Adapted from AutoSchemaKG's TRIPLE_INSTRUCTIONS for the memory domain.
"""

KG_SYSTEM_PROMPT = (
    "You are a helpful assistant who always responds in a valid JSON array. "
    "No explanation, no markdown, just the JSON array."
)

KG_ENTITY_RELATION_PROMPT = """Given a list of atomic facts from a conversation memory, extract all important entities and the relations between them.

Rules:
- Entities must be specific nouns (people names, places, organizations, objects, concepts). Exclude pronouns.
- Relations should be concise verbs or verb phrases capturing the connection.
- Do not repeat information from head/tail entities in the relation.
- Each triple must be self-contained and meaningful.

You must **strictly output in the following JSON format**:
[
    {
        "Head": "{a specific noun}",
        "Relation": "{a verb or verb phrase}",
        "Tail": "{a specific noun}"
    }
]

If no meaningful entity relations can be extracted, return an empty array: []

Here are the atomic facts:
"""

KG_EVENT_ENTITY_PROMPT = """Given a list of atomic facts from a conversation memory, identify the events and their participating entities.

Rules:
- Each event should be a single independent sentence describing what happened.
- List all entities (people, places, objects) that participated in each event.
- Entities must be specific nouns, not pronouns.
- Do not use ellipses.

You must **strictly output in the following JSON format**:
[
    {
        "Event": "{a simple sentence describing an event}",
        "Entity": ["entity 1", "entity 2"]
    }
]

If no meaningful events can be extracted, return an empty array: []

Here are the atomic facts:
"""

KG_EVENT_RELATION_PROMPT = """Given a list of atomic facts from a conversation memory, identify temporal and causal relationships between events.

Rules:
- Each event should be a single independent sentence.
- Relation types: "before", "after", "at the same time", "because", "as a result"
- Each triple must be specific, meaningful, and able to stand alone.
- Do not use ellipses.

You must **strictly output in the following JSON format**:
[
    {
        "Head": "{a simple sentence describing event 1}",
        "Relation": "{temporal or causal relation}",
        "Tail": "{a simple sentence describing event 2}"
    }
]

If no meaningful event relations can be extracted, return an empty array: []

Here are the atomic facts:
"""
