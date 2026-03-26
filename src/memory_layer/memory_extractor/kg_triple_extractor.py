"""
Knowledge Graph Triple Extractor

Extracts entity-relation triples, event-entity links, and event-event relations
from EventLog atomic facts. Adapted from AutoSchemaKG's triple extraction pipeline.
"""

import json
import re
from typing import List, Optional

from api_specs.kg_types import EventEntity, Triple, TripleExtractionResult
from core.observation.logger import get_logger
from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.prompts import get_prompt_by

logger = get_logger(__name__)


class KGTripleExtractor:
    """Extracts structured triples from atomic facts for knowledge graph construction."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.system_prompt = get_prompt_by("KG_SYSTEM_PROMPT")
        self.entity_relation_prompt = get_prompt_by("KG_ENTITY_RELATION_PROMPT")
        self.event_entity_prompt = get_prompt_by("KG_EVENT_ENTITY_PROMPT")
        self.event_relation_prompt = get_prompt_by("KG_EVENT_RELATION_PROMPT")

    def _parse_json_array(self, response: str) -> List[dict]:
        """Parse LLM response into a JSON array, with fallback strategies."""
        # Try code block extraction
        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            if end > start:
                try:
                    return json.loads(response[start:end].strip())
                except json.JSONDecodeError:
                    pass

        if '```' in response:
            start = response.find('```') + 3
            newline = response.find('\n', start)
            if newline > start:
                start = newline + 1
            end = response.find('```', start)
            if end > start:
                try:
                    return json.loads(response[start:end].strip())
                except json.JSONDecodeError:
                    pass

        # Try extracting JSON array directly
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try full response
        try:
            result = json.loads(response.strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        return []

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with system prompt prepended."""
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        return await self.llm_provider.generate(full_prompt)

    async def extract_entity_relations(
        self, atomic_facts: List[str]
    ) -> List[Triple]:
        """Extract entity-relation triples from atomic facts."""
        facts_text = "\n".join(f"- {fact}" for fact in atomic_facts)
        prompt = self.entity_relation_prompt + facts_text

        try:
            response = await self._call_llm(prompt)
            items = self._parse_json_array(response)
            triples = []
            for item in items:
                if all(k in item for k in ("Head", "Relation", "Tail")):
                    triples.append(
                        Triple(
                            head=str(item["Head"]).strip(),
                            relation=str(item["Relation"]).strip(),
                            tail=str(item["Tail"]).strip(),
                        )
                    )
            return triples
        except Exception as e:
            logger.warning(f"entity_relation extraction failed: {e}")
            return []

    async def extract_event_entities(
        self, atomic_facts: List[str]
    ) -> List[EventEntity]:
        """Extract event-entity links from atomic facts."""
        facts_text = "\n".join(f"- {fact}" for fact in atomic_facts)
        prompt = self.event_entity_prompt + facts_text

        try:
            response = await self._call_llm(prompt)
            items = self._parse_json_array(response)
            results = []
            for item in items:
                if "Event" in item and "Entity" in item:
                    entities = item["Entity"]
                    if isinstance(entities, str):
                        entities = [entities]
                    results.append(
                        EventEntity(
                            event=str(item["Event"]).strip(),
                            entities=[str(e).strip() for e in entities],
                        )
                    )
            return results
        except Exception as e:
            logger.warning(f"event_entity extraction failed: {e}")
            return []

    async def extract_event_relations(
        self, atomic_facts: List[str]
    ) -> List[Triple]:
        """Extract temporal/causal relations between events."""
        facts_text = "\n".join(f"- {fact}" for fact in atomic_facts)
        prompt = self.event_relation_prompt + facts_text

        try:
            response = await self._call_llm(prompt)
            items = self._parse_json_array(response)
            triples = []
            for item in items:
                if all(k in item for k in ("Head", "Relation", "Tail")):
                    triples.append(
                        Triple(
                            head=str(item["Head"]).strip(),
                            relation=str(item["Relation"]).strip(),
                            tail=str(item["Tail"]).strip(),
                        )
                    )
            return triples
        except Exception as e:
            logger.warning(f"event_relation extraction failed: {e}")
            return []

    async def extract_all(
        self, atomic_facts: List[str], batch_size: int = 15,
    ) -> TripleExtractionResult:
        """Run all three extraction stages, batching long fact lists."""
        if not atomic_facts:
            return TripleExtractionResult()

        import asyncio

        # Split into batches to avoid prompt-too-long errors
        batches = [
            atomic_facts[i:i + batch_size]
            for i in range(0, len(atomic_facts), batch_size)
        ]

        all_entity_rels, all_event_ents, all_event_rels = [], [], []
        for batch in batches:
            er, ee, evr = await asyncio.gather(
                self.extract_entity_relations(batch),
                self.extract_event_entities(batch),
                self.extract_event_relations(batch),
            )
            all_entity_rels.extend(er)
            all_event_ents.extend(ee)
            all_event_rels.extend(evr)

        result = TripleExtractionResult(
            entity_relations=all_entity_rels,
            event_entities=all_event_ents,
            event_relations=all_event_rels,
        )
        logger.info(
            f"KG extraction: {len(all_entity_rels)} entity_relations, "
            f"{len(all_event_ents)} event_entities, {len(all_event_rels)} event_relations"
        )
        return result
