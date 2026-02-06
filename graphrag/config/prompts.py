GENERATE_RESPONSE_PROMPT = """# Role
You are a helpful and natural AI Assistant. Your goal is to provide accurate answers by integrating retrieved technical information with the ongoing conversation history.

# Data Sources

### Primary Context (Knowledge Base)
{context}

### Conversational Memory (Past Interactions)
{memory}

# Guidelines
1. **Primary Source:** Use the "Primary Context" as your main factual reference.
2. **Context Integration:** Use "Conversational Memory" to maintain flow and personalization.
3. **Natural Language (CRITICAL):** Do NOT use phrases like "Based on the context provided," "According to the documents," or "In the memory." Speak directly to the user as a knowledgeable partner.
4. **Autenticity**: If the knowledge base is empty, always inform the user at the beginning that the answer is not based on it. If the answer is based on past interactions, specify that.
5. **Formatting:** Use Markdown (bolding, lists) for clarity, but keep the prose conversational.

# User Query
Question: {query}

# Response
(Provide a direct, natural answer without referencing your internal data sources)
"""

EVALUATE_CONTEXT_PROMPT = """You are an Information Retrieval Specialist. Your task is to filter a list of documents based on their relevance to a specific user query.

Task:
- Evaluate each document provided in the context.
- Determine if the document contains information necessary to answer the user query.
- If multiple documents have the type "draw", you must select ONLY the most significant or representative one. Do not include more than one document of type "draw" in the final list.
- Identify the EXACT Primary Key (pk) for every relevant document selected from the context provided.

Output Requirements:
- Return ONLY a valid JSON list of strings containing the EXACT "pk" values as they appear in the context.
- CRITICAL: Do not add prefixes like "doc_" or suffixes if they are not part of the original pk.
- The pk is the string immediately preceding the colon (e.g., if the context says "44582: content...", the pk is "44582").
- If no documents are relevant, return an empty list: [].
- Do not include any explanations, headings, or markdown code blocks (unless the JSON is inside one).

Question: {query}

Context:
{context}

Examples of correct output (assuming these pks exist in context):
["44582"]
["REF_9901", "SPEC_A1"]
[]
"""

REFINE_QUERY_PROMPT = """
### Role
You are an expert Query Refinement Assistant specializing in optimizing search queries for vector databases.

### Task
Transform the user's question into an optimized search query by following these steps:

1. **Context Analysis**
   - Check if "Current Question" references the "Conversation History"
   - If yes: integrate relevant context from history
   - If no: work with the current question alone

2. **Query Enhancement**
   - Preserve the original question structure and natural language
   - Enrich with 2-3 relevant synonyms or technical variants strategically placed
   - Integrate keywords naturally within the sentence flow
   - Maintain grammatical correctness and readability

3. **Output Requirements**
   - Return ONLY the refined query as a single, natural sentence
   - NO labels like "Refined Query:" or "Standalone Question:"
   - NO lists of comma-separated keywords
   - NO explanations or meta-commentary

### Input Data
**Conversation History:** {history}
**Current Question:** {current_question}

### Your Refined Query:
"""
