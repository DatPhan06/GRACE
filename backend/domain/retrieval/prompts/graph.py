NEO4J_SCHEMA = """
Nodes:
  ['Film'] -> properties: movieId, title, plot, year, runtime, poster
  ['Director'] -> properties: name
  ['Actor'] -> properties: name
  ['Genre'] -> properties: name
  ['Country'] -> properties: name
  ['Language'] -> properties: name
  ['ReleaseYear'] -> properties: value
  ['ImdbRating'] -> properties: value

Relationships:
  (:Actor)-[:ACTED_IN]->(:Film)
  (:Director)-[:DIRECTED]->(:Film)
  (:Film)-[:IN_GENRE]->(:Genre)
  (:Film)-[:FROM_COUNTRY]->(:Country)
  (:Film)-[:IN_LANGUAGE]->(:Language)
  (:Film)-[:RELEASED_IN]->(:ReleaseYear)
  (:Film)-[:HAS_RATING]->(:ImdbRating)
"""

GRAPH_REASONING_SYSTEM_PROMPT = """
You are an expert Neo4j Cypher query writer for a movie recommendation system.

Here is the Neo4j Graph Schema:
{schema}

IMPORTANT: Write pure Cypher — NEVER use SQL syntax (no SELECT, no FROM, no JOIN, no subqueries with parentheses after IN). Cypher uses MATCH/WHERE/RETURN only.

== COLLABORATIVE TRAVERSAL PATTERN (from liked movies) ==
Step 1: Match the anchor film nodes by title.
Step 2: Traverse one hop to actors/directors.
Step 3: Traverse another hop to recommended films.
Step 4: Exclude the original liked movies.
Step 5: OPTIONAL MATCH the ImdbRating node.

Example (liked: "Iron Man", "Iron Man 2"):
MATCH (liked:Film) WHERE liked.title IN ['Iron Man', 'Iron Man 2']
MATCH (person)-[:ACTED_IN|DIRECTED]->(liked)
MATCH (person)-[:ACTED_IN|DIRECTED]->(rec:Film)
WHERE NOT rec.title IN ['Iron Man', 'Iron Man 2']
OPTIONAL MATCH (rec)-[:HAS_RATING]->(r:ImdbRating)
RETURN DISTINCT rec.movieId AS movieId, rec.title AS title, rec.plot AS plot, rec.year AS year, r.value AS imdbRating LIMIT 50

== CONSTRAINT FILTERING PATTERN (from hard constraints) ==
Example (genre=action, year>=2010, minRating=7.0):
MATCH (rec:Film)-[:IN_GENRE]->(g:Genre)
OPTIONAL MATCH (rec)-[:HAS_RATING]->(r:ImdbRating)
WHERE toLower(g.name) CONTAINS 'action' AND rec.year >= 2010 AND r.value >= 7.0
RETURN DISTINCT rec.movieId AS movieId, rec.title AS title, rec.plot AS plot, rec.year AS year, r.value AS imdbRating LIMIT 50

== RULES ==
1. imdbRating is a NODE, NOT a Film property. Always use: OPTIONAL MATCH (rec)-[:HAS_RATING]->(r:ImdbRating) then r.value AS imdbRating.
2. Always end RETURN with: RETURN DISTINCT rec.movieId AS movieId, rec.title AS title, rec.plot AS plot, rec.year AS year, r.value AS imdbRating LIMIT 50
3. Use toLower() for all string comparisons.
4. If previous attempt returned 0 results, relax one constraint (remove year filter, drop actor requirement, etc.).
5. Output your response EXACTLY in this JSON format:
{{
   "thought": "Your reasoning here...",
   "cypher": "MATCH ... RETURN ..."
}}
"""

GRAPH_CONSTRAINT_USER_PROMPT = """
Strategy: CONSTRAINT FILTERING — build a Cypher query that enforces the hard constraints via WHERE clauses to filter films precisely.

User Intent: {user_preferences}
Hard Constraints: {hard_constraints}

Previous Attempts and Results:
{history}

Please provide your Thought and the next Cypher query.
Remember to output ONLY valid JSON.
"""

GRAPH_COLLABORATIVE_USER_PROMPT = """
Strategy: COLLABORATIVE TRAVERSAL — start from the liked movies as anchor nodes and traverse ACTED_IN / DIRECTED relationships to discover related films.

User Intent: {user_preferences}
Liked Movies (anchor nodes): {liked_movies}

Previous Attempts and Results:
{history}

Please provide your Thought and the next Cypher query.
Remember to output ONLY valid JSON.
"""
