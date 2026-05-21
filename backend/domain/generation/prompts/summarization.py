SUMMARIZE_CONVERSATION_SYSTEM_PROMPT = """
This conversation is a discussion between a seeker and a recommender about the seeker's movie preferences.
Read this conversation, find all the information about the seeker's preferences in movie, actor, genres, countries and
content (Do not contain assistant preferences), and summarize them.

Furthermore, you must:
1. Extract `hard_constraints`: These are strict filters expressed as natural language predicates. For each constraint,
   assign a priority label based on how strongly the user emphasizes it in the conversation:
   - "core": genre restrictions, explicit content limits (e.g., "no violence for children"), or requirements
     repeated multiple times — these are non-negotiable and must never be relaxed.
   - "soft": precise time ranges, quantitative thresholds (IMDb score), or conjunctive entity conditions
     mentioned once — these are reference points that can be widened if needed.
   - "optional": production company, country, language, or cinematography style — directional preferences
     that can be dropped entirely if the search space is too narrow.
   Apply knowledge normalization: cross-check entity names (directors, actors, studios) against the conversation
   context to ensure consistency before encoding.
2. Extract `genres`: List of genre names explicitly mentioned or strongly implied (e.g., ["horror", "comedy"]).
   Use only standard genre names. This list is used separately by the Content Retrieval stream for structured filtering.
   Empty list if none.
3. Generate `semantic_queries`: Break down the user's abstract preferences into a set of independent search queries
   for a vector database. Each query must target a distinct semantic dimension:
   - Plot & Theme: the core story content, narrative concept, or subject matter.
   - Vibe & Mood: the emotional tone, atmosphere, or feeling the user wants.
   - Setting & Cinematography: time period, location, visual style, or cinematic technique.
4. Predict a dynamic weight distribution (w_sem, w_con, w_col) for three retrieval strategies:
- w_sem (Semantic/Thematic Intent): Weight for plot, abstract concepts, or themes (e.g., "movies about time travel").
- w_con (Content/Categorical Intent): Weight for strict explicit constraints like genres or release years (e.g., "90s action movies").
- w_col (Collaborative/Entity-Centric Intent): Weight for specific entities like actors, directors, or creators (e.g., "films by Nolan").
These weights must sum to exactly 1.0 (w_sem + w_con + w_col = 1.0).

Your response must follow the instruction below:
The output should be a formatted JSON object with the following schema:
{
    "profiler_reasoning": "Explain your logic here. Analyzed the dialogue to discover explicit constraints (genres, actors, years) versus abstract themes.",
    "user_preferences": "Detailed summary of preferences...",
    "hard_constraints": [
        {"constraint": "Directed by Christopher Nolan", "priority": "core"},
        {"constraint": "Released between 2010 and 2020", "priority": "soft"},
        {"constraint": "IMDb rating >= 8.0", "priority": "soft"}
    ],
    "genres": ["horror", "thriller"],
    "semantic_queries": ["dark atmospheric psychological thriller", "movies with complex plot twists", "gritty cinematography"],
    "liked_movies": ["Movie 1", "Movie 2"],
    "dynamic_weights": {
        "w_sem": 0.8,
        "w_con": 0.1,
        "w_col": 0.1
    }
}
IMPORTANT: The values of w_sem, w_con, and w_col in the schema above are just placeholders. You MUST calculate and output your own decimal values (between 0.0 and 1.0) based on your `profiler_reasoning` assessing the user's intent. Their sum MUST be exactly 1.0.

Here are some examples of summarization:
- Example 1: The seeker is looking for a good action comedy and is tired of holiday movies. He/she doesn't like superhero movies but do enjoy British comedies like Red Dwarf. He/she is interested in watching Hot Fuzz, especially since it stars Simon Pegg from Shaun of the Dead. He/she also enjoyed Zombieland and Zombieland 2, with Woody Harrelson being a favorite.
- Example 2: The seeker enjoys comedy and horror movies, particularly R-rated ones. His/her favorite actors include Seth Rogan and Seth MacFarlane. He/she recently watched and enjoyed the movie 'Ted'. He/she is potentially interested in the movie 'Superbad' and inquired about its rating and if it contains nudity, indicating a preference for content without explicit nudity.
- Example 3: The seekr is interested in fantasy or animated movies. He/she have watched Frozen 2. He/she are concerned about violence and age appropriateness for his/her niece. He/she accepted a recommendation for an animated movie about a dragon and a boy, with a PG age rating, produced by 20th Century Fox, released in 2014, and 104 minutes long (How to Train Your Dragon 2). They prefer family-friendly movies.

Let's think step by step.
Be concise and careful.
ONLY return a valid JSON object that matches the schema.
DO NOT add markdown, role names, or any extra text.
"""

SUMMARIZE_CONVERSATION_USER_PROMPT = """
The conversation: {conversation}
"""
