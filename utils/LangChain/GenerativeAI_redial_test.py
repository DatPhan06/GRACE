# Error log
import logging

# Limit time request
import time

# Import Langchain components for LLM integration
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# Google's generative AI integration is now accessed via infra helpers
from infra.llm import create_gemini_langchain_llm
from google.api_core.exceptions import TooManyRequests

# Together's generative AI integration
from langchain_together import ChatTogether

# For creating structured prompts
from langchain_core.prompts import ChatPromptTemplate

# Chaining
from langchain.chains import LLMChain

# Type hints
from typing import Any, Dict, List, Literal

# Import Pydantic for data validation and parsing
# BaseModel for schema definition, Field for field validation
from pydantic import BaseModel, Field
from pydantic import SecretStr

import random
import json
import re

current_index_key = 0
_movie_data_cache = None


def _load_movie_data(movie_data_path: str) -> dict:
    """Loads movie data from a JSONL file and caches it in a title-keyed dictionary."""
    global _movie_data_cache
    if _movie_data_cache is None:
        _movie_data_cache = {}
        try:
            with open(movie_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        movie = json.loads(line)
                        title = movie.get("title")
                        if title:
                            # Use a normalized title as a key for case-insensitive matching
                            _movie_data_cache[title.lower().strip()] = movie
                    except json.JSONDecodeError:
                        logging.warning(
                            f"Skipping malformed JSON line in {movie_data_path}")
                        continue
        except FileNotFoundError:
            logging.error(f"Movie data file not found at: {movie_data_path}")
            # Return an empty dict if file not found
            return {}
    return _movie_data_cache


def get_movie_details_as_string(movie_titles: list[str], movie_data_path: str) -> str:
    """
    Looks up movies by title, extracts key details, and formats them into a single string.
    """
    if not movie_titles:
        return "No specific movies were mentioned as liked in the conversation history."

    movie_db = _load_movie_data(movie_data_path)
    if not movie_db:
        return "Movie details database is unavailable."

    details_list = []
    for title in movie_titles:
        # Remove year from title like "Movie Title (YYYY)" for lookup
        lookup_title = re.sub(r'\s*\(\d{4}\)$', '', title).strip()
        # Find the movie using the normalized key
        movie = movie_db.get(lookup_title.lower())
        if movie:
            details = {
                "Title": movie.get("title"),
                "Year": movie.get("year"),
                "Rated": movie.get("rated"),
                "Runtime": movie.get("runtime"),
                "Genre": movie.get("genre"),
                "Director": movie.get("director"),
                "Actors": movie.get("actors"),
                "Plot": movie.get("plot"),
            }
            # Format the details into a string, filtering out empty values
            formatted_details = "\n".join(
                [f"- {key}: {value}" for key, value in details.items() if value])
            if formatted_details:
                details_list.append(
                    f"--- Movie: {movie.get('title')} ---\n{formatted_details}")

    if not details_list:
        return "No detailed information found for the liked movies mentioned."

    return "\n\n".join(details_list)


class MovieList(BaseModel):
    """
    Pydantic model for parsing movie recommendation output.
    Contains a single field for the re-ranked movie list.
    """

    # Defines output format for movie recommendations
    # movie_list: str = Field(description="Re-ranked movie list, only movie title, separate by '|'")
    movie_list: List[str] = Field(
        description="Re-ranked movie list, only movie title")


class UserPreference(BaseModel):
    """
    Pydantic model for parsing user preferences output.
    Contains a single field for the summarized user preferences.
    """

    # Defines output format for user preference summary
    user_preferences: str = Field(
        description="Summarized seeker's preferences")


def LangChainLLMSummarization(model: str, api_key: str) -> LLMChain:
    """
    Creates a language model chain for summarizing user preferences from conversation.

    Args:
        model: The generative model name/identifier
        api_key: API key for authentication

    Returns:
        A langchain chain that processes conversation text and outputs summarized user preferences
    """
    # Initialize JSON parser with our preferences schema
    parser = JsonOutputParser(pydantic_object=UserPreference)

    system_prompt_text = """<role>
You are an intelligent assistant specialized in analyzing conversations to create detailed summaries of user preferences for movies.
</role>

<instruction>
This conversation is a discussion between an 'Initiator' and a 'Respondent' about movies. One of them is looking for movie recommendations. Your task is to:
1.  Read the entire conversation carefully to identify the 'seeker' (the person looking for recommendations).
2.  Extract ALL stated preferences from the seeker. This includes likes and dislikes for movies, actors, genres, directors, countries, specific plot points, or themes.
3.  Synthesize these points into a comprehensive and detailed paragraph. The summary should be rich with detail, capturing not only what the seeker wants but also why. Include mentions of movies they have already seen as context.
4.  Pay special attention to the additional movie details provided - use this information about genres, actors, directors etc. to make your summary more complete.
</instruction>

<constraint>
- Your summary must be long and detailed, written in a narrative style as shown in the examples.
- Focus ONLY on the seeker's preferences. Do not include preferences or suggestions from the recommender.
- Let's think step-by-step to ensure no detail is missed.
- You must return ONLY a valid JSON object that matches the specified schema. DO NOT add any markdown, role names, or other extraneous text to your response.
</constraint>"""

    user_prompt_text = """<input>
The conversation: {document}

Here is additional information about movies mentioned in the conversation. Use this to make your summary more detailed and accurate:
{movie_details}
</input>

<example>
- Example 1: The seeker has expressed a strong interest in action comedies while specifically mentioning their fatigue with holiday-themed films. They have a particular affinity for British comedy, citing Red Dwarf as a favorite example, but explicitly dislike superhero movies. The seeker shows enthusiasm for Hot Fuzz, primarily due to Simon Pegg's involvement, whom they know from Shaun of the Dead. In terms of zombie comedies, they've enjoyed both Zombieland and its sequel, with special appreciation for Woody Harrelson's performances. This suggests a preference for dark humor and action-comedy blends with strong character performances.

- Example 2: The seeker demonstrates a clear preference for adult-oriented comedy and horror content, specifically mentioning their interest in R-rated films. They show strong appreciation for comedic actors, naming Seth Rogen and Seth MacFarlane as favorites. Their recent positive experience with 'Ted' indicates comfort with mature comedy, though their inquiries about 'Superbad' regarding nudity content suggests some boundaries regarding explicit content. The seeker appears to value humor but shows thoughtful consideration about content boundaries.

- Example 3: The seeker is actively seeking family-appropriate content, with a specific focus on fantasy and animation genres. Having watched Frozen 2, they're familiar with contemporary animated features but are particularly conscious about content appropriateness for their young niece. Their positive response to How to Train Your Dragon 2 (2014, 20th Century Fox, PG rating, 104 minutes) demonstrates acceptance of mild fantasy action while maintaining family-friendly standards. Their concerns about violence levels and age ratings indicate a strong priority for safe, child-appropriate entertainment that can be enjoyed together with young family members.
</example>

<output_format>
{format_instructions}
</output_format>"""

    # Template text for summarizing user preferences from conversation
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            ("user", user_prompt_text),
        ]
    )

    # Add formatting instructions for JSON output
    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions())

    if model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.5-pro-exp-03-25"]:
        # Initialize Google's generative AI with the specified model and API key via infra helper
        llm_langchain = create_gemini_langchain_llm(
            api_key=api_key,
            model_name=model,
            max_output_tokens=10000,
        )

    elif model in [
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    ]:
        llm_langchain = ChatTogether(model=model, api_key=SecretStr(api_key))

    # Create processing pipeline: prompt -> LLM -> parser
    llm_chain = prompt | llm_langchain | parser

    return llm_chain


def callLangChainLLMSummarization_redial(
    document: Any,
    liked_movies: list[str],
    movie_data_path: str,
    api_key: list[str],
    model: str,
) -> Dict[str, str]:
    """
    Invokes the summarization chain to extract seeker preferences from conversation.

    Args:
        document: The conversation text to analyze
        liked_movies: A list of movie titles that the user has liked.
        movie_data_path: The file path to the movie details JSONL file.
        model: The generative model name/identifier
        api_key: Google API key for authentication

    Returns:
        Parsed summary of user movie preferences
    """

    global current_index_key
    current_index_key = random.randint(0, len(api_key) - 1)
    key_len = len(api_key)

    # Get detailed movie information as a formatted string
    movie_details = get_movie_details_as_string(liked_movies, movie_data_path)
    # print(liked_movies)
    # print(movie_details)

    # Create and invoke the chain with the conversation document
    max_retries = 10

    for attempt in range(max_retries):
        try:
            output = LangChainLLMSummarization(model, api_key[current_index_key]).invoke(
                {"document": document, "movie_details": movie_details}
            )
            return output

        except OutputParserException as e:
            logging.warning(
                f"Attempt {attempt+1} failed with OutputParserException. Trying to clean and re-parse...")

            error_string = str(e)
            try:
                # The actual output from the LLM is often embedded in the exception string.
                # We'll try to find the start of the JSON object and parse from there.
                json_start_index = error_string.find('{')
                if json_start_index != -1:
                    json_string = error_string[json_start_index:]

                    # Manually create a parser and parse the cleaned string
                    parser = JsonOutputParser(pydantic_object=UserPreference)
                    parsed_output = parser.parse(json_string)

                    print("Successfully parsed after cleaning the malformed output.")
                    return parsed_output
                else:
                    logging.error(
                        f"Cleaning failed: No JSON object start '{{' found in the output. Full error: {error_string}")

            except Exception as parse_error:
                # If cleaning and re-parsing fails, log and retry
                logging.error(
                    f"Failed to parse after cleaning. Error: {parse_error}")

            # Fall through to retry logic if cleaning fails
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(
                    "All retries failed after attempting to clean, returning fallback response")
                return {"user_preferences": ""}

        except TooManyRequests as e:
            logging.error(
                f"Attempt {attempt+1} failed. HTTP error occurred: {str(e)}")
            current_index_key = (current_index_key + 1) % len(api_key)
            key_len -= 1
            print(
                f"Switching to next API key : #{current_index_key} ({api_key[current_index_key]})")

            if key_len == 0 and attempt < max_retries - 1:
                # We've cycled through all keys, wait longer before retrying
                print("Exhausted all API keys.")
                exit()
            else:
                # Wait briefly before retrying with the new key
                print("Retrying with new API key in 5 seconds...")
                time.sleep(5)

        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                # Fallback mechanism - return a simple structure that matches the expected format
                print("All retries failed, returning fallback response")
                return {"user_preferences": ""}
    return {"user_preferences": ""}


# ------------------- RE-RANKING OUTPUT ------------------------------
def LangChainLLMReranking(model: str, api_key: str) -> LLMChain:
    """
    Creates a language model chain for re-ranking movie recommendations based on user preferences.

    Args:
        gen_model: The generative model name/identifier
        api_key: Google API key for authentication

    Returns:
        A langchain chain that re-ranks movie candidates according to user preferences
    """

    # Initialize JSON parser with our recommendation schema
    parser = JsonOutputParser(pydantic_object=MovieList)

    # (exclude the conversation from reranking prompt)
    # This conversation is a discussion between a seeker and a recommender about the seeker's movie preferences.
    # The conversation begins with "SEEKER/RECOMMENDER" defining the role he/she is a seeker or a recommender.
    # Here is the conversation: {document}

    system_prompt_text = """<role>
You are a powerful movie re-ranking recommendation system specialized in analyzing user preferences and movie content.
</role>

<instruction>
Your task is to re-rank a given list of candidate movies based on a user's preferences and conversation context. Each movie in the candidate list includes its plot summary, which provides crucial information about the movie's content, themes, and storyline.

You need to:
1. Carefully analyze the user's stated preferences from both the conversation and the preference summary
2. Match these preferences against the plot content and themes of each candidate movie
3. Consider genre preferences, character types, storylines, themes, and any specific elements mentioned by the user
4. Return the top {k} movies that best align with the user's tastes, ranked from most to least relevant

Pay special attention to the plot summaries as they contain rich information about movie content that goes beyond just titles and genres.
</instruction>

<constraint>
- Analyze both the conversation context and preference summary for complete understanding
- Use the plot information to make informed ranking decisions based on content relevance
- Consider thematic elements, character types, and story elements that match user preferences
- Rank movies based on how well their plots align with expressed preferences
- You must return ONLY a valid JSON object that matches the specified schema
- DO NOT add any markdown, role names, or other extraneous text to your response
</constraint>"""

    user_prompt_text = """<input>
Conversation: {document}

Seeker's Preference Summary: {summary_preference}

Candidate Movies with Plot Summaries:
{movie_list}

Note: Each movie entry follows the format "Movie Title: Plot Summary". Use both the movie titles and their detailed plot summaries to understand the content, themes, and storylines when making ranking decisions.
</input>

<example_analysis>
When analyzing preferences like "I enjoy action comedies with witty dialogue" against a movie like "Super Troopers: A group of Vermont state troopers compete with a rival department...", consider how the plot indicates comedy elements, character dynamics, and thematic content that would appeal to someone seeking action-comedy with humor.
</example_analysis>

<output_format>
{format_instructions}
</output_format>"""

    # Template text for re-ranking movies based on user preferences
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            ("user", user_prompt_text),
        ]
    )

    # Add formatting instructions for JSON output
    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions())

    if model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.5-pro-exp-03-25"]:
        # Initialize Google's generative AI with the specified model and API key via infra helper
        llm_langchain = create_gemini_langchain_llm(
            api_key=api_key,
            model_name=model,
            max_output_tokens=10000,
        )

    elif model in [
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    ]:
        llm_langchain = ChatTogether(model=model, api_key=SecretStr(api_key))

    # Create processing pipeline: prompt -> LLM -> parser
    llm_chain = prompt | llm_langchain | parser

    return llm_chain


def callLangChainLLMReranking_redial(
    context: Any,
    user_preferences: str,
    movie_str: str,
    model: str,
    api_key: list[str],
    k: Literal[1, 5, 10, 50] = 50
) -> Dict[str, str]:
    """
    Invokes the re-ranking chain to prioritize movies based on user preferences.

    Args:
        context_str: The conversation text for context
        user_preferences: Summarized user preferences
        movie_str: List of candidate movies to re-rank
        k: Top k movies to return
        gen_model: The generative model name/identifier
        api_key: Google API key for authentication
        k: Top k movies to return

    Returns:
        Parsed re-ranked list of movies tailored to the user's preferences
    """

    max_retries = 100
    global current_index_key
    current_index_key = random.randint(0, len(api_key) - 1)
    key_len = len(api_key)

    for attempt in range(max_retries):
        try:
            # Create and invoke chain with all required inputs
            chain = LangChainLLMReranking(model, api_key[current_index_key])
            # chain = LangChainLLMReranking(model, api_key)
            output = chain.invoke(
                {
                    "document": context,
                    "summary_preference": user_preferences,
                    "movie_list": movie_str,
                    "k": k,
                }
            )
            return output

        except TooManyRequests as e:
            logging.error(
                f"Attempt {attempt+1} failed. HTTP error occurred: {str(e)}")
            current_index_key = (current_index_key + 1) % len(api_key)
            key_len -= 1
            print(
                f"Switching to next API key : #{current_index_key} ({api_key[current_index_key]})")

            if key_len == 0 and attempt < max_retries - 1:
                # We've cycled through all keys, wait longer before retrying
                print("Exhausted all API keys.")
                exit()
            else:
                # Wait briefly before retrying with the new key
                print("Retrying with new API key in 5 seconds")
                time.sleep(5)

        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed: {str(e)}")

            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                # Fallback mechanism - return a simple structure that matches the expected format
                print("All retries failed, returning fallback response")
                return {"user_preferences": ""}
