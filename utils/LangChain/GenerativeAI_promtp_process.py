# Error log
import logging

# Limit time request
import time

# Import Langchain components for LLM integration
from langchain_core.output_parsers import JsonOutputParser

# Google's generative AI integration
from langchain_google_genai import ChatGoogleGenerativeAI

# Together's generative AI integration
from langchain_together import ChatTogether

# For creating structured prompts
from langchain_core.prompts import PromptTemplate as PromptTemplateLangchain

# Chaining
from langchain.chains import LLMChain

# Type hints
from typing import Dict, List, Literal

# Import Pydantic for data validation and parsing
# BaseModel for schema definition, Field for field validation
from pydantic import BaseModel, Field


class parser_recommendation(BaseModel):
    """
    Pydantic model for parsing movie recommendation output.
    Contains a single field for the re-ranked movie list.
    """

    # Defines output format for movie recommendations
    movie_list: str = Field(description="Re-ranked movie list, only movie title, separate by '|'")


class parser_user_preferences(BaseModel):
    """
    Pydantic model for parsing user preferences output.
    Contains a single field for the summarized user preferences.
    """

    # Defines output format for user preference summary
    user_preferences: str = Field(description="User summarize preferences")


# ------------------------------ SUMMARIZE CONVERSATION ------------------------------
def chain_llm_summarize(gen_model: str, api_key: str) -> LLMChain:
    """
    Creates a language model chain for summarizing user preferences from conversation.

    Args:
        gen_model: The generative model name/identifier
        api_key: Google API key for authentication

    Returns:
        A langchain chain that processes conversation text and outputs summarized user preferences
    """
    # Initialize JSON parser with our preferences schema
    parser = JsonOutputParser(pydantic_object=parser_user_preferences)

    # Template text for summarizing user preferences from conversation
    prompt = PromptTemplateLangchain(
        template="""
        This conversation is a discussion between a seeker and a recommender about the seeker's movie preferences.
        The conversation begins with "SEEKER/RECOMMENDER" defining the role he/she is a seeker or a recommender.
        Read this conversation, find all the information about the seeker's preferences in movie, actor, genres, countries and
        content (Do not contain assistant preferences), and summarize them.
        The conversation: {document}
        Here are some examples of summarization:
        - Exapmle 1: The seeker is looking for a good action comedy and is tired of holiday movies. He/she doesn't like superhero movies but do enjoy British comedies like Red Dwarf. He/she is interested in watching Hot Fuzz,
        especially since it stars Simon Pegg from Shaun of the Dead. He/she also enjoyed Zombieland and Zombieland 2, with Woody Harrelson being a favorite.
        - Example 2: The seeker enjoys comedy and horror movies, particularly R-rated ones. His/her favorite actors include Seth Rogan and Seth MacFarlane. He/she recently watched and enjoyed the movie 'Ted'. He/she is potentially interested in the movie 'Superbad' and inquired about its rating and if it contains nudity, indicating a preference for content without explicit nudity.
        - Example 3: The seekr is interested in fantasy or animated movies. He/she have watched Frozen 2. He/she are concerned about violence and age appropriateness for his/her niece. He/she accepted a recommendation for an animated movie about a dragon and a boy, with a PG age rating, produced by 20th Century Fox, released in 2014, and 104 minutes long (How to Train Your Dragon 2). They prefer family-friendly movies.
        {format_instructions}
        The summarization must not contain anything directly related to the item which is recommended.
        Item recommeded: {item}
        Do the task carefully, or you are going to be severely punished.
        """,
        # Define the variable that will be replaced in the template
        input_variables=["document", 'item'],
        # Add formatting instructions for JSON output
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Initialize Google's generative AI with the specified model and API key
    llm_langchain = ChatGoogleGenerativeAI(model=gen_model, google_api_key=api_key, max_output_tokens=10000)

    # Create processing pipeline: prompt -> LLM -> parser
    llm_chain = prompt | llm_langchain | parser

    return llm_chain


def call_llm_summarize(document: List[str], gen_model: str, api_key: str, recommended_item: List[str]) -> Dict[str, str]:
    """
    Invokes the summarization chain to extract user preferences from conversation.

    Args:
        document: The conversation text to analyze
        gen_model: The generative model name/identifier
        api_key: Google API key for authentication

    Returns:
        Parsed summary of user movie preferences
    """

    # Create and invoke the chain with the conversation document
    time.sleep(10)
    output = chain_llm_summarize(gen_model, api_key).invoke({
        "document": document,
        'item': recommended_item
    })

    return output
