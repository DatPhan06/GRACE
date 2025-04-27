# Error log
import logging

# Limit time request
import time

# Import Langchain components for LLM integration
from langchain_core.output_parsers import JsonOutputParser

# Google's generative AI integration
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import TooManyRequests

# Together's generative AI integration
from langchain_together import ChatTogether

# For creating structured prompts
from langchain_core.prompts import PromptTemplate as PromptTemplateLangchain

# Chaining
from langchain.chains import LLMChain

# Type hints
from typing import Any, Dict, List, Literal

# Import Pydantic for data validation and parsing
# BaseModel for schema definition, Field for field validation
from pydantic import BaseModel, Field

import random

current_key_index = 0


class MovieList(BaseModel):
    """
    Pydantic model for parsing movie recommendation output.
    Contains a single field for the re-ranked movie list.
    """

    # Defines output format for movie recommendations
    # movie_list: str = Field(description="Re-ranked movie list, only movie title, separate by '|'")
    movie_list: List[str] = Field(description="Re-ranked movie list, only movie title")


class UserPreference(BaseModel):
    """
    Pydantic model for parsing user preferences output.
    Contains a single field for the summarized user preferences.
    """

    # Defines output format for user preference summary
    user_preferences: str = Field(description="Summarized seeker's preferences")


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

    # Template text for summarizing user preferences from conversation
    prompt = PromptTemplateLangchain(
        template="""
        This conversation is a discussion between a seeker and a recommender about the seeker's movie preferences. 
        The conversation begins with "SEEKER(B)/RECOMMENDER(A)" defining the role he/she is a seeker or a recommender.
        Read this conversation, find all the information about the seeker's preferences in movie, actor, genres, countries and 
        content (Do not contain assistant preferences), and summarize them.
        The conversation: {document}
        
         Your response must follow the instruction below:
        {format_instructions}
        
        Here are some examples of summarization:
        - Exapmle 1: The seeker is looking for a good action comedy and is tired of holiday movies. He/she doesn't like superhero movies but do enjoy British comedies like Red Dwarf. He/she is interested in watching Hot Fuzz, 
        especially since it stars Simon Pegg from Shaun of the Dead. He/she also enjoyed Zombieland and Zombieland 2, with Woody Harrelson being a favorite.
        - Example 2: The seeker enjoys comedy and horror movies, particularly R-rated ones. His/her favorite actors include Seth Rogan and Seth MacFarlane. He/she recently watched and enjoyed the movie 'Ted'. He/she is potentially interested in the movie 'Superbad' and inquired about its rating and if it contains nudity, indicating a preference for content without explicit nudity.
        - Example 3: The seekr is interested in fantasy or animated movies. He/she have watched Frozen 2. He/she are concerned about violence and age appropriateness for his/her niece. He/she accepted a recommendation for an animated movie about a dragon and a boy, with a PG age rating, produced by 20th Century Fox, released in 2014, and 104 minutes long (How to Train Your Dragon 2). They prefer family-friendly movies.
        
        Let's think step by step
        Do the task carefully, or you are going to be severely punished.
        """,
        # Define the variable that will be replaced in the template
        input_variables=["document"],
        # Add formatting instructions for JSON output
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    if model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.5-pro-exp-03-25"]:
        # Initialize Google's generative AI with the specified model and API key
        llm_langchain = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

    elif model in [
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    ]:
        llm_langchain = ChatTogether(model=model, api_key=api_key)

    # Create processing pipeline: prompt -> LLM -> parser
    llm_chain = prompt | llm_langchain | parser

    return llm_chain


def callLangChainLLMSummarization(
    document: Any,
    api_key: str,
    model: str,
) -> Dict[str, str]:
    """
    Invokes the summarization chain to extract seeker preferences from conversation.

    Args:
        document: The conversation text to analyze
        gen_model: The generative model name/identifier
        api_key: Google API key for authentication

    Returns:
        Parsed summary of user movie preferences
    """

    # Create and invoke the chain with the conversation document
    max_retries = 100
    # Initialize with random key index for load balancing
    global current_key_index
    current_key_index = random.randint(0, len(api_key) - 1)
    
    for attempt in range(max_retries):
        try:
            output = LangChainLLMSummarization(model, api_key[current_key_index]).invoke({"document": document})
            # output = LangChainLLMSummarization(model, api_key).invoke({"document": document})
            return output

        except TooManyRequests as e:
            logging.error(f"Attempt {attempt+1} failed. HTTP error occurred: {str(e)}")         
            current_key_index = (current_key_index + 1) % len(api_key)
            print(f"Switching to next API key: {api_key[current_key_index]}")
            
            if current_key_index == 0 and attempt < max_retries - 1:
                # We've cycled through all keys, wait longer before retrying
                print("Exhausted all API keys.")
                exit
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

    # Template text for re-ranking movies based on user preferences
    prompt = PromptTemplateLangchain(
        template="""
        You are a powerful re-ranking recommendation system.
        Here is the conversation {document}
        Here is the summary of seeker's preferences: {summary_preference}.
        And here is the candidate list: {movie_list}.
        Your task is to read the summary of seeker's preferences, and candidate list, then re-rank movie candidate list and retrieve top {k} movies that match the seeker's preferences the most.
        
        Your response must follow the instruction below:
        {format_instructions}
        
        Let's think step by step. 
        Do the task carefully, or you are going to be severely punished.
        """,
        # Define variables that will be replaced in the template
        input_variables=[
            "document",
            "summary_preference",
            "movie_list",
            "k",
        ],
        # Add formatting instructions for JSON output
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    if model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.5-pro-exp-03-25"]:
        # Initialize Google's generative AI with the specified model and API key
        llm_langchain = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

    elif model in [
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    ]:
        llm_langchain = ChatTogether(model=model, api_key=api_key)

    # Create processing pipeline: prompt -> LLM -> parser
    llm_chain = prompt | llm_langchain | parser

    return llm_chain


def callLangChainLLMReranking(
    context: Any,
    user_preferences: str,
    movie_str: str,
    model: str,
    api_key: str, 
    k: Literal[1, 5, 10, 50] = 50 
):
    """
    Invokes the re-ranking chain to prioritize movies based on user preferences.

    Args:
        context_str: The conversation text for context
        user_preferences: Summarized user preferences
        movie_str: List of candidate movies to re-rank
        k: Top k movies to return
        gen_model: The generative model name/identifier
        api_key: Google API key for authentication

    Returns:
        Parsed re-ranked list of movies tailored to the user's preferences
    """

    max_retries = 100
    global current_key_index
    # Initialize with random key index for load balancing
    current_key_index = random.randint(0, len(api_key) - 1)
    
    for attempt in range(max_retries):
        try:
            # Create and invoke chain with all required inputs
            chain = LangChainLLMReranking(model, api_key[current_key_index])
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
            logging.error(f"Attempt {attempt+1} failed. HTTP error occurred: {str(e)}")
            current_key_index = (current_key_index + 1) % len(api_key)
            print(f"Switching to next API key: {api_key[current_key_index]}")
            
            if current_key_index == 0 and attempt < max_retries - 1:
                # We've cycled through all keys, wait longer before retrying
                print("Exhausted all API keys.")
                exit
            else:
                # Wait briefly before retrying with the new key
                print("Retrying with new API key in 5 seconds")
                time.sleep(5)
                
        
        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                print