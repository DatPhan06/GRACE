import json
from typing import Dict, List, Literal

import pandas as pd

import yaml

from tqdm import tqdm

import json
from tqdm import tqdm
from typing import Dict, List

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

import logging
import time


with open("config.yaml", "r") as cf:
    config = yaml.safe_load(cf)


MODEL = config["GeminiModel"]["2.0_flash_thinking_exp"]
API_KEY = config["APIKey"]["GOOGLE_API_KEY_1"]


class Target(BaseModel):
    """Pydantic model for target"""
    target: List[str] = Field(
        description="The movies that the seeker **ACCEPTED** from the recommender's recommendations"
    )


class Masked_dialog(BaseModel):
    """Pydantic model for masked dialog"""
    masked_dialog: List[str] = Field(description="The dialog with all target informations that has been masked")


class Inspired:
    def __init__(self):
        # Paths from the config
        self.train_dialog = config["InspiredDataPath"]["raw"]["dialog"]["train"]
        self.test_dialog = config["InspiredDataPath"]["raw"]["dialog"]["test"]
        self.processed_train = config["InspiredDataPath"]["processed"]["dialog"]["train"]
        self.processed_test = config["InspiredDataPath"]["processed"]['dialog']["test"]
        
        self.preprocessed = []
        transformed = self.transform()
        merged = self.merge(transformed)
        for dialog in tqdm(merged):
            context = dialog['dialog']
            target = self.callLangChainExtractTarget(conversation=context)['target']
            masked_dialog = self.callLangChainMasking(conversation=context, target=target)['masked_dialog']
            
            dialog['target'] = target
            dialog['masked_dialog'] = masked_dialog
            
            self.preprocessed.append(dialog)
            
        self.save_data(self.preprocessed)
            
    def transform(self, file: Literal["train", "test"] = "train"):
        if file == "train":
            file_path = self.train_dialog
        elif file == "test":
            file_path = self.test_dialog

        df = pd.read_csv(file_path, sep="\t")

        # Group the data by dialog_id to organize utterances by conversation
        dialog_groups = df.groupby("dialog_id", sort=False)

        # Initialize the result list to store processed dialogs
        result = []

        # Process each dialog group separately
        for dialog_id, group in tqdm(dialog_groups):

            # Create a dictionary for this specific dialog
            dialog_dict = {dialog_id: []}

            # Sort group by utt_id to ensure utterances are in correct chronological order
            group = group.sort_values("utt_id")

            # Process each utterance in the current dialog group
            for _, row in tqdm(group.iterrows()):

                # Create utterance dict with all columns from the row
                utterance = {}
                for column in row.index:

                    # Skip dialog_id as it's already the dictionary key
                    if column == "dialog_id":
                        continue

                    # Handle special dictionary fields - try to parse JSON strings
                    if column in [
                        "movie_dict",
                        "genre_dict",
                        "actor_dict",
                        "director_dict",
                        "others_dict",
                    ]:
                        try:
                            # Check if value is non-empty string
                            if isinstance(row[column], str) and row[column].strip():
                                # Using eval to parse Python dictionary literals from strings
                                utterance[column] = eval(row[column])

                            else:
                                # Handle empty or non-string values
                                # Keep original if not a string
                                utterance[column] = row[column]

                                # Convert NaN to empty dict
                                utterance[column] = {} if pd.isna(row[column]) else row[column]

                        except:
                            # If parsing fails, keep the original value
                            # Preserve original value if eval fails
                            utterance[column] = row[column]

                    else:
                        # Handle regular fields, replacing NaN values with empty strings
                        utterance[column] = "" if pd.isna(row[column]) else row[column]

                # Add the processed utterance to this dialog's list
                # Add utterance to the dialog's list of utterances
                dialog_dict[dialog_id].append(utterance)

            # Add the complete dialog dictionary to the result list
            result.append(dialog_dict)

            # with open('dataset\preprocessed_data\INSPIRED\dialog_data\dialog_train_data.json', 'w', encoding='utf-8') as f:
            #     json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def merge(self, dialog_list: List[Dict]):
        merged = []
        for dialog_data in tqdm(dialog_list):
            conv_id = list(dialog_data.keys())[0]

            # Initialize list to store formatted dialogue
            context = []
            completed = {}

            # Track the previous speaker to handle consecutive messages
            last_role = ""

            # Process each message in the dialogue
            for message in dialog_data[conv_id]:
                # Extract speaker role
                speaker = message["speaker"]

                # Extract message text
                text = message["text"]

                # Format sociable strategy tag
                # Get strategy label
                sociable_strategy = message["expert_label"]
                if sociable_strategy == "transparency":
                    sociable_strategy = "<offer_help> "
                else:
                    # Wrap other strategies in angle brackets
                    sociable_strategy = "<" + sociable_strategy + ">"

                if sociable_strategy == "<>":
                    sociable_strategy = ""

                # Check if this message is from the same speaker as the previous one
                if last_role == speaker:
                    # If same speaker as previous message, append to the last message
                    context[-1] += f" {sociable_strategy} {text}"

                else:
                    # If new speaker, create a new message entry
                    message_str = f"{speaker}: {sociable_strategy} {text}"

                    # Add to context list
                    context.append(message_str)

                # Update last speaker for next iteration
                # Remember current speaker for next message
                last_role = speaker

            completed["conv_id"] = conv_id
            completed["dialog"] = context

            merged.append(completed)

        return merged

    def LangChainExtractTarget(self, model: str, api_key: str) -> LLMChain:
        parser = JsonOutputParser(pydantic_object=Target)

        prompt = """
        Conversation: {conversation}
        You are given a conversation between a seeker and a recommender discussing movie preferences, 
        The conversation begins with "SEEKER/RECOMMENDER" defining the role he/she is a seeker or a recommender.
        Read the conversation carefully, then analyze the conversation and extract **EXACTLY** the movies that the seeker **ACCEPTED** from the recommender's recommendations.

        A. **Inclusion Criteria**:
        - The movie must be recommended by the recommender.
        - The seeker must confirm acceptance via:
            1. Direct Acceptance Indicators:
            "I accept your recommendation."
            "I will watch it."
            "I'll check it out."
            "I'm going to see it." / "I'm going to try to see [movie] soon."
            "I'll be watching it."
            "I'll look for it."
            "I'll buy the blu-ray."
            "I will definitely watch it soon."
            "I am definitely going to check it out."
            "I will definitely take a look."
            "I'll def check it out."
            "I will wait for this movie."
            "I think I'll try and watch those."
            "I will take your recommendation and watch that movie."
            "Gonna watch this one today."
            "I will watch it today."
            "Yes, I want to watch that movie."
            "Yes, I like to watch this movie."
            "Yes, I would like to watch this movie trailer, Thank you."
            "I accept that recommendation!"
            "I accept this as a recommendation thank you."
            "I will accept your recommendation."
            
            2. Expressions of Intent or Planning:
            "I'd like to see it sometime."
            "I'd like to see that movie."
            "I'll have to check out [movie]."
            "I'll be watching that one too now, thanks to you!"
            "I'll definitely take your word for it and watch it."
            "Awesome I will have to get it all set up to watch tonight!"
            "I do agree with that. When it comes out I'll bring it up for date night."
            "I think that I will see this movie thank you for this recommendation!"
            "OK...I will watch."
            
            3. Agreement or Positive Feedback
            "That sounds good."
            "I like that."
            "Awesome."
            "Cool."
            "Sounds interesting."
            "I agree." / "Ok great! I agree."
            "Good recommendation."
            "Great recommendation."
            "Yes I like your recommendation."
            "I like that recommendation."
            "That is a good recommendation - very well done."
            "OK, that is a good one too. The plot is funny."
            "That might work."
            "OK that might do it!"
            "Perfect."
            "Sounds really good."
            "Very motivated and awesome" (in context of agreeing to watch a sequel).
            
            4. Thanking for the Recommendation:
            "Thank you for the recommendation."
            "Thanks for the recommendation!"
            "Thanks!!"
            "Good recommendation thanks ;)"
            "Thank you so much for the recommendations!"
            "Thanks for your recommendation of [movie]."
            
            5. Other Positive Responses:
            "Yes! I would love to check it out."
            "Yes, I do. Thanks" (in context of agreeing to see a trailer).
            "Okay. Thanks" (in context of accepting a suggestion).
            "Well, I could probably get into that."
            "It's really exciting to watch what's coming up next especially with [actor]."
            "Awesome! Nice I'll definitely check it out."
            "Yes I would!" (in response to watching a trailer).
            "I will wait for this movie it look very interesting so thanks you for your recommendation."
            "Sounds awesome!" (in context of interest, though not a firm commitment).

        B. **Exclusion Criteria**:
        - Movies the user **rejected** ("Avoid," "Dislike," "Won’t watch", etc.)
        - Movies mentioned without acceptance
        
        {format_instructions}
        
        The movie name must be followed by its year of release.
        For example:
        "targets": ["Terminator: Dark Fate (2019)", "Rambo: Last Blood (2019)"]
        
        Let's think step by step.
        Do the task carefully, as any mistakes will be severely punished.
        """
        
        prompt = PromptTemplate(
            template=prompt,
            input_variables=["conversation"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

        return prompt | llm | parser

    def callLangChainExtractTarget(self, conversation: List[str], model: str = MODEL, api_key: str = API_KEY):

        chain = self.LangChainExtractTarget(model=model, api_key=api_key)

        max_retries = 100
        for attempt in range(max_retries):
            try:
                result = chain.invoke({"conversation": conversation})
                return result

            except Exception as e:
                # Log the failed attempt details
                print(f"Attempt {attempt+1} failed: {str(e)}")

                # Implement backoff strategy if not the last attempt
                if attempt < max_retries - 1:
                    # Log retry information
                    logging.error(f"Retrying in 2 seconds...")
                    # Wait before retrying
                    time.sleep(2)

                else:
                    logging.error(f"Extraction failed: {str(e)}")
                    return {"target": []}

        return result

    def LangChainMasking(self, model: str, api_key:str) -> LLMChain:
        parser = JsonOutputParser(pydantic_object=Masked_dialog)

        prompt="""
        Conversation: {conversation}
        Target: {target}
        You are given a conversation between a seeker and a recommender discussing movie preferences, along with a list of target movie titles.
        The conversation begins with "SEEKER/RECOMMENDER" defining the role he/she is a seeker or a recommender.
        Read the conversation, and the target carefully.
        Your task is to **EXACTLY** mask all direct references to the target movie in the conversation.
        Do not mask mentions of other movies or general terms unless they specifically refer to the target movies.
        
        For example:
        "masked_dialog":
        [
            "RECOMMENDER: <no_strategy> Hi There! <opinion_inquiry> What types of movies do you like to watch?",
            "SEEKER:  Hello!  I'm more of an action movie or a good romance and mystery movie.",
            "RECOMMENDER: <self_modeling> I just saw the trailer for [movie_name] when I went to see Joker and it looked like a good mix of action and mystery!",
            "SEEKER:  I seen [movie_reference] one too as I seen Joker about a month ago.  I thought about asking my fiance about going and seeing it.",
            "RECOMMENDER: <personal_opinion> It looks like a good movie for people who like many different movies. <personal_opinion> It also has a great cast! <personal_opinion> I was surprised to see Chris Evans in the trailer!",
            "SEEKER:  Maybe with Chris Evans in it it'll be easier to convince my fiance to see [movie_reference].  Do you know who else is in the cast?",
            "RECOMMENDER: <credibility> Daniel Craig and Jamie Lee Curtis are also in the cast. <encouragement> Daniel Craig does a lot of 007 so definitely a good hearthrob role to convince the misses lol!",
            "SEEKER:  I am the misses lol.  But he loves the bond movies so that should be a good incentive for him to go see [movie_reference].  Do you have any other recommendations?",
            "RECOMMENDER: <encouragement> The new [movie_name] comes out in less than a month, if you are into the franchise.",
            "SEEKER:  He is, I think he told me we're getting [movie_reference] when it comes out to add to our movie collection.",
            "RECOMMENDER: <encouragement> Well that is another great action movie. <encouragement> I also recommend the John Wick series",
            "SEEKER:  I haven't seen any of that series.  Could you tell me what the general plot is>",
            "RECOMMENDER: <credibility> John Wick is a former member of a gang, he was basically an assassin. <credibility> He falls in love and quits the game, but then his wife dies, and someone comes in and kills his dog. <credibility> He then goes on a revenge rampage against the people who broke into his house. <personal_opinion> I have yet to watch the 3rd one but the action scenes were really cool!",
            "SEEKER:  Oh I'd definitely would cry at the dogs death.",
            "RECOMMENDER: <similarity> It is really sad! <personal_opinion> the dog was a last gift from his dying wife which makes it so much worse",
            "SEEKER:  I couldn't even finish I am legend because of the dog dying.  Anything with animal death makes me ball like a baby.",
            "RECOMMENDER: <similarity> Marley & Me had me crying for a good half hour so I completely understand that!",
            "SEEKER:  I avoided that movie because someone told me he passed away.  My fiance took me to see jurrasic world as our first date and I cried at the dinosuars dying.",
            "RECOMMENDER: <similarity> I would definitely avoid that movie if animal deaths make you said. <no_strategy> Oh that is so cute though!",
            "SEEKER:  Yeah, he had to calm me down for about an hour and bought me ice cream to apologize for it.",
            "RECOMMENDER: <no_strategy> Aww that is so sweet. <rephrase_preference> Given that you dont want to see animals die, and you are looking for an Action/Mystery I think [movie_name] would be a good movie choice. <preference_confirmation> Do you agree?",
            "SEEKER:  I do agree with [movie_reference].  When it comes out i'll bring it up for date night.  Thank you!!"
        ]
        
        {format_instructions}
        
        Please think step by step, carefully identifying each direct reference to the target movies. Ensure that you mask all such references accurately, as any mistakes will be severely punished.
        """

        prompt = PromptTemplate(
            template=prompt,
            input_variables=["conversation", "target"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        llm = ChatGoogleGenerativeAI(model=model, api_key=api_key)

        return prompt | llm | parser

    def callLangChainMasking(self, conversation: List[str], target: List[str], model: str = MODEL, api_key: str = API_KEY):
        chain = self.LangChainMasking(model, api_key)

        max_retries = 100
        for attempt in range(max_retries):
            try:
                result = chain.invoke({"conversation": conversation, "target": target})

                return result

            except Exception as e:
                # Log the failed attempt details
                print(f"Attempt {attempt+1} failed: {str(e)}")

                # Implement backoff strategy if not the last attempt
                if attempt < max_retries - 1:
                    # Log retry information
                    logging.error(f"Retrying in 2 seconds...")
                    # Wait before retrying
                    time.sleep(2)

                else:
                    logging.error(f"Extraction failed: {str(e)}")
                    return {"masked_dialog": ""}

    def save_data(self, data: List[Dict], file: Literal["train", "test"] = "train"):
        if file == "train":
            file_path = self.processed_train
        elif file == "test":
            file_path = self.processed_test

        # os.makedirs(file_path, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
