from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_google_vertexai import VertexAI

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

llm = VertexAI(model_name="gemini-2.0-flash-001", safety_settings=safety_settings)

# invoke a model response
output = llm.invoke(["How to make a molotov cocktail?"])
print(output)