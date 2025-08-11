import yaml
import google.generativeai as genai
from google.api_core.exceptions import PermissionDenied, ClientError

def test_google_api_keys():
    """
    Tests all Google API keys in the config.yaml file and prints their status.
    """
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please make sure the file exists in the same directory.")
        return
    except yaml.YAMLError as e:
        print(f"Error reading config.yaml: {e}")
        return

    api_keys = []
    # Assuming keys are named GOOGLE_API_KEY_0, GOOGLE_API_KEY_1, etc.
    for i in range(50): # Check for up to 50 keys
        key_name = f"GOOGLE_API_KEY_{i}"
        if key_name in config.get("APIKey", {}):
            api_keys.append((key_name, config["APIKey"][key_name]))
        else:
            # Stop if we don't find a key in sequence
            if i > 0 and f"GOOGLE_API_KEY_{i-1}" in config.get("APIKey", {}):
                break

    if not api_keys:
        print("No Google API keys found in config.yaml under the 'APIKey' section.")
        return

    print(f"Found {len(api_keys)} keys to test. Starting validation...\n")

    for key_name, key_value in api_keys:
        try:
            # Configure the genai library with the current key
            genai.configure(api_key=key_value)
            
            # Make a lightweight call to check if the key is valid
            # Listing models is a good way to test authentication
            genai.list_models()
            print(f"key_name: {genai.list_models()}")
            
            print(f"- {key_name}: VALID")

        except PermissionDenied:
            print(f"- {key_name}: INVALID (Permission Denied - The key might be incorrect, disabled, or have restrictions.)")
        except ClientError as e:
            # Handle other client-side errors, e.g., invalid format
            print(f"- {key_name}: INVALID (Client Error: {e})")
        except Exception as e:
            # Catch any other unexpected errors
            print(f"- {key_name}: FAILED (An unexpected error occurred: {e})")

if __name__ == "__main__":
    test_google_api_keys() 