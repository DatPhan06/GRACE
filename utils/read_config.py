import configparser
import os
from typing import Dict, Any


def read_config(config_path: os.PathLike = "config.ini") -> Dict[str, Any]:
    """
    Retrieve all configuration values from the specified config file

    Args:
        config_path: Path to the configuration file (defaults to 'config.ini')

    Returns:
        Dictionary containing all configuration values organized by category
    """
    # Create a ConfigParser object to read and parse the INI file
    config = configparser.ConfigParser()

    # Read the configuration file from the specified path
    config.read(config_path)

    # Create a dictionary to store all configuration values
    config_values = {}

    # Read Embedding Model configuration
    if "EmbeddingModel" in config:
        config_values["model_embedding"] = config.get("EmbeddingModel", "model_embedding")

    # Read Gemini Model configuration
    if "GeminiModel" in config:
        config_values["model_generate_data"] = config.get("GeminiModel", "model_generate_data")
        config_values["model_infer"] = config.get("GeminiModel", "model_infer")

    # Read INSPIRED dataset paths
    if "InspiredDataPath" in config:
        config_values["insp_dialog_train_data_path"] = config.get("InspiredDataPath", "dialog_train_data_path")
        config_values["insp_dialog_test_data_path"] = config.get("InspiredDataPath", "dialog_test_data_path")
        config_values["insp_movie_data_path"] = config.get("InspiredDataPath", "movie_data_path")

    # Read ReDial dataset paths
    if "RedialDataPath" in config:
        config_values["redial_movie_data_path"] = config.get("RedialDataPath", "movie_data_path")
        config_values["redial_dialog_train_data_path"] = config.get("RedialDataPath", "dialog_train_data_path")
        config_values["redial_dialog_test_data_path"] = config.get("RedialDataPath", "dialog_test_data_path")

    # Read preprocessed data paths
    if "ProcessedDataPath" in config:
        config_values["processed_redial_dialog_train_data_path"] = config.get("ProcessedDataPath", "redial_dialog_train_data_path")
        config_values["processed_redial_dialog_test_data_path"] = config.get("ProcessedDataPath", "redial_dialog_test_data_path")
        config_values["processed_redial_movie_data_path"] = config.get("ProcessedDataPath", "redial_movie_data_path")
        config_values["processed_insp_dialog_train_data_path"] = config.get("ProcessedDataPath", "insp_dialog_train_data_path")
        config_values["processed_insp_dialog_test_data_path"] = config.get("ProcessedDataPath", "insp_dialog_test_data_path")
        config_values["processed_insp_movie_data_path"] = config.get("ProcessedDataPath", "insp_movie_data_path")

    # Read output paths
    if "OutputPath" in config:
        config_values["redial_output_path"] = config.get("OutputPath", "redial_output_path")
        config_values["insp_output_path"] = config.get("OutputPath", "insp_output_path")

    # Read database configuration
    if "DB" in config:
        config_values["redial_chroma_db_path"] = config.get("DB", "redial_chroma_db_path")
        config_values["redial_collection_name"] = config.get("DB", "redial_collection_name")
        config_values["insp_chroma_db_path"] = config.get("DB", "insp_chroma_db_path")
        config_values["insp_collection_name"] = config.get("DB", "insp_collection_name")
        config_values["top_k"] = config.get("DB", "top_k")

    # Read API keys
    if "APIKey" in config:
        config_values["google_api_key"] = config.get("APIKey", "GOOGLE_API_KEY_2")
        config_values["omdb_api_key"] = config.get("APIKey", "OMDB_API_KEY")

    return config_values
