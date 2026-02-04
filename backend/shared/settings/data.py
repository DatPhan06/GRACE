from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class DataSettings(BaseSettings):
    # REDIAL
    # Maps to config["RedialDataPath"]["processed"]["dialog"]["test_with_liked_movies"]
    REDIAL_DIALOG_DATA: str = "dataset/REDIAL/processed/dialog_data/data_test_with_liked_movies.json"
    # Maps to config["RedialDataPath"]["raw"]["movie"]
    REDIAL_MOVIE_DATA: str = "dataset/REDIAL/raw/movie_data/movies_with_mentions.csv"

    # INSPIRED
    # Maps to config['InspiredDataPath']['processed']['dialog']['train']
    INSPIRED_DIALOG_DATA: str = "dataset/INSPIRED/processed/dialog_data/train_processed_all.json"
    # Maps to config['InspiredDataPath']['processed']['movie']
    INSPIRED_MOVIE_DATA: str = "dataset/INSPIRED/processed/movie_data/movie_database_no_missing.json"

    # Output
    OUTPUT_DIR: str = "output"

    model_config = SettingsConfigDict(
        env_file=[".env", "../.env"],
        env_ignore_empty=True,
        extra="ignore"
    )
