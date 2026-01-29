import json
from neo4j import GraphDatabase
import os
import re
from dotenv import load_dotenv
# from openai import AsyncAzureOpenAI
from openai import AsyncOpenAI
import asyncio
import yaml

load_dotenv()

def clean_movie_data_for_cypher(movie_data):
    """
    Cleans and transforms movie data into a flat dictionary suitable for the Cypher query.
    """
    params = {}
    params["movieId"] = movie_data.get("movieId")
    params["title"] = movie_data.get("title")
    params["plot"] = movie_data.get("plot")
    params["poster"] = movie_data.get("poster")
    params["imdbId"] = movie_data.get("imdb_id")
    params["imdbVotes"] = movie_data.get("imdb_votes")
    params["awards"] = movie_data.get("awards")
    params["runtime"] = movie_data.get("runtime")

    year_str = movie_data.get("year", "")
    if isinstance(year_str, str):
        match = re.search(r'\d{4}', year_str)
        params["year"] = int(match.group(0)) if match else None
    elif isinstance(year_str, int):
        params["year"] = year_str
    else:
        params["year"] = None

    try:
        params["imdbRating"] = float(movie_data.get("imdb_rating"))
    except (ValueError, TypeError):
        params["imdbRating"] = None
        
    list_keys = {
        "language": "language", "country": "country", "genre": "genre", 
        "director": "director", "writer": "writer", "actors": "actors"
    }
    for raw_key, cypher_key in list_keys.items():
        value = movie_data.get(raw_key)
        if isinstance(value, str):
            items = []
            for item in value.split(','):
                cleaned_item = item.strip()
                # Filter out empty strings and "N/A"
                if cleaned_item and cleaned_item.upper() != "N/A":
                    # Normalize casing for specific fields to ensure consistency
                    if cypher_key in ["genre", "language", "country"]:
                        cleaned_item = cleaned_item.title()
                    items.append(cleaned_item)
            params[cypher_key] = items
        else:
            params[cypher_key] = []
    
    return params

class GraphBuilder:
    """
    A class to build a movie graph in a Neo4j database, including embeddings.
    """
    def __init__(self, uri, port, user, password, embedding_client, embedding_deployment):
        self.driver = GraphDatabase.driver(f"{uri}:{port}", auth=(user, password))
        self.embedding_client = embedding_client
        self.embedding_deployment = embedding_deployment
        try:
            self.driver.verify_connectivity()
            print("Successfully connected to Neo4j.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed.")

    async def get_plot_embedding(self, plot_text):
        if not plot_text or not self.embedding_client:
            return None
        try:
            response = await self.embedding_client.embeddings.create(
                input=plot_text, model=self.embedding_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Could not get embedding for plot: {e}")
            return None

    async def insert_movie_with_embedding(self, movie_data):
        if not self.driver:
            print("Cannot insert data, no database connection.")
            return

        params = clean_movie_data_for_cypher(movie_data)
        plot_embedding = await self.get_plot_embedding(params.get("plot"))
        params['plot_embedding'] = plot_embedding
        
        with self.driver.session() as session:
            session.execute_write(self._create_film_with_relationships, params)

    @staticmethod
    def _create_film_with_relationships(tx, params):
        # 1. Create or Merge the main Film node
        tx.run("""
        MERGE (f:Film {movieId: $movieId})
        SET
            f.title = $title,
            f.plot = $plot,
            f.plot_embedding = $plot_embedding,
            f.poster = $poster,
            f.imdbId = $imdbId,
            f.imdbVotes = $imdbVotes,
            f.awards = $awards,
            f.runtime = $runtime
        """, params)

        # 2. Conditionally create related nodes and relationships
        if params.get("year"):
            tx.run("""
            MATCH (f:Film {movieId: $movieId})
            MERGE (y:ReleaseYear {year: $year})
            MERGE (f)-[:RELEASED_IN]->(y)
            """, {"movieId": params["movieId"], "year": params["year"]})

        if params.get("imdbRating"):
            tx.run("""
            MATCH (f:Film {movieId: $movieId})
            MERGE (r:ImdbRating {value: $imdbRating})
            MERGE (f)-[:HAS_RATING]->(r)
            """, {"movieId": params["movieId"], "imdbRating": params["imdbRating"]})

        # 3. Batch-create nodes and relationships from lists
        list_creation_queries = {
            "language": "UNWIND $names AS name MATCH (f:Film {movieId: $movieId}) MERGE (l:Language {name: name}) MERGE (f)-[:IN_LANGUAGE]->(l)",
            "country": "UNWIND $names AS name MATCH (f:Film {movieId: $movieId}) MERGE (c:Country {name: name}) MERGE (f)-[:FROM_COUNTRY]->(c)",
            "genre": "UNWIND $names AS name MATCH (f:Film {movieId: $movieId}) MERGE (g:Genre {name: name}) MERGE (f)-[:IN_GENRE]->(g)",
            "director": "UNWIND $names AS name MATCH (f:Film {movieId: $movieId}) MERGE (d:Director {name: name}) MERGE (d)-[:DIRECTED]->(f)",
            "writer": "UNWIND $names AS name MATCH (f:Film {movieId: $movieId}) MERGE (w:Writer {name: name}) MERGE (w)-[:WROTE]->(f)",
            "actors": "UNWIND $names AS name MATCH (f:Film {movieId: $movieId}) MERGE (a:Actor {name: name}) MERGE (a)-[:ACTED_IN]->(f)"
        }

        for key, query in list_creation_queries.items():
            if params[key]:
                tx.run(query, {"movieId": params["movieId"], "names": params[key]})


def load_movies_from_file(filepath):
    movies = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    movies.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{filepath}'.")
    return movies

async def main():
    print("[INFO] Đang đọc file config.yaml...")
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    print("[INFO] Đã đọc config.yaml thành công.")

    print("[INFO] Đang lấy các tham số từ config...")
    movie_file_path = config["RedialDataPath"]["processed"]["movie"]
    movie_file_path = os.path.join(PROJECT_ROOT, movie_file_path)
    all_movies = load_movies_from_file(movie_file_path)
    
    if not all_movies:
        print("No movie data loaded. Exiting.")
        return

    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost")
    NEO4J_PORT = os.getenv("NEO4J_PORT", "7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    # embedding_api_key = os.getenv("EMBEDDING__KEY")
    # embedding_api_version = os.getenv("EMBEDDING__API_VERSION")
    # embedding_azure_endpoint = os.getenv("EMBEDDING__ENDPOINT")
    # embedding_deployment_name = os.getenv("EMBEDDING__DEPLOYMENT_NAME")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "10"))

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    # azure_embedding_client = AsyncAzureOpenAI(
    #     api_key=embedding_api_key,
    #     api_version=embedding_api_version,
    #     azure_endpoint=embedding_azure_endpoint,
    # )
    openai_client = AsyncOpenAI(api_key=openai_api_key)

    builder = GraphBuilder(
        NEO4J_URI, NEO4J_PORT, NEO4J_USER, NEO4J_PASSWORD,
        openai_client, embedding_model
    )
    
    async def insert_with_semaphore(movie_data):
        async with semaphore:
            await builder.insert_movie_with_embedding(movie_data)

    if builder.driver:
        tasks = [insert_with_semaphore(movie) for movie in all_movies]
        print(f"[INFO] Bắt đầu xử lý {len(all_movies)} phim với giới hạn đồng thời là {CONCURRENCY_LIMIT}...")
        
        for i, task in enumerate(asyncio.as_completed(tasks)):
            await task
            print(f"({i + 1}/{len(all_movies)}) Đã xử lý phim xong.")

        print(f"\nĐã chèn xong {len(all_movies)} phim.")
        builder.close()

if __name__ == "__main__":
    asyncio.run(main())
