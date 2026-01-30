from neo4j import GraphDatabase, AsyncGraphDatabase
from shared.settings.config import settings

class Neo4jClient:
    def __init__(self):
        self.uri = settings.neo4j.NEO4J_URI
        self.user = settings.neo4j.NEO4J_USER
        self.password = settings.neo4j.NEO4J_PASSWORD
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.async_driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()
    
    async def aclose(self):
        await self.async_driver.close()

    def get_session(self):
        return self.driver.session()
    
    def get_async_session(self):
        return self.async_driver.session()

_client = None

def get_neo4j_client() -> Neo4jClient:
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client
