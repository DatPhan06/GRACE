#!/usr/bin/env python3
"""
Test script to verify graph database integration
"""

import yaml
import pandas as pd
from utils.GraphDB.graph_retriever import query_parse_output_graph

def test_graph_integration():
    """Test the graph database integration"""
    
    # Load config
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        print("✓ Config loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False
    
    # Create dummy dataframe for testing
    test_df = pd.DataFrame({
        'movieId': [1, 2, 3],
        'title': ['Test Movie 1', 'Test Movie 2', 'Test Movie 3'],
        'movieName': ['Test Movie 1', 'Test Movie 2', 'Test Movie 3']
    })
    
    # Test preferences
    test_preferences = "I like action movies with good ratings"
    
    print("Testing Neo4j database structure...")
    from utils.GraphDB.graph_retriever import GraphRetriever
    import os
    
    # Quick database structure check
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost")
    NEO4J_PORT = os.getenv("NEO4J_PORT", "7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    try:
        retriever = GraphRetriever(NEO4J_URI, NEO4J_PORT, NEO4J_USER, NEO4J_PASSWORD)
        with retriever.driver.session() as session:
            # Check if Film nodes exist
            result = session.run("MATCH (f:Film) RETURN count(f) AS filmCount")
            film_count = result.single()["filmCount"]
            print(f"✓ Found {film_count} Film nodes in database")
            
            # Check if plot_embedding exists
            result = session.run("MATCH (f:Film) WHERE f.plot_embedding IS NOT NULL RETURN count(f) AS embeddingCount")
            embedding_count = result.single()["embeddingCount"]
            print(f"✓ Found {embedding_count} films with plot embeddings")
            
        retriever.close()
    except Exception as e:
        print(f"⚠ Could not check database structure: {e}")
    
    try:
        # Test with inspired dataset
        print("Testing with 'inspired' dataset...")
        result_inspired = query_parse_output_graph(
            df_movie=test_df,
            summarized_preferences=test_preferences,
            data="inspired",
            n=10
        )
        print(f"✓ Inspired test completed. Retrieved {len(result_inspired)} movies")
        
        # Test with redial dataset
        print("Testing with 'redial' dataset...")
        result_redial = query_parse_output_graph(
            df_movie=test_df,
            summarized_preferences=test_preferences,
            data="redial",
            n=10
        )
        print(f"✓ Redial test completed. Retrieved {len(result_redial)} movies")
        
        # Test if we got any results
        total_results = len(result_inspired) + len(result_redial)
        if total_results > 0:
            print(f"✓ Successfully retrieved {total_results} total movies from graph database")
        else:
            print("⚠ No movies retrieved from graph database, but connection works")
        
        # Test with specific movie content
        print("\n" + "="*50)
        print("Testing with specific movie content...")
        movie_content_test = """
        I'm looking for a movie similar to The Dark Knight. 
        I love action movies with complex villains, dark atmosphere, 
        superhero themes, and psychological depth. The movie should have 
        great cinematography, intense fight scenes, and moral dilemmas.
        """
        
        print("Testing semantic search with specific movie preferences:")
        print(f"Query: {movie_content_test.strip()}")
        
        result_content = query_parse_output_graph(
            df_movie=test_df,
            summarized_preferences=movie_content_test,
            data="redial",
            n=15
        )
        
        print(f"\n✓ Content-based test completed. Retrieved {len(result_content)} movies")
        if result_content:
            print("Retrieved movies:")
            for i, movie in enumerate(result_content[:10], 1):
                print(f"  {i}. {movie}")
        else:
            print("  No movies retrieved for content-based search")
            
        return True
        
    except Exception as e:
        print(f"✗ Graph database test failed: {e}")
        print("This is expected if Neo4j is not running or no data is loaded")
        return False

if __name__ == "__main__":
    print("Testing Graph Database Integration...")
    print("=" * 50)
    
    success = test_graph_integration()
    
    if success:
        print("=" * 50)
        print("✓ All tests passed! Graph integration is working.")
    else:
        print("=" * 50)
        print("✗ Some tests failed. Check Neo4j connection and data.")
        print("\nTo fix:")
        print("1. Ensure Neo4j is running")
        print("2. Ensure movie data is loaded using graph_builder.py")
        print("3. Check config.yaml for correct Neo4j credentials")
