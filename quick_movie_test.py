#!/usr/bin/env python3
"""
Quick test of movie retrieval with specific content
"""

import pandas as pd
from utils.GraphDB.graph_retriever import query_parse_output_graph

def quick_test():
    # Dummy dataframe
    test_df = pd.DataFrame({
        'movieId': [1, 2, 3],
        'title': ['Fallback Movie 1', 'Fallback Movie 2', 'Fallback Movie 3'],
        'movieName': ['Fallback Movie 1', 'Fallback Movie 2', 'Fallback Movie 3']
    })
    
    # Test with The Dark Knight-like preferences
    preferences = """
    I love action movies with:
    - Complex, psychological villains like the Joker
    - Dark, gritty atmosphere and cinematography
    - Superhero or vigilante themes
    - Moral dilemmas and ethical questions
    - Intense action sequences and fight scenes
    - Great acting performances
    - Crime thriller elements
    """
    
    print("🎬 Testing Movie Retrieval")
    print("=" * 50)
    print("Movie Preferences:")
    print(preferences.strip())
    print("=" * 50)
    
    try:
        result = query_parse_output_graph(
            df_movie=test_df,
            summarized_preferences=preferences,
            data="redial",
            n=500
        )
        
        print(f"\n✅ Successfully retrieved {len(result)} movies!")
        
        if result:
            print("\n🎥 Recommended Movies:")
            for i, movie in enumerate(result, 1):
                print(f"  {i:2d}. {movie}")
        else:
            print("\n❌ No movies were retrieved from the graph database")
            print("This might mean:")
            print("- Neo4j database is empty")
            print("- No embeddings were created")
            print("- Connection issues")
            
    except Exception as e:
        print(f"\n❌ Error during retrieval: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()