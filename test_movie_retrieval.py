#!/usr/bin/env python3
"""
Test script to verify movie retrieval with specific movie content and preferences
"""

import pandas as pd
from utils.GraphDB.graph_retriever import query_parse_output_graph

def test_movie_scenarios():
    """Test movie retrieval with various movie preferences"""
    
    # Create dummy dataframe for fallback
    test_df = pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5],
        'title': ['Test Movie 1', 'Test Movie 2', 'Test Movie 3', 'Test Movie 4', 'Test Movie 5'],
        'movieName': ['Test Movie 1', 'Test Movie 2', 'Test Movie 3', 'Test Movie 4', 'Test Movie 5']
    })
    
    # Define test scenarios
    test_scenarios = [
        {
            "name": "Dark Knight Fan",
            "preferences": """
            I'm looking for movies similar to The Dark Knight. 
            I love action movies with complex villains, dark atmosphere, 
            superhero themes, and psychological depth. The movie should have 
            great cinematography, intense fight scenes, and moral dilemmas.
            """
        },
        {
            "name": "Romantic Comedy Lover",
            "preferences": """
            I want romantic comedies that are light-hearted and funny.
            I enjoy movies with great chemistry between leads, witty dialogue,
            and happy endings. Something like The Proposal or Crazy Stupid Love.
            """
        },
        {
            "name": "Sci-Fi Thriller Fan",
            "preferences": """
            I'm interested in science fiction thrillers with mind-bending plots.
            Movies like Inception, The Matrix, or Blade Runner 2049.
            I like films that make you think and have great visual effects.
            """
        },
        {
            "name": "Horror Movie Enthusiast",
            "preferences": """
            I love horror movies that are genuinely scary and suspenseful.
            Psychological horror and supernatural themes are my favorite.
            Movies like The Conjuring, Hereditary, or Get Out.
            """
        },
        {
            "name": "Family Adventure Seeker",
            "preferences": """
            I want family-friendly adventure movies that kids and adults can enjoy.
            Movies with great animation, humor, and heartwarming stories.
            Something like Pixar movies or How to Train Your Dragon.
            """
        }
    ]
    
    print("🎬 Testing Movie Retrieval with Specific Content")
    print("=" * 60)
    
    for scenario in test_scenarios:
        print(f"\n🎭 Scenario: {scenario['name']}")
        print("-" * 40)
        print(f"Preferences: {scenario['preferences'].strip()}")
        print("-" * 40)
        
        try:
            # Test retrieval
            result = query_parse_output_graph(
                df_movie=test_df,
                summarized_preferences=scenario['preferences'],
                data="redial",
                n=10
            )
            
            print(f"✅ Retrieved {len(result)} movies")
            if result:
                print("🎥 Top recommended movies:")
                for i, movie in enumerate(result[:5], 1):
                    print(f"   {i}. {movie}")
            else:
                print("❌ No movies retrieved")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()

def test_specific_genres():
    """Test retrieval for specific genres"""
    
    test_df = pd.DataFrame({
        'movieId': [1, 2, 3],
        'title': ['Test Movie 1', 'Test Movie 2', 'Test Movie 3'],
        'movieName': ['Test Movie 1', 'Test Movie 2', 'Test Movie 3']
    })
    
    genres_to_test = [
        "action", "comedy", "drama", "horror", "romance", 
        "thriller", "sci-fi", "fantasy", "animation", "documentary"
    ]
    
    print("\n🎬 Testing Genre-Specific Retrieval")
    print("=" * 60)
    
    for genre in genres_to_test:
        print(f"\n🎭 Testing genre: {genre.upper()}")
        preferences = f"I love {genre} movies. Give me the best {genre} films with high ratings."
        
        try:
            result = query_parse_output_graph(
                df_movie=test_df,
                summarized_preferences=preferences,
                data="redial",
                n=8
            )
            
            print(f"✅ Retrieved {len(result)} {genre} movies")
            if result:
                print("🎥 Top recommendations:")
                for i, movie in enumerate(result[:3], 1):
                    print(f"   {i}. {movie}")
                    
        except Exception as e:
            print(f"❌ Error for {genre}: {e}")

if __name__ == "__main__":
    print("🚀 Starting Movie Retrieval Tests...")
    
    # Test specific movie scenarios
    test_movie_scenarios()
    
    # Test genre-specific retrieval
    test_specific_genres()
    
    print("\n✅ All movie retrieval tests completed!")
    print("=" * 60)
