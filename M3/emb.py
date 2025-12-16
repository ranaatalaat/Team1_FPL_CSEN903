"""
Enhanced Embedding Population Script
Creates embeddings that include fixture details for better query matching

Key improvement: Descriptions now include fixture information (home team, away team)
so queries like "how many fixtures did salah play in" can show fixture details
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Fantasy1234"  # UPDATE THIS!

# ============================================================================
# LOAD MODELS
# ============================================================================

print("Loading sentence-transformer models...")
print("1. Loading MiniLM-L6-v2 (384 dimensions)...")
model_minilm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✓ MiniLM loaded")

print("2. Loading MPNet-base-v2 (768 dimensions)...")
model_mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print("✓ MPNet loaded")

# ============================================================================
# CONNECT TO NEO4J
# ============================================================================

print(f"\nConnecting to Neo4j at {NEO4J_URI}...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

try:
    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        result.single()
    print("✓ Connected successfully")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print("\nPlease update NEO4J_PASSWORD in this script")
    exit(1)

# ============================================================================
# CREATE VECTOR INDEXES
# ============================================================================

print("\n" + "="*60)
print("STEP 1: Creating Vector Indexes")
print("="*60)

create_minilm_index = """
CREATE VECTOR INDEX player_embedding_minilm IF NOT EXISTS
FOR (p:Player) 
ON (p.embedding_minilm)
OPTIONS {indexConfig: {
  `vector.dimensions`: 384,
  `vector.similarity_function`: 'cosine'
}}
"""

create_mpnet_index = """
CREATE VECTOR INDEX player_embedding_mpnet IF NOT EXISTS
FOR (p:Player) 
ON (p.embedding_mpnet)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}}
"""

try:
    with driver.session() as session:
        print("Creating MiniLM index (384-dim)...")
        session.run(create_minilm_index)
        print("✓ MiniLM index created")
        
        print("Creating MPNet index (768-dim)...")
        session.run(create_mpnet_index)
        print("✓ MPNet index created")
        
        print("\nWaiting for indexes to come online...")
        time.sleep(5)
        
        result = session.run("SHOW INDEXES YIELD name, type WHERE type = 'VECTOR'")
        indexes = [record['name'] for record in result]
        print(f"✓ Active vector indexes: {indexes}")
        
except Exception as e:
    print(f"⚠ Index creation note: {e}")
    print("(Indexes may already exist - continuing...)")

# ============================================================================
# FETCH ALL PLAYERS WITH COMPLETE DATA INCLUDING FIXTURES
# ============================================================================

print("\n" + "="*60)
print("STEP 2: Fetching ALL Player Data with Fixture Details")
print("="*60)

fetch_players_query = """
MATCH (p:Player)
OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
OPTIONAL MATCH (p)-[:PLAYS_FOR]->(t:Team)
OPTIONAL MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
WITH p, pos.name as position, t.name as team,
     COUNT(r) as total_appearances,
     SUM(r.total_points) as total_points,
     SUM(r.goals_scored) as total_goals,
     SUM(r.assists) as total_assists,
     SUM(r.bonus) as total_bonus,
     SUM(r.bps) as total_bps,
     SUM(r.clean_sheets) as total_clean_sheets,
     SUM(r.goals_conceded) as total_goals_conceded,
     SUM(r.yellow_cards) as total_yellow_cards,
     SUM(r.red_cards) as total_red_cards,
     SUM(r.minutes) as total_minutes,
     SUM(r.saves) as total_saves,
     SUM(r.penalties_saved) as total_penalties_saved,
     SUM(r.penalties_missed) as total_penalties_missed,
     SUM(r.own_goals) as total_own_goals,
     AVG(r.ict_index) as avg_ict_index,
     AVG(r.influence) as avg_influence,
     AVG(r.creativity) as avg_creativity,
     AVG(r.threat) as avg_threat,
     AVG(r.value) as avg_value,
     SUM(r.selected) as total_selected,
     SUM(r.transfers_in) as total_transfers_in,
     SUM(r.transfers_out) as total_transfers_out,
     COLLECT(DISTINCT {home: home.name, away: away.name})[0..10] as sample_fixtures
RETURN p.player_name as player_name,
       position, team,
       total_appearances, total_points, total_goals, total_assists,
       total_bonus, total_bps,
       total_clean_sheets, total_goals_conceded,
       total_yellow_cards, total_red_cards,
       total_minutes, total_saves,
       total_penalties_saved, total_penalties_missed, total_own_goals,
       avg_ict_index, avg_influence, avg_creativity, avg_threat,
       avg_value, total_selected, total_transfers_in, total_transfers_out,
       sample_fixtures
ORDER BY p.player_name
"""

with driver.session() as session:
    result = session.run(fetch_players_query)
    players = [dict(record) for record in result]

print(f"✓ Found {len(players)} players")

if len(players) == 0:
    print("✗ No players found. Check your Neo4j data.")
    driver.close()
    exit(1)

# ============================================================================
# CREATE QUERY-OPTIMIZED PLAYER DESCRIPTIONS WITH FIXTURE INFO
# ============================================================================

def create_player_description_with_fixtures(player_data):
    """
    Enhanced: Create descriptions that include fixture details
    
    This format helps with queries like "how many fixtures did salah play in":
    - Includes fixture count prominently
    - Lists sample fixtures (home vs away teams)
    - Uses query-like language for better matching
    """
    name = player_data.get('player_name', 'Unknown Player')
    position = player_data.get('position', 'Unknown')
    team = player_data.get('team', 'Unknown Team')
    
    # Get stats with None handling
    goals = int(player_data.get('total_goals', 0) or 0)
    assists = int(player_data.get('total_assists', 0) or 0)
    points = int(player_data.get('total_points', 0) or 0)
    bonus = int(player_data.get('total_bonus', 0) or 0)
    apps = int(player_data.get('total_appearances', 0) or 0)
    ict = float(player_data.get('avg_ict_index', 0) or 0)
    yellows = int(player_data.get('total_yellow_cards', 0) or 0)
    cs = int(player_data.get('total_clean_sheets', 0) or 0)
    
    # Get fixture information
    sample_fixtures = player_data.get('sample_fixtures', [])
    
    # Build query-optimized description
    parts = []
    
    # FIXTURE COUNT - prominently featured for "how many fixtures" queries
    if apps > 0:
        parts.append(f"{name} played in {apps} fixtures")
        parts.append(f"appeared in {apps} games")
        parts.append(f"total appearances {apps}")
    
    # FIXTURE DETAILS - include sample fixtures with home/away teams
    if sample_fixtures and len(sample_fixtures) > 0:
        fixture_list = []
        for idx, fixture in enumerate(sample_fixtures[:5], 1):  # Limit to first 5 fixtures
            home = fixture.get('home', 'Unknown')
            away = fixture.get('away', 'Unknown')
            if home and away and home != 'Unknown' and away != 'Unknown':
                fixture_list.append(f"{home} vs {away}")
        
        if fixture_list:
            parts.append(f"fixtures include: {', '.join(fixture_list)}")
    
    # GOALS - Lead with performance in query-relevant language
    if goals > 0:
        parts.append(f"scored {goals} goals")
        if goals >= 20:
            parts.append(f"top goal scorer with {goals} goals")
        elif goals >= 10:
            parts.append(f"high scoring player {goals} goals")
    
    # ASSISTS
    if assists > 0:
        parts.append(f"provided {assists} assists")
        if assists >= 10:
            parts.append(f"top assist provider {assists} assists")
    
    # POINTS
    if points > 0:
        parts.append(f"earned {points} Fantasy Premier League points")
        if points >= 200:
            parts.append(f"highest points scorer {points} FPL points")
        elif points >= 150:
            parts.append(f"top points earner {points} points")
    
    # Add position and team context
    parts.append(f"{position} playing for {team}")
    
    # Bonus and ICT
    if bonus > 0:
        parts.append(f"bonus points {bonus}")
        if bonus >= 20:
            parts.append(f"top bonus performer {bonus} bonus")
    
    if ict > 0:
        parts.append(f"ICT index {ict:.1f}")
        if ict >= 100:
            parts.append(f"highest ICT {ict:.1f}")
    
    if cs > 0:
        parts.append(f"clean sheets {cs}")
        parts.append(f"kept {cs} clean sheets")
        if cs >= 15:
            parts.append(f"most clean sheets {cs}")
            parts.append(f"top clean sheet keeper with {cs} clean sheets")
        elif cs >= 10:
            parts.append(f"high clean sheet count {cs}")
            parts.append(f"excellent defensive record {cs} clean sheets")
    
    # Disciplinary
    if yellows > 0:
        parts.append(f"yellow cards {yellows}")
        if yellows >= 5:
            parts.append(f"most bookings {yellows} yellows")
    
    # Create final description with fixture information prominently featured
    description = f"{name}. " + ". ".join(parts) + "."
    
    return description


# ============================================================================
# GENERATE AND STORE EMBEDDINGS
# ============================================================================

update_embeddings_query = """
MATCH (p:Player {player_name: $player_name})
SET p.embedding_minilm = $embedding_minilm,
    p.embedding_mpnet = $embedding_mpnet,
    p.embedding_description = $description
RETURN p.player_name as updated_player
"""

print("\n" + "="*60)
print("STEP 3: Generating Enhanced Embeddings with Fixture Info")
print("="*60)
print("Using descriptions that include fixture details for better matching...\n")

success_count = 0
error_count = 0

for player_data in tqdm(players, desc="Processing players"):
    player_name = player_data.get('player_name')
    
    if not player_name:
        error_count += 1
        continue
    
    try:
        description = create_player_description_with_fixtures(player_data)
        
        # Generate embeddings
        embedding_minilm = model_minilm.encode(description).tolist()
        embedding_mpnet = model_mpnet.encode(description).tolist()
        
        # Verify dimensions
        if len(embedding_minilm) != 384:
            raise ValueError(f"MiniLM: {len(embedding_minilm)} dims, expected 384")
        if len(embedding_mpnet) != 768:
            raise ValueError(f"MPNet: {len(embedding_mpnet)} dims, expected 768")
        
        # Store in Neo4j
        with driver.session() as session:
            session.run(update_embeddings_query, {
                'player_name': player_name,
                'embedding_minilm': embedding_minilm,
                'embedding_mpnet': embedding_mpnet,
                'description': description
            })
        
        success_count += 1
        time.sleep(0.01)
        
    except Exception as e:
        error_count += 1

# ============================================================================
# VERIFICATION
# ============================================================================

print("\n" + "="*60)
print("STEP 4: Verifying Enhanced Embeddings")
print("="*60)

verify_query = """
MATCH (p:Player)
WHERE p.embedding_minilm IS NOT NULL AND p.embedding_mpnet IS NOT NULL
RETURN COUNT(p) as players_with_embeddings
"""

try:
    with driver.session() as session:
        result = session.run(verify_query)
        record = result.single()
        if record:
            print(f"✓ Players with embeddings: {record['players_with_embeddings']}")
        
        print("\n" + "="*60)
        print("Testing Enhanced Semantic Search with Fixture Queries")
        print("="*60)
        
        test_queries = [
            ("how many fixtures did salah play in", "Salah fixtures"),
            ("which games did haaland appear in", "Haaland appearances"),
            ("top goal scorer highest goals", "Top goal scorers"),
        ]
        
        for query_text, label in test_queries:
            print(f"\n✓ Test: {label}")
            print(f"  Query: '{query_text}'")
            
            vec = model_minilm.encode(query_text).tolist()
            
            search_query = """
            CALL db.index.vector.queryNodes(
                'player_embedding_minilm',
                10,
                $query_vector
            ) YIELD node AS p, score AS similarity
            OPTIONAL MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
            OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
            OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
            WITH p, similarity,
                COUNT(DISTINCT f) as fixture_count,
                SUM(r.total_points) as total_points,
                SUM(r.goals_scored) as total_goals,
                SUM(r.assists) as total_assists,
                COLLECT(DISTINCT {home: home.name, away: away.name})[0..3] as sample_fixtures
            RETURN p.player_name as player, 
                   similarity,
                   fixture_count,
                   total_goals,
                   total_assists,
                   total_points,
                   sample_fixtures
            ORDER BY similarity DESC
            LIMIT 5
            """
            
            result = session.run(search_query, {'query_vector': vec})
            print("  Results (with fixture information):")
            for rec in result:
                fixtures_str = ""
                if rec['sample_fixtures'] and len(rec['sample_fixtures']) > 0:
                    fixture_list = [f"{f['home']} vs {f['away']}" 
                                   for f in rec['sample_fixtures'] 
                                   if f.get('home') and f.get('away')]
                    if fixture_list:
                        fixtures_str = f" | Fixtures: {', '.join(fixture_list[:2])}"
                
                print(f"    - {rec['player']}: {rec['fixture_count']} fixtures, "
                      f"{rec['total_goals']}G {rec['total_assists']}A "
                      f"{rec['total_points']}pts (sim: {rec['similarity']:.4f}){fixtures_str}")
            
except Exception as e:
    print(f"⚠ Verification error: {e}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print("ENHANCED EMBEDDING POPULATION COMPLETE")
print("="*60)
print(f"✓ Successfully processed: {success_count} players")
print(f"✗ Errors: {error_count}")

print(f"\n✓ Embeddings now include fixture information:")
print(f"  - Fixture count prominently featured")
print(f"  - Sample fixtures with home/away teams included")
print(f"  - Better matching for 'how many fixtures' queries")
print(f"  - Descriptions show fixture details in results")

print("\n" + "="*60)
print("✓ Database ready for fixture-aware semantic search!")
print("="*60)

driver.close()
