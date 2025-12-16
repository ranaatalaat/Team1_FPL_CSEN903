# Force offline mode BEFORE any other imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import streamlit as st
import re
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from neo4j import GraphDatabase
import requests
from datetime import datetime


# ============================================================================
# SECTION 1: TRIVIA INTENT CLASSIFICATION (10 DIVERSE TYPES)
# ============================================================================

class TriviaIntentType(Enum):
    """10 Different Trivia Question Types - Diverse & Fun"""
    TOP_GOAL_SCORER = "top_goal_scorer"              # 1. Who scored the most goals overall?
    TOP_ASSIST_PROVIDER = "top_assist_provider"      # 2. Who provided the most assists?
    HIGHEST_POINTS_PLAYER = "highest_points_player"  # 3. Who earned the most FPL points?
    PLAYER_FIXTURES = "player_fixtures"              # 4. Which fixtures did a player appear in?
    PLAYER_VS_PLAYER = "player_vs_player"            # 5. Compare two players' performance
    BEST_BONUS_PERFORMER = "best_bonus_performer"    # 6. Who earned the most bonus ?
    MOST_YELLOW_CARDS = "most_yellow_cards"          # 7. Who received the most yellow cards?
    HIGHEST_ICT_INDEX = "highest_ict_index"          # 8. Who had the highest ICT index?
    PLAYER_CLEAN_SHEETS = "player_clean_sheets"      # 9. Which player has the most clean sheets?
    GAMEWEEK_TOP_SCORER = "gameweek_top_scorer"      # 10. Who scored the most in a specific gameweek?
    UNKNOWN = "unknown"

# Valid seasons in the database
VALID_SEASONS = ["2021-22", "2022-23"]


@dataclass
class TriviaResult:
    """Trivia classification result"""
    intent: TriviaIntentType
    confidence: float
    entities: Dict[str, List[str]]
    suggested_query: str
    description: str


@dataclass
class EvaluationMetrics:
    """Metrics for LLM evaluation - Quantitative Analysis"""
    model_name: str
    response_time: float
    token_count: int
    answer_length: int
    query: str
    answer: str
    context_size: int
    timestamp: str
    retrieval_method: str
    embedding_strategy: str = "none"
    relevance_score: float = 0.0  # 0-5 scale
    accuracy_score: float = 0.0   # 0-5 scale
    completeness_score: float = 0.0  # 0-5 scale
    
    # Quantitative: Cost Calculation
    price_per_token: float = 0.0  # Price per token in USD (e.g., 0.000002 for GPT-4)
    cost: float = 0.0  # Total cost: tokens × price_per_token
    
    # Quantitative: Retrieval Effectiveness
    retrieval_correct: bool = False  # Did this retrieval method produce correct answer?
    kg_strategy: str = "baseline"  # "baseline" or "embeddings"
    
    def calculate_cost(self) -> float:
        """Calculate cost: tokens × price_per_token"""
        self.cost = self.token_count * self.price_per_token
        return self.cost


@dataclass
class QualitativeEvaluation:
    """Framework for human evaluation of answer quality - Qualitative Analysis"""
    query: str
    answer: str
    model_name: str
    retrieval_method: str
    embedding_model: str
    
    # Rating scales (1-5)
    relevance: int = 0      # Is the answer relevant to the question?
    accuracy: int = 0       # Is the answer factually correct based on context?
    completeness: int = 0   # Does the answer fully address the question?
    clarity: int = 0        # Is the answer clear and well-structured?
    
    # Qualitative: Naturalness Score
    naturalness: int = 0    # How natural/human-like is the answer? (1-5 scale)
    
    # Qualitative: User Preference (Side-by-Side Comparison)
    comparison_id: str = ""  # ID to group A/B comparisons
    variant: str = ""        # "A" or "B" for side-by-side testing
    user_preferred: bool = False  # Did user prefer this variant?
    preference_reason: str = ""   # Why user preferred/rejected this answer
    
    notes: str = ""
    timestamp: str = ""
    
    def calculate_overall_score(self) -> float:
        """Calculate weighted overall score including naturalness"""
        if self.relevance == 0:
            return 0.0
        # Updated weights to include naturalness
        return (self.relevance * 0.25 + self.accuracy * 0.25 + 
                self.completeness * 0.20 + self.clarity * 0.15 + 
                self.naturalness * 0.15)


@dataclass
class RetrievalComparison:
    """Track which KG strategy (baseline vs embeddings) performed better"""
    query: str
    timestamp: str
    
    # Baseline strategy results
    baseline_answer: str = ""
    baseline_correct: bool = False
    baseline_retrieval_time: float = 0.0
    baseline_context_nodes: int = 0
    
    # Embeddings strategy results
    embeddings_answer: str = ""
    embeddings_correct: bool = False
    embeddings_retrieval_time: float = 0.0
    embeddings_context_nodes: int = 0
    
    # Comparison outcome
    winner: str = ""  # "baseline", "embeddings", or "tie"
    confidence: float = 0.0  # How confident in the winner determination
    
    def determine_winner(self) -> str:
        """Determine which strategy performed better"""
        if self.baseline_correct and not self.embeddings_correct:
            self.winner = "baseline"
            self.confidence = 1.0
        elif self.embeddings_correct and not self.baseline_correct:
            self.winner = "embeddings"
            self.confidence = 1.0
        elif self.baseline_correct and self.embeddings_correct:
            # Both correct, compare retrieval time
            if self.baseline_retrieval_time < self.embeddings_retrieval_time:
                self.winner = "baseline"
            else:
                self.winner = "embeddings"
            self.confidence = 0.5  # Lower confidence when both are correct
        else:
            self.winner = "tie"
            self.confidence = 0.0
        
        return self.winner


class FPLTriviaClassifier:
    """Classifies trivia questions using Regex + SpaCy"""
    
    def __init__(self):
        # Initialize SpaCy for NER
        self.nlp = None
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
                st.sidebar.success("✓ SpaCy NER loaded")
            except:
                st.sidebar.warning("⚠ SpaCy model not found, using regex only")
        except ImportError:
            st.sidebar.info("ℹ SpaCy not installed, using regex only")
        
        # Known entities - expanded player list
        self.known_players = [
            "Mohamed Salah", "Erling Haaland", "Harry Kane", "Kevin De Bruyne",
            "Bukayo Saka", "Son Heung-min", "Bruno Fernandes", "Phil Foden",
            "Alexander Isak", "Cole Palmer", "Martin Odegaard", "Ollie Watkins",
            "Darwin Nunez", "Gabriel Jesus", "Ivan Toney", "Declan Rice",
            "James Maddison", "Marcus Rashford", "Callum Wilson", "Alisson",
            "Ederson", "Aaron Ramsdale", "Virgil van Dijk", "Ruben Dias",
            "Trent Alexander-Arnold", "Kyle Walker", "Reece James", "Luke Shaw",
            "Salah", "Haaland", "Kane", "De Bruyne", "Saka", "Son", "Foden"
        ]
        
        self.known_teams = [
            "Arsenal", "Liverpool", "Manchester City", "Manchester United",
            "Chelsea", "Tottenham", "Newcastle", "Brighton", "Aston Villa",
            "West Ham", "Crystal Palace", "Wolves", "Fulham", "Everton",
            "Brentford", "Nottingham Forest", "Bournemouth", "Leeds",
            "Leicester", "Southampton"
        ]
    
    def extract_entities_regex(self, query: str) -> Dict[str, List[str]]:
        """Baseline: Regex-based entity extraction"""
        entities = {}
        query_lower = query.lower()
        
        # Season extraction
        season_match = re.search(r'\b(202[12]-2[23]|2021/22|2022/23)\b', query_lower)
        if season_match:
            season_raw = season_match.group(1).replace('/', '-')
            if season_raw in VALID_SEASONS:
                entities['season'] = [season_raw]
        
        # Players
        found_players = []
        for player in self.known_players:
            if player.lower() in query_lower:
                found_players.append(player)
            elif ' ' in player:
                last_name = player.split()[-1]
                if len(last_name) > 3 and last_name.lower() in query_lower:
                    found_players.append(player)
        if found_players:
            entities['player_name'] = list(set(found_players))
        
        # Teams
        found_teams = [team for team in self.known_teams if team.lower() in query_lower]
        if found_teams:
            entities['team_name'] = found_teams
        
        # Gameweek extraction
        gw_match = re.search(r'\b(?:gameweek|gw)\s*(\d+)\b', query_lower)
        if gw_match:
            entities['gameweek'] = [gw_match.group(1)]
        
        # Metrics
        if re.search(r'\b(goal|goals|scored|scoring)\b', query_lower):
            entities['metric'] = ['goals']
        elif re.search(r'\b(assist|assists)\b', query_lower):
            entities['metric'] = ['assists']
        elif re.search(r'\b(point|points|fpl)\b', query_lower):
            entities['metric'] = ['points']
        elif re.search(r'\b(bonus)\b', query_lower):
            entities['metric'] = ['bonus']
        elif re.search(r'\b(yellow|card|cards|bookings)\b', query_lower):
            entities['metric'] = ['yellow_cards']
        elif re.search(r'\b(ict|index|influence|creativity|threat)\b', query_lower):
            entities['metric'] = ['ict']
        elif re.search(r'\b(clean\s*sheet|clean\s*sheets)\b', query_lower):
            entities['metric'] = ['clean_sheets']
        
        return entities
    
    def extract_entities_spacy(self, query: str) -> Dict[str, List[str]]:
        """Enhanced: SpaCy NER extraction"""
        if not self.nlp:
            return {}
        
        entities = {}
        doc = self.nlp(query)
        
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        
        if persons:
            entities['spacy_persons'] = persons
        if orgs:
            entities['spacy_orgs'] = orgs
        
        return entities
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Combined entity extraction"""
        entities = self.extract_entities_regex(query)
        
        if self.nlp:
            spacy_entities = self.extract_entities_spacy(query)
            
            # Merge SpaCy results
            if 'spacy_persons' in spacy_entities:
                for person in spacy_entities['spacy_persons']:
                    for known_player in self.known_players:
                        if person.lower() in known_player.lower():
                            if 'player_name' not in entities:
                                entities['player_name'] = []
                            if known_player not in entities['player_name']:
                                entities['player_name'].append(known_player)
        
        return entities
    
    def classify_intent(self, query: str) -> TriviaResult:
        """Classify trivia question intent"""
        query_lower = query.lower()
        entities = self.extract_entities(query)
        
        # Intent classification with priority
        if re.search(r'who.*(?:scored|score).*(?:most|highest|top).*(?:goal|goals)', query_lower):
            intent = TriviaIntentType.TOP_GOAL_SCORER
        elif re.search(r'who.*(?:most|highest|top).*(?:assist|assists)', query_lower):
            intent = TriviaIntentType.TOP_ASSIST_PROVIDER
        elif re.search(r'(?:most|highest|top).*(?:point|points|fpl)', query_lower):
            intent = TriviaIntentType.HIGHEST_POINTS_PLAYER
        elif re.search(r'(?:which|what).*(?:fixture|fixtures|game|games|match|matches)', query_lower) and 'player_name' in entities:
            intent = TriviaIntentType.PLAYER_FIXTURES
        elif ('player_name' in entities and len(entities['player_name']) >= 2) or re.search(r'\bvs\b|\bversus\b|\bcompare\b', query_lower):
            intent = TriviaIntentType.PLAYER_VS_PLAYER
        elif re.search(r'(?:most|highest|top).*(?:bonus|bonus points)', query_lower):
            intent = TriviaIntentType.BEST_BONUS_PERFORMER
        elif re.search(r'(?:most|highest).*(?:yellow card|yellow cards|bookings|disciplined)', query_lower):
            intent = TriviaIntentType.MOST_YELLOW_CARDS
        elif re.search(r'(?:highest|best|top).*(?:ict|ict index|influence)', query_lower):
            intent = TriviaIntentType.HIGHEST_ICT_INDEX
        elif re.search(r'(?:player|which player|who).*(?:clean sheet|clean sheets)', query_lower):
            intent = TriviaIntentType.PLAYER_CLEAN_SHEETS
        elif re.search(r'gameweek|gw|week', query_lower) and re.search(r'(?:top|best|scored|most)', query_lower):
            intent = TriviaIntentType.GAMEWEEK_TOP_SCORER
        else:
            intent = TriviaIntentType.UNKNOWN
        
        confidence = 90 if intent != TriviaIntentType.UNKNOWN else 30
        cypher = self.generate_cypher_query(intent, entities)
        desc = self.get_intent_description(intent, entities)
        
        return TriviaResult(intent, confidence, entities, cypher, desc)
    
    def generate_cypher_query(self, intent: TriviaIntentType, entities: Dict) -> str:
        """
        Generate optimized Cypher queries based on intent
        """
        # Season filter
        season_filter = ""
        if 'season' in entities:
            season_filter = f"AND f.season = '{entities['season'][0]}'"
        
        if intent == TriviaIntentType.TOP_GOAL_SCORER:
            return f"""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            WHERE r.goals_scored IS NOT NULL {season_filter}
            WITH p, SUM(r.goals_scored) as total_goals, 
                COUNT(DISTINCT f) as appearances,
                f.season as season
            WHERE total_goals > 0
            RETURN p.player_name as player, 
                total_goals,
                appearances,
                season
            ORDER BY total_goals DESC, appearances DESC
            LIMIT 10
                        """.strip()
                    
        if intent == TriviaIntentType.TOP_ASSIST_PROVIDER:
                        return f"""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            WHERE r.assists IS NOT NULL {season_filter}
            WITH p, SUM(r.assists) as total_assists,
                COUNT(DISTINCT f) as appearances,
                f.season as season
            WHERE total_assists > 0
            RETURN p.player_name as player, 
                total_assists,
                appearances,
                season
            ORDER BY total_assists DESC, appearances DESC
            LIMIT 10
                        """.strip()
                    
        if intent == TriviaIntentType.HIGHEST_POINTS_PLAYER:
                        return f"""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            WHERE r.total_points IS NOT NULL {season_filter}
            WITH p, SUM(r.total_points) as total_points,
                COUNT(DISTINCT f) as appearances,
                SUM(r.goals_scored) as goals,
                SUM(r.assists) as assists,
                f.season as season
            WHERE total_points > 0
            RETURN p.player_name as player, 
                total_points,
                appearances,
                goals,
                assists,
                season
            ORDER BY total_points DESC
            LIMIT 10
                        """.strip()
        
        if intent == TriviaIntentType.PLAYER_FIXTURES and 'player_name' in entities:
            return f"""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            WHERE toLower(p.player_name) CONTAINS toLower($player_name)
                {season_filter}
            MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
            MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
            OPTIONAL MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f)
            RETURN p.player_name as player,
                home.name as home_team,
                away.name as away_team,
                gw.GW_number as gameweek,
                f.season as season,
                r.goals_scored as goals,
                r.assists as assists,
                r.total_points as points,
                r.minutes as minutes,
                r.bonus as bonus
            ORDER BY gw.GW_number
            LIMIT 38
                        """.strip()
        
        if intent == TriviaIntentType.PLAYER_VS_PLAYER and 'player_name' in entities and len(entities['player_name']) >= 2:
            return f"""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            WHERE toLower(p.player_name) IN [toLower($player1), toLower($player2)]
                {season_filter}
            WITH p, 
                COUNT(DISTINCT f) as appearances,
                SUM(r.goals_scored) as goals,
                SUM(r.assists) as assists,
                SUM(r.total_points) as points,
                SUM(r.bonus) as bonus,
                SUM(r.minutes) as minutes,
                AVG(r.ict_index) as avg_ict,
                COLLECT(DISTINCT f.season) as seasons
            RETURN p.player_name as player, 
                CASE WHEN size(seasons) = 1 THEN seasons[0] ELSE 'All Seasons' END as season,
                appearances, 
                goals, 
                assists, 
                points, 
                bonus, 
                minutes,
                round(avg_ict, 1) as avg_ict
            ORDER BY points DESC
                        """.strip()
                
        if intent == TriviaIntentType.BEST_BONUS_PERFORMER:
            return f"""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            WHERE r.bonus IS NOT NULL {season_filter}
            WITH p, SUM(r.bonus) as total_bonus,
                COUNT(DISTINCT f) as appearances,
                f.season as season
            WHERE total_bonus > 0
            RETURN p.player_name as player, 
                total_bonus,
                appearances,
                season
            ORDER BY total_bonus DESC
            LIMIT 10
                        """.strip()
                
        if intent == TriviaIntentType.MOST_YELLOW_CARDS:
            return f"""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            WHERE r.yellow_cards IS NOT NULL {season_filter}
            WITH p, SUM(r.yellow_cards) as total_yellows,
                COUNT(DISTINCT f) as appearances,
                f.season as season
            WHERE total_yellows > 0
            RETURN p.player_name as player, 
                total_yellows,
                appearances,
                season
            ORDER BY total_yellows DESC
            LIMIT 10
                        """.strip()
        
        if intent == TriviaIntentType.HIGHEST_ICT_INDEX:
            return f"""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            WHERE r.ict_index IS NOT NULL {season_filter}
            WITH p, 
                AVG(r.ict_index) as avg_ict,
                SUM(r.influence) as total_influence,
                SUM(r.creativity) as total_creativity,
                SUM(r.threat) as total_threat,
                COUNT(DISTINCT f) as appearances,
                f.season as season
            RETURN p.player_name as player, 
                round(avg_ict, 1) as avg_ict,
                round(total_influence, 1) as total_influence,
                round(total_creativity, 1) as total_creativity,
                round(total_threat, 1) as total_threat,
                appearances,
                season
            ORDER BY avg_ict DESC
            LIMIT 10
                        """.strip()
        
        if intent == TriviaIntentType.PLAYER_CLEAN_SHEETS:
            return f"""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            WHERE r.clean_sheets IS NOT NULL AND r.clean_sheets > 0 {season_filter}
            WITH p, f.season as season, SUM(r.clean_sheets) as total_clean_sheets
            OPTIONAL MATCH (p)-[:PLAYS_FOR]->(t:Team)
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
            WHERE total_clean_sheets > 0
            RETURN p.player_name as player, 
                pos.name as position,
                t.name as team,
                total_clean_sheets,
                season
            ORDER BY total_clean_sheets DESC
            LIMIT 10
                        """.strip()
        
        if intent == TriviaIntentType.GAMEWEEK_TOP_SCORER:
            if 'gameweek' in entities:
                return f"""
                MATCH (gw:Gameweek {{GW_number: toInteger($gameweek)}})-[:HAS_FIXTURE]->(f:Fixture)<-[r:PLAYED_IN]-(p:Player)
                WHERE r.total_points IS NOT NULL {season_filter}
                WITH p, f.season as season, SUM(r.total_points) as gw_points,
                    SUM(r.goals_scored) as goals, SUM(r.assists) as assists
                WHERE gw_points > 0
                RETURN p.player_name as player, 
                    gw_points, 
                    goals,
                    assists,
                    $gameweek as gameweek,
                    season
                ORDER BY gw_points DESC
                LIMIT 10
                                """.strip()
            else:
                return f"""
                MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)<-[r:PLAYED_IN]-(p:Player)
                WHERE r.total_points IS NOT NULL {season_filter}
                WITH gw, p, f.season as season, SUM(r.total_points) as gw_points
                WHERE gw_points > 0
                RETURN gw.GW_number as gameweek, 
                    p.player_name as player, 
                    gw_points,
                    season
                ORDER BY gw.GW_number DESC, gw_points DESC
                LIMIT 20
                                """.strip()
        
        return "// No query template for this intent"
    
    def get_intent_description(self, intent: TriviaIntentType, entities: Dict) -> str:
        """Human-readable description"""
        descriptions = {
            TriviaIntentType.TOP_GOAL_SCORER: "Find the top goal scorers",
            TriviaIntentType.TOP_ASSIST_PROVIDER: "Find players with most assists",
            TriviaIntentType.HIGHEST_POINTS_PLAYER: "Find highest FPL points scorers",
            TriviaIntentType.PLAYER_FIXTURES: "Find all fixtures a player appeared in",
            TriviaIntentType.PLAYER_VS_PLAYER: "Compare two players' performance",
            TriviaIntentType.BEST_BONUS_PERFORMER: "Find players with most bonus ",
            TriviaIntentType.MOST_YELLOW_CARDS: "Find players with most yellow cards (disciplinary records)",
            TriviaIntentType.HIGHEST_ICT_INDEX: "Find players with highest ICT index (Influence, Creativity, Threat)",
            # Updated description for player clean sheets
            TriviaIntentType.PLAYER_CLEAN_SHEETS: "Find players with most clean sheets",
            TriviaIntentType.GAMEWEEK_TOP_SCORER: "Find top scorers by gameweek",
            TriviaIntentType.UNKNOWN: "Unknown trivia question"
        }
        
        desc = descriptions.get(intent, "Unknown")
        if entities:
            ent_str = ", ".join([f"{k}: {', '.join(map(str, v))}" for k, v in entities.items()])
            return f"{desc} | Entities: {ent_str}"
        return desc


# ============================================================================
# SECTION 2: NEO4J QUERY EXECUTOR WITH EMBEDDING SUPPORT
# ============================================================================

class Neo4jQueryExecutor:
    """Execute queries against Neo4j with embedding support"""
    
    def __init__(self, uri: str, username: str, password: str):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.connected = True
            self.error = None
        except Exception as e:
            self.connected = False
            self.error = str(e)
    
    def execute_query_with_entities(self, query: str, entities: Dict) -> List[Dict]:
        """Execute with entity parameters"""
        if not self.connected:
            return []
        
        params = {}
        if 'player_name' in entities:
            params['player_name'] = entities['player_name'][0]
            if len(entities['player_name']) >= 2:
                params['player1'] = entities['player_name'][0]
                params['player2'] = entities['player_name'][1]
        
        if 'team_name' in entities:
            params['team_name'] = entities['team_name'][0]
        
        if 'gameweek' in entities:
            params['gameweek'] = entities['gameweek'][0]
        
        try:
            with self.driver.session() as session:
                return [dict(r) for r in session.run(query, params)]
        except Exception as e:
            st.error(f"Query error: {e}")
            return []
    
    def semantic_search(self, query_embedding: List[float], embedding_property: str, 
                            intent: TriviaIntentType = None, entities: Dict = None, 
                            top_k: int = 10) -> List[Dict]:
        """
        FIXED: Returns statistically relevant players instead of name-similar ones
        
        Strategy:
        1. Retrieve 100 candidates (wide net)
        2. Filter by relevant statistic > 0  
        3. Sort by STATISTICS FIRST, then similarity
        4. Return top_k results
        """
        if not self.connected:
            return []
        
        # Map embedding property names
        property_mapping = {
            "minilm_embedding": "embedding_minilm",
            "mpnet_embedding": "embedding_mpnet"
        }
        
        db_property = property_mapping.get(embedding_property, embedding_property)
        index_name = f'player_{db_property}'
        
        # For team queries, we need different logic since we don't have team embeddings
        if intent == TriviaIntentType.PLAYER_CLEAN_SHEETS:
            # Use player embeddings and filter by clean sheets
            season_filter = ""
            if entities and 'season' in entities:
                season_filter = "AND f.season = $season"
            
            cypher = f"""
            CALL db.index.vector.queryNodes(
                '{index_name}',
                100,
                $query_vector
            ) YIELD node AS p, score AS similarity
            OPTIONAL MATCH (p)-[:PLAYS_FOR]->(t:Team)
            OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
            OPTIONAL MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
            WHERE r.clean_sheets > 0 {season_filter}
            WITH p.player_name as player,
                pos.name as position,
                t.name as team,
                f.season as season,
                SUM(r.clean_sheets) as total_clean_sheets,
                AVG(similarity) as avg_similarity
            WHERE player IS NOT NULL AND total_clean_sheets > 0
            RETURN player,
                position,
                team,
                season,
                total_clean_sheets,
                round(avg_similarity, 3) as similarity
            ORDER BY total_clean_sheets DESC, similarity DESC
            LIMIT $top_k
            """
            
            try:
                params = {
                    'query_vector': query_embedding,
                    'top_k': top_k
                }
                if entities and 'season' in entities:
                    params['season'] = entities['season'][0]
                
                with self.driver.session() as session:
                    result = session.run(cypher, params)
                    return [dict(record) for record in result]
            except Exception as e:
                print(f"Semantic search error: {e}")
                return []
        
        # Determine which statistic to prioritize
        stat_column = "total_points"
        stat_filter = "total_points > 0"
        
        if intent == TriviaIntentType.TOP_GOAL_SCORER:
            stat_column = "total_goals"
            stat_filter = "total_goals > 0"
        elif intent == TriviaIntentType.TOP_ASSIST_PROVIDER:
            stat_column = "total_assists"
            stat_filter = "total_assists > 0"
        elif intent == TriviaIntentType.HIGHEST_POINTS_PLAYER:
            stat_column = "total_points"
            stat_filter = "total_points > 0"
        elif intent == TriviaIntentType.BEST_BONUS_PERFORMER:
            stat_column = "total_bonus"
            stat_filter = "total_bonus > 0"
        elif intent == TriviaIntentType.MOST_YELLOW_CARDS:
            stat_column = "total_yellows"
            stat_filter = "total_yellows > 0"
        elif intent == TriviaIntentType.HIGHEST_ICT_INDEX:
            stat_column = "avg_ict"
            stat_filter = "avg_ict > 0"
        
        # Season filter
        season_filter = ""
        if entities and 'season' in entities:
            season_filter = "AND f.season = $season"
        
        # KEY FIX: Get 100 candidates, filter by stats, sort by STATS not similarity
        cypher = f"""
        CALL db.index.vector.queryNodes(
            '{index_name}',
            100,
            $query_vector
        ) YIELD node AS p, score AS similarity
        OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
        OPTIONAL MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
        {season_filter.replace('AND', 'WHERE') if season_filter else ''}
        WITH p, pos, similarity,
            f.season as season,
            SUM(r.total_points) as total_points,
            SUM(r.goals_scored) as total_goals,
            SUM(r.assists) as total_assists,
            SUM(r.bonus) as total_bonus,
            SUM(r.minutes) as total_minutes,
            SUM(r.yellow_cards) as total_yellows,
            AVG(r.ict_index) as avg_ict,
            COUNT(DISTINCT f) as appearances
        WHERE {stat_filter}
        RETURN p.player_name as player,
            pos.name as position,
            similarity,
            season,
            total_points, 
            total_goals, 
            total_assists,
            total_bonus,
            total_minutes,
            total_yellows,
            round(avg_ict, 1) as avg_ict,
            appearances
        ORDER BY {stat_column} DESC, similarity DESC
        LIMIT $top_k
        """
        
        try:
            params = {
                'query_vector': query_embedding,
                'top_k': top_k
            }
            if entities and 'season' in entities:
                params['season'] = entities['season'][0]
            
            with self.driver.session() as session:
                result = session.run(cypher, params)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
            
    # /** rest of code here **/
    def semantic_search_with_filter(self, query_embedding: List[float], embedding_property: str, 
                                     intent: TriviaIntentType, top_k: int = 5) -> List[Dict]:
        """Semantic search with intent-based filtering"""
        if not self.connected:
            return []
        
        property_mapping = {
            "minilm_embedding": "embedding_minilm",
            "mpnet_embedding": "embedding_mpnet"
        }
        
        db_property = property_mapping.get(embedding_property, embedding_property)
        index_name = f'player_{db_property}'
        
        # Create intent-specific filters
        filter_clause = ""
        order_clause = "similarity DESC"
        
        if intent == TriviaIntentType.TOP_GOAL_SCORER:
            filter_clause = "WHERE total_goals > 0"
            order_clause = "total_goals DESC, similarity DESC"  # Prioritize goals over similarity
        elif intent == TriviaIntentType.TOP_ASSIST_PROVIDER:
            filter_clause = "WHERE total_assists > 0"
            order_clause = "total_assists DESC, similarity DESC"
        elif intent == TriviaIntentType.HIGHEST_POINTS_PLAYER:
            filter_clause = "WHERE total_points > 0"
            order_clause = "total_points DESC, similarity DESC"
        elif intent == TriviaIntentType.BEST_BONUS_PERFORMER:
            filter_clause = "WHERE total_bonus > 0"
            order_clause = "total_bonus DESC, similarity DESC"
        elif intent == TriviaIntentType.HIGHEST_ICT_INDEX:
            filter_clause = "WHERE avg_ict > 0"
            order_clause = "avg_ict DESC, similarity DESC"
        elif intent == TriviaIntentType.PLAYER_CLEAN_SHEETS:
            filter_clause = "WHERE total_clean_sheets > 0"
            order_clause = "total_clean_sheets DESC, similarity DESC"
        
        cypher = f"""
        CALL db.index.vector.queryNodes(
            '{index_name}',
            $top_k * 10,  
            $query_vector
        ) YIELD node AS p, score AS similarity
        OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
        OPTIONAL MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
        WITH p, pos, similarity,
             f.season as season,
             SUM(r.total_points) as total_points,
             SUM(r.goals_scored) as total_goals,
             SUM(r.assists) as total_assists,
             SUM(r.bonus) as total_bonus,
             SUM(r.minutes) as total_minutes,
             SUM(r.yellow_cards) as total_yellows,
             AVG(r.ict_index) as avg_ict,
             SUM(r.clean_sheets) as total_clean_sheets,
             COUNT(DISTINCT f) as appearances
        {filter_clause}
        RETURN p.player_name as player,
               pos.name as position,
               similarity,
               season,
               total_points, 
               total_goals, 
               total_assists,
               total_bonus,
               total_minutes,
               total_yellows,
               round(avg_ict, 1) as avg_ict,
               total_clean_sheets,
               appearances
        ORDER BY {order_clause}
        LIMIT $top_k
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher, {
                    'query_vector': query_embedding,
                    'top_k': top_k
                })
                return [dict(r) for r in result]
        except Exception as e:
            st.warning(f"Vector search unavailable: {e}")
            return []

    def hybrid_search(self, query_embedding: List[float], embedding_property: str, 
                    intent: TriviaIntentType, entities: Dict, top_k: int = 10) -> List[Dict]:
        """
        Alternative approach: Pure hybrid search with explicit weighting
        Use this if you want more control over similarity vs. statistics weighting
        """
        if not self.connected:
            return []
        
        # Map embedding property
        property_mapping = {
            "minilm_embedding": "embedding_minilm",
            "mpnet_embedding": "embedding_mpnet"
        }
        db_property = property_mapping.get(embedding_property, embedding_property)
        index_name = f'player_{db_property}'
        
        # Determine which stat to prioritize
        stat_column = "total_goals"
        if intent == TriviaIntentType.TOP_GOAL_SCORER:
            stat_column = "total_goals"
        elif intent == TriviaIntentType.TOP_ASSIST_PROVIDER:
            stat_column = "total_assists"
        elif intent == TriviaIntentType.HIGHEST_POINTS_PLAYER:
            stat_column = "total_points"
        elif intent == TriviaIntentType.BEST_BONUS_PERFORMER:
            stat_column = "total_bonus"
        elif intent == TriviaIntentType.MOST_YELLOW_CARDS:
            stat_column = "total_yellows"
        elif intent == TriviaIntentType.HIGHEST_ICT_INDEX:
            stat_column = "avg_ict"
        # Updated stat_column for player clean sheets
        elif intent == TriviaIntentType.PLAYER_CLEAN_SHEETS:
            stat_column = "total_clean_sheets"
        
        season_filter = ""
        if entities and 'season' in entities:
            season_filter = "WHERE f.season = $season"
        
        cypher = f"""
        CALL db.index.vector.queryNodes(
            '{index_name}',
            50,
            $query_vector
        ) YIELD node AS p, score AS similarity
        OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
        OPTIONAL MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
        {season_filter}
        WITH p, pos, similarity,
            f.season as season,
            SUM(r.total_points) as total_points,
            SUM(r.goals_scored) as total_goals,
            SUM(r.assists) as total_assists,
            SUM(r.bonus) as total_bonus,
            SUM(r.minutes) as total_minutes,
            SUM(r.yellow_cards) as total_yellows,
            AVG(r.ict_index) as avg_ict,
            SUM(r.clean_sheets) as total_clean_sheets,
            COUNT(DISTINCT f) as appearances
        WITH p, pos, similarity, season, total_points, total_goals, total_assists,
            total_bonus, total_minutes, total_yellows, avg_ict, total_clean_sheets, appearances,
            ({stat_column}) as stat_value
        WHERE stat_value > 0
        WITH p, pos, similarity, season, total_points, total_goals, total_assists,
            total_bonus, total_minutes, total_yellows, avg_ict, total_clean_sheets, appearances,
            stat_value,
            (similarity * 0.3 + (toFloat(stat_value) / 100.0) * 0.7) as hybrid_score
        RETURN p.player_name as player,
            pos.name as position,
            similarity,
            season,
            total_points,
            total_goals,
            total_assists,
            total_bonus,
            total_minutes,
            total_yellows,
            round(avg_ict, 1) as avg_ict,
            total_clean_sheets,
            appearances,
            hybrid_score
        ORDER BY hybrid_score DESC
        LIMIT $top_k
        """
        
        try:
            params = {
                'query_vector': query_embedding,
                'top_k': top_k
            }
            if entities and 'season' in entities:
                params['season'] = entities['season'][0]
                
            with self.driver.session() as session:
                result = session.run(cypher, params)
                return [dict(r) for r in result]
        except Exception as e:
            st.warning(f"Hybrid search unavailable: {e}")
            return self.semantic_search(query_embedding, embedding_property, intent, entities, top_k)

    @staticmethod
    def create_player_embedding_text(player_data: Dict) -> str:
        """
        Create rich text representation of player for embedding generation
        This should match the format used when creating embeddings in the database
        """
        name = player_data.get('player', 'Unknown')
        position = player_data.get('position', 'Unknown')
        goals = player_data.get('total_goals', 0)
        assists = player_data.get('total_assists', 0)
        points = player_data.get('total_points', 0)
        clean_sheets = player_data.get('total_clean_sheets', 0) # Added clean sheets
        
        return f"{name} is a {position} with {goals} goals, {assists} assists, {clean_sheets} clean sheets, and {points} FPL points"

    def close(self):
        if self.connected:
            self.driver.close()



# ============================================================================
# SECTION 3: EMBEDDING STRATEGIES (2 Transformer Models)
# ============================================================================

class QueryEmbeddingGenerator:
    """Generate query embeddings - 2 transformer models for comparison"""
    
    def __init__(self):
        self.transformer_models = {}
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Model 1: MiniLM - Fast and lightweight (384 dimensions)
            try:
                self.transformer_models['minilm'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                st.sidebar.success("✓ MiniLM-L6-v2 loaded (384-dim)")
            except:
                st.sidebar.warning("⚠ MiniLM model not available")
            
            # Model 2: MPNet - Higher quality (768 dimensions)
            try:
                self.transformer_models['mpnet'] = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                st.sidebar.success("✓ MPNet-base-v2 loaded (768-dim)")
            except:
                st.sidebar.warning("⚠ MPNet model not available")
        
        except ImportError:
            st.sidebar.info("ℹ sentence-transformers not installed (pip install sentence-transformers)")
    
    def _create_semantic_query(self, query: str, entities: Dict, intent: TriviaIntentType) -> str:
        """Create rich semantic query text for transformer models"""
        parts = [query]
        
        # Add intent context
        intent_descriptions = {
            TriviaIntentType.TOP_GOAL_SCORER: "Finding players with highest goal-scoring records",
            TriviaIntentType.TOP_ASSIST_PROVIDER: "Finding players with most assists",
            TriviaIntentType.HIGHEST_POINTS_PLAYER: "Finding players with most Fantasy Premier League points",
            TriviaIntentType.BEST_BONUS_PERFORMER: "Finding players with most bonus points",
            TriviaIntentType.MOST_YELLOW_CARDS: "Finding players with most yellow cards",
            TriviaIntentType.HIGHEST_ICT_INDEX: "Finding players with highest ICT (Influence, Creativity, Threat) index",
            # Updated description for player clean sheets
            TriviaIntentType.PLAYER_CLEAN_SHEETS: "Finding players with the most clean sheets",
            TriviaIntentType.PLAYER_VS_PLAYER: "Comparing player statistics and performance",
            TriviaIntentType.PLAYER_FIXTURES: "Finding player's fixture history and performance",
            TriviaIntentType.GAMEWEEK_TOP_SCORER: "Finding top-performing players in specific gameweeks"
        }
        
        if intent in intent_descriptions:
            parts.append(intent_descriptions[intent])
        
        # Add entity context
        if entities:
            if 'player_name' in entities:
                parts.append(f"Players: {', '.join(entities['player_name'])}")
            if 'team_name' in entities:
                parts.append(f"Teams: {', '.join(entities['team_name'])}")
            if 'gameweek' in entities:
                parts.append(f"Gameweeks: {', '.join(map(str, entities['gameweek']))}")
            if 'season' in entities:
                parts.append(f"Season: {', '.join(entities['season'])}")
        
        return " | ".join(parts)
    
    def create_minilm_embedding(self, query: str, entities: Dict, intent: TriviaIntentType) -> Optional[List[float]]:
        """Strategy 1: MiniLM-L6-v2 transformer (384 dimensions)"""
        if 'minilm' not in self.transformer_models:
            return None
        
        # Create rich query description
        query_text = self._create_semantic_query(query, entities, intent)
        
        try:
            embedding = self.transformer_models['minilm'].encode(query_text)
            return embedding.tolist()
        except Exception as e:
            st.warning(f"MiniLM encoding failed: {e}")
            return None
    
    def create_mpnet_embedding(self, query: str, entities: Dict, intent: TriviaIntentType) -> Optional[List[float]]:
        """Strategy 2: MPNet-base-v2 transformer (768 dimensions)"""
        if 'mpnet' not in self.transformer_models:
            return None
        
        # Create rich query description
        query_text = self._create_semantic_query(query, entities, intent)
        
        try:
            embedding = self.transformer_models['mpnet'].encode(query_text)
            return embedding.tolist()
        except Exception as e:
            st.warning(f"MPNet encoding failed: {e}")
            return None

# ============================================================================
# SECTION 4: LLM ANSWER GENERATION
# ============================================================================

class LLMAnswerGenerator:
    """Generate answers using 3 different LLMs"""
    
    MODELS = {
        "Phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "Flan-T5": "google/flan-t5-base",
        "Gemma-2B": "google/gemma-2-2b-it"
    }
    
    def __init__(self, model_key: str = "Phi-3-mini"):
        self.model_key = model_key
        self.model_name = self.MODELS.get(model_key, self.MODELS["Phi-3-mini"])
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.headers = {}
    
    def generate_answer(self, context: str, question: str, persona: str = "FPL Trivia Expert") -> Tuple[str, EvaluationMetrics]:
        """Generate answer and return metrics"""
        start_time = time.time()
        
        prompt = f"""You are a {persona}. Answer the trivia question using ONLY the provided information.

Context from Knowledge Graph:
{context}

Question: {question}

Instructions:
- Give a direct, concise answer
- Use only facts from the context
- If information is missing, say "I don't have enough information"
- Keep it under 3 sentences

Answer:"""
        
        # Try HuggingFace API
        answer = ""
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": prompt, "parameters": {"max_new_tokens": 100, "temperature": 0.7}},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    answer = result[0].get('generated_text', '').replace(prompt, '').strip()
            else:
                answer = self._template_answer(context, question)
        except Exception as e:
            answer = self._template_answer(context, question)
        
        response_time = time.time() - start_time
        
        # Create metrics
        metrics = EvaluationMetrics(
            model_name=self.model_key,
            response_time=response_time,
            token_count=len(prompt.split()) + len(answer.split()),
            answer_length=len(answer),
            query=question,
            answer=answer,
            context_size=len(context),
            timestamp=datetime.now().isoformat(),
            retrieval_method="unknown"
        )
        
        return answer, metrics
    
    def _template_answer(self, context: str, question: str) -> str:
        """Fallback template-based answer"""
        if not context or context == "No data found":
            return "I couldn't find information to answer this question."
        
        lines = [line.strip() for line in context.split('\n') if line.strip()]
        if len(lines) > 0:
            return f"Based on the data: {lines[0]}"
        
        return "Here's what I found: " + context[:200]


# ============================================================================
# SECTION 5: EVALUATION AND COMPARISON WITH QUALITATIVE FRAMEWORK
# ============================================================================

EVALUATION_TEST_CASES = [
    {
        "query": "Who scored the most goals in 2022-23?",
        "expected_type": "aggregate_statistic",
        "expected_entities": ["goals", "2022-23"],
        "evaluation_criteria": "Should return top 10 goal scorers with accurate counts for 2022-23 season"
    },
    {
        "query": "how many did Salah play in? fixtures did Salah play in during 2021-22?",
        "expected_type": "player_specific",
        "expected_entities": ["Salah", "fixtures", "2021-22"],
        "evaluation_criteria": "Should list all fixtures with Salah's statistics for 2021-22"
    },
    {
        "query": "Compare Haaland vs Kane in 2022-23",
        "expected_type": "comparison",
        "expected_entities": ["Haaland", "Kane", "2022-23"],
        "evaluation_criteria": "Should show side-by-side stats for both players in 2022-23"
    },
    {
        "query": "Who had the most bonus points in 2021-22?",
        "expected_type": "aggregate_statistic",
        "expected_entities": ["bonus", "2021-22"],
        "evaluation_criteria": "Should return top 10 bonus point earners for 2021-22"
    },
    # Updated test case for player clean sheets
    {
        "query": "Which player kept the most clean sheets in 2022-23?",
        "expected_type": "player_statistic",
        "expected_entities": ["clean sheets", "player", "2022-23"],
        "evaluation_criteria": "Should rank players by clean sheet count in 2022-23"
    },
    {
        "query": "Top assist providers across both seasons",
        "expected_type": "aggregate_statistic",
        "expected_entities": ["assists"],
        "evaluation_criteria": "Should return top assist providers combining both 2021-22 and 2022-23"
    }
]


class ModelComparator:
    """Compare multiple LLM models with qualitative evaluation"""
    
    def __init__(self):
        self.results: List[EvaluationMetrics] = []
        self.qualitative_evaluations: List[QualitativeEvaluation] = []
        self.retrieval_comparisons: List[RetrievalComparison] = [] # NEW: For retrieval effectiveness
    
    def add_result(self, metrics: EvaluationMetrics):
        """Add evaluation result"""
        self.results.append(metrics)
    
    def add_qualitative_evaluation(self, evaluation: QualitativeEvaluation):
        """Add human evaluation"""
        self.qualitative_evaluations.append(evaluation)
    
    def add_retrieval_comparison(self, comparison: RetrievalComparison):
        """Add retrieval comparison result"""
        self.retrieval_comparisons.append(comparison)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics by model"""
        if not self.results:
            return {}
        
        summary = {}
        for result in self.results:
            model = result.model_name
            if model not in summary:
                summary[model] = {
                    'total_queries': 0,
                    'total_response_time': 0,
                    'total_answer_length': 0,
                    'total_tokens': 0,
                    'total_cost': 0.0,
                    'total_retrieval_correct': 0
                }
            
            summary[model]['total_queries'] += 1
            summary[model]['total_response_time'] += result.response_time
            summary[model]['total_answer_length'] += result.answer_length
            summary[model]['total_tokens'] += result.token_count
            summary[model]['total_cost'] += result.calculate_cost() # Calculate cost
            if result.retrieval_correct:
                summary[model]['total_retrieval_correct'] += 1
        
        # Calculate averages
        for model in summary:
            count = summary[model]['total_queries']
            summary[model]['avg_response_time'] = summary[model]['total_response_time'] / count
            summary[model]['avg_answer_length'] = summary[model]['total_answer_length'] / count
            summary[model]['avg_tokens'] = summary[model]['total_tokens'] / count
            summary[model]['avg_cost'] = summary[model]['total_cost'] / count
            summary[model]['retrieval_accuracy'] = (summary[model]['total_retrieval_correct'] / count) * 100 if count > 0 else 0
        
        return summary
    
    def get_qualitative_summary(self) -> Dict:
        """Get summary of qualitative evaluations"""
        if not self.qualitative_evaluations:
            return {}
        
        summary = {}
        for eval in self.qualitative_evaluations:
            key = f"{eval.model_name}_{eval.retrieval_method}_{eval.embedding_model}"
            if key not in summary:
                summary[key] = {
                    'count': 0,
                    'total_relevance': 0,
                    'total_accuracy': 0,
                    'total_completeness': 0,
                    'total_clarity': 0,
                    'total_naturalness': 0, # Added naturalness
                    'total_overall': 0
                }
            
            summary[key]['count'] += 1
            summary[key]['total_relevance'] += eval.relevance
            summary[key]['total_accuracy'] += eval.accuracy
            summary[key]['total_completeness'] += eval.completeness
            summary[key]['total_clarity'] += eval.clarity
            summary[key]['total_naturalness'] += eval.naturalness # Accumulate naturalness
            summary[key]['total_overall'] += eval.calculate_overall_score()
        
        # Calculate averages
        for key in summary:
            count = summary[key]['count']
            summary[key]['avg_relevance'] = summary[key]['total_relevance'] / count
            summary[key]['avg_accuracy'] = summary[key]['total_accuracy'] / count
            summary[key]['avg_completeness'] = summary[key]['total_completeness'] / count
            summary[key]['avg_clarity'] = summary[key]['total_clarity'] / count
            summary[key]['avg_naturalness'] = summary[key]['total_naturalness'] / count # Average naturalness
            summary[key]['avg_overall'] = summary[key]['total_overall'] / count
        
        return summary
    
    def get_retrieval_comparison_summary(self) -> Dict:
        """Get summary of retrieval comparison results"""
        if not self.retrieval_comparisons:
            return {}
        
        summary = {
            "baseline_wins": 0,
            "embeddings_wins": 0,
            "ties": 0,
            "total_comparisons": len(self.retrieval_comparisons)
        }
        
        for comp in self.retrieval_comparisons:
            if comp.winner == "baseline":
                summary["baseline_wins"] += 1
            elif comp.winner == "embeddings":
                summary["embeddings_wins"] += 1
            else:
                summary["ties"] += 1
        
        return summary

    def get_comparison_table(self) -> List[Dict]:
        """Get full comparison table"""
        return [asdict(r) for r in self.results]
    
    def get_qualitative_evaluations_list(self) -> List[Dict]:
        """Get list of qualitative evaluations"""
        return [asdict(e) for e in self.qualitative_evaluations]
    
    def get_retrieval_comparison_list(self) -> List[Dict]:
        """Get list of retrieval comparison results"""
        return [asdict(c) for c in self.retrieval_comparisons]

    def export_results(self, filepath: str = "evaluation_results.json"):
        """Export all results to JSON"""
        export_data = {
            'quantitative': self.get_comparison_table(),
            'qualitative': self.get_qualitative_evaluations_list(),
            'retrieval_comparisons': self.get_retrieval_comparison_list(),
            'summary': self.get_summary_stats(),
            'qualitative_summary': self.get_qualitative_summary(),
            'retrieval_comparison_summary': self.get_retrieval_comparison_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


# ============================================================================
# SECTION 6: STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(page_title="FPL Fantasy Trivia - Enhanced", page_icon="🎯", layout="wide")
    
    st.title("FPL Trivia expert")

    
    # Season info banner
    st.info("📅 This system covers **2 seasons**: 2021-22 and 2022-23. You can query by specific season or across both!")
    
    # Initialize session state
    if 'comparator' not in st.session_state:
        st.session_state.comparator = ModelComparator()
    if 'trivia_result' not in st.session_state:
        st.session_state.trivia_result = None
    if 'embedding_generator' not in st.session_state:
        st.session_state.embedding_generator = QueryEmbeddingGenerator()
    
    # -------------------------------------------------------------------------
    # SIDEBAR CONFIGURATION
    # -------------------------------------------------------------------------
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Experiment selection
        st.subheader("📊 Experiment Mode")
        experiment_mode = st.radio(
            "Select:",
            ["Baseline Only", "Baseline + Embeddings"]
        )
        use_embeddings = ("Embeddings" in experiment_mode)
        
        # Embedding strategy selection
        if use_embeddings:
            st.subheader("🎯 Embedding Model")
            embedding_strategy = st.selectbox(
                "Choose Model:",
                [
                    "MiniLM-L6-v2 (384-dim)",
                    "MPNet-base-v2 (768-dim)"
                ]
            )
        else:
            embedding_strategy = "None"
        
        # Database connection
        st.subheader("🔌 Neo4j Connection")
        uri = st.text_input("URI", "neo4j://localhost:7687")
        username = st.text_input("Username", "neo4j")
        password = st.text_input("Password", type="password", value="password")
        
        # LLM Model Selection
        st.subheader("🤖 LLM Model")
        model_choice = st.selectbox(
            "Select Model:",
            list(LLMAnswerGenerator.MODELS.keys())
        )

        # Retrieval Effectiveness Comparison
        st.subheader("📈 Retrieval Comparison")
        compare_retrieval = st.checkbox("Enable Retrieval Comparison", value=False)
        
        # Connect button
        connect_btn = st.button("🔗 Connect to Neo4j", use_container_width=True)
        
        if connect_btn:
            executor = Neo4jQueryExecutor(uri, username, password)
            if executor.connected:
                st.session_state.executor = executor
                st.success("✅ Connected successfully!")
            else:
                st.error(f"❌ Connection failed: {executor.error}")
        
        st.markdown("---")
        st.subheader("📝 Qualitative Evaluation")
        if st.button("View Test Cases", use_container_width=True):
            st.session_state.show_test_cases = True
        
        if st.button("View Evaluation Guide", use_container_width=True):
            st.session_state.show_eval_guide = True
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = FPLTriviaClassifier()
    
    classifier = st.session_state.classifier
    emb_gen = st.session_state.embedding_generator
    
    if st.session_state.get('show_test_cases', False):
        with st.expander("📋 Evaluation Test Cases", expanded=True):
            for idx, test_case in enumerate(EVALUATION_TEST_CASES, 1):
                st.markdown(f"**Test Case {idx}:**")
                st.info(f"Query: {test_case['query']}")
                st.caption(f"Type: {test_case['expected_type']} | Criteria: {test_case['evaluation_criteria']}")
                st.markdown("---")
            
            if st.button("Close Test Cases"):
                st.session_state.show_test_cases = False
                st.rerun()
    
    if st.session_state.get('show_eval_guide', False):
        with st.expander("📖 Qualitative Evaluation Guide", expanded=True):
            st.markdown("""
            ### Rating Scale (1-5):
            
            **Relevance** - Does the answer address the question?
            - 5: Perfectly relevant
            - 4: Mostly relevant
            - 3: Somewhat relevant
            - 2: Barely relevant
            - 1: Not relevant
            
            **Accuracy** - Is the information factually correct?
            - 5: Completely accurate
            - 4: Mostly accurate with minor errors
            - 3: Partially accurate
            - 2: Several inaccuracies
            - 1: Completely inaccurate
            
            **Completeness** - Does it fully answer the question?
            - 5: Complete answer
            - 4: Mostly complete
            - 3: Partially complete
            - 2: Incomplete
            - 1: Very incomplete
            
            **Clarity** - Is the answer clear and well-structured?
            - 5: Very clear
            - 4: Clear
            - 3: Somewhat clear
            - 2: Unclear
            - 1: Very unclear

            **Naturalness** - How natural/human-like is the answer?
            - 5: Very natural, indistinguishable from human
            - 4: Mostly natural, minor artificiality
            - 3: Somewhat natural, some robotic phrasing
            - 2: Noticeably artificial
            - 1: Very unnatural, clearly machine-generated
            """)
            
            if st.button("Close Guide"):
                st.session_state.show_eval_guide = False
                st.rerun()
    
    # Example questions
    with st.expander("📝 Example Questions (10 Diverse Types)", expanded=False):
        st.markdown("""
        ### 🎯 Try These Questions:
        
        **General Queries (Both Seasons):**
        1. **TOP_GOAL_SCORER**: "Who scored the most goals?"
        2. **TOP_ASSIST_PROVIDER**: "Who had the most assists?"
        3. **HIGHEST_POINTS_PLAYER**: "Who earned the most FPL points?"
        4. **PLAYER_FIXTURES**: "how many fixtures did Salah play in?"
        5. **PLAYER_VS_PLAYER**: "Compare Salah vs Haaland"
        6. **BEST_BONUS_PERFORMER**: "Who got the most bonus?"
        7. **MOST_YELLOW_CARDS**: "Who received the most yellow cards?"
        8. **HIGHEST_ICT_INDEX**: "Who had the highest ICT index?"
        9. **PLAYER_CLEAN_SHEETS**: "Who had the most clean sheets?"
        10. **GAMEWEEK_TOP_SCORER**: "Who were the top scorers by gameweek?"
        
        **Season-Specific Queries:**
        - "Who scored the most goals in 2021-22?"
        - "Top assist providers in 2022-23?"
        - "Compare Salah vs Haaland in 2022-23"
        - "Which player had most clean sheets in 2021-22?"
        - "Who was the top scorer in gameweek 10 in 2022-23?"
        """)
    
    # Query input
    st.subheader("❓ Ask a Trivia Question")
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., Who scored the most goals in 2022-23?",
        key="trivia_query"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)
    with col2:
        if st.button("🧹 Clear History", use_container_width=True):
            st.session_state.comparator = ModelComparator()
            st.session_state.trivia_result = None
            st.rerun()
    with col3:
        st.caption("Experiment with different models and retrieval strategies for FPL trivia analysis.")
    
    # -------------------------------------------------------------------------
    # PROCESSING LOGIC
    # -------------------------------------------------------------------------
    
    if analyze_btn and query:
        # Step 1: Intent Classification & Entity Extraction
        st.markdown("---")
        st.subheader("🔬 Step 1: Intent Classification & Entity Extraction")
        
        with st.spinner("Analyzing query..."):
            trivia_result = classifier.classify_intent(query)
            st.session_state.trivia_result = trivia_result
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Classification Results**")
            st.info(f"""
            **Intent**: `{trivia_result.intent.value}`  
            **Confidence**: {trivia_result.confidence}%  
            **Description**: {trivia_result.description}
            """)
        
        with col2:
            st.markdown("**🏷️ Extracted Entities**")
            if trivia_result.entities:
                for entity_type, values in trivia_result.entities.items():
                    emoji = "📅" if entity_type == "season" else "⚽" if entity_type == "player_name" else "🏟️" if entity_type == "team_name" else "📊"
                    st.write(f"{emoji} **{entity_type}**: {', '.join(map(str, values))}")
                
                # Season context
                if 'season' in trivia_result.entities:
                    st.success(f"🎯 Filtering by season: {trivia_result.entities['season'][0]}")
                else:
                    st.info("🌐 Query spans both seasons (2021-22 & 2022-23)")
            else:
                st.write("No entities extracted")
        
        # Display Cypher Query
        with st.expander("📜 View Generated Cypher Query", expanded=False):
            st.code(trivia_result.suggested_query, language="cypher")
        
        # Step 2: Graph Retrieval
        st.markdown("---")
        st.subheader("🗄️ Step 2: Graph Retrieval")
        
        if 'executor' not in st.session_state:
            st.warning("⚠️ Please connect to Neo4j first (see sidebar)")
        else:
            executor = st.session_state.executor
            
            # Baseline retrieval
            with st.spinner("Executing Cypher query..."):
                baseline_results = executor.execute_query_with_entities(
                    trivia_result.suggested_query,
                    trivia_result.entities
                )
            
            baseline_context = ""
            if baseline_results:
                baseline_context = "\n".join([
                    str(record) for record in baseline_results[:10]
                ])
            else:
                baseline_context = "No data found"
            
            # Embeddings retrieval
            embedding_results = []
            embedding_context = ""
            embedding_model_name = "none"
            
            if use_embeddings and baseline_results:
                st.markdown("**🎯 Embedding-Based Retrieval**")
                
                with st.spinner("Generating query embedding..."):
                    query_emb = None
                    emb_property = None
                    
                    if embedding_strategy == "MiniLM-L6-v2 (384-dim)":
                        query_emb = emb_gen.create_minilm_embedding(
                            query, trivia_result.entities, trivia_result.intent
                        )
                        emb_property = "minilm_embedding"
                        embedding_model_name = "MiniLM-L6-v2-384dim"
                        if query_emb:
                            st.success(f"✓ Generated 384-dimensional MiniLM embedding")
                        else:
                            st.error("❌ MiniLM model not available")
                    
                    elif embedding_strategy == "MPNet-base-v2 (768-dim)":
                        query_emb = emb_gen.create_mpnet_embedding(
                            query, trivia_result.entities, trivia_result.intent
                        )
                        emb_property = "mpnet_embedding"
                        embedding_model_name = "MPNet-base-v2-768dim"
                        if query_emb:
                            st.success(f"✓ Generated 768-dimensional MPNet embedding")
                        else:
                            st.error("❌ MPNet model not available")
                    
                    # Show embedding vector
                    if query_emb:
                        with st.expander("🔢 View Embedding Vector"):
                            st.write(f"Dimensions: {len(query_emb)}")
                            st.write(f"First 10 values: {query_emb[:10]}")
                        
                        # Semantic search
                        embedding_results = executor.semantic_search(
                            query_emb, emb_property, 
                            intent=trivia_result.intent,  # NEW: Pass intent for stat-based sorting
                            entities=trivia_result.entities,  # NEW: Pass entities for season filtering
                            top_k=5
                        )
                        
                        if embedding_results:
                            embedding_context = "\n".join([
                                # Check if result is team-based or player-based
                                f"Player: {r.get('player', 'N/A')}, " +
                                f"Position: {r.get('position', 'N/A')}, " +
                                f"Team: {r.get('team', 'N/A')}, " +
                                f"Season: {r.get('season', 'N/A')}, " +
                                f"Clean Sheets: {r.get('total_clean_sheets', 0)}, " +
                                f"Similarity: {r.get('similarity', 0):.3f}"
                                for r in embedding_results
                            ])
                            st.success(f"✓ Found {len(embedding_results)} similar results via semantic search")
                            
                            with st.expander("🎯 Detailed Embedding Results", expanded=True):
                                for idx, result in enumerate(embedding_results, 1):
                                    if 'player' in result:
                                        st.markdown(f"**{idx}. {result.get('player', 'N/A')}** ({result.get('position', 'N/A')}, {result.get('team', 'N/A')})")
                                        
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Similarity", f"{result.get('similarity', 0):.3f}")
                                        with col2:
                                            st.metric("Clean Sheets", result.get('total_clean_sheets', 0))
                                        with col3:
                                            st.metric("Appearances", result.get('appearances', 0))
                                        with col4:
                                            st.metric("Season", result.get('season', 'N/A'))
                                    
                                    st.markdown("---")
                        else:
                            st.info("ℹ️ No embedding results (vector index may not exist in database)")
            
            # Display retrieved context
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📄 Baseline Results**")
                with st.expander("View Cypher Results", expanded=True):
                    if baseline_results:
                        st.json(baseline_results[:5])
                        st.caption(f"Showing 5/{len(baseline_results)} results")
                    else:
                        st.write("No results found")
            
            if use_embeddings:
                with col2:
                    st.markdown(f"**🎯 Embedding Results ({embedding_model_name})**")
                    with st.expander("View Semantic Search Results", expanded=True):
                        if embedding_results:
                            st.json(embedding_results)
                        else:
                            st.write("No embedding results available")
            
            # Step 3: LLM Answer Generation
            st.markdown("---")
            st.subheader("🤖 Step 3: LLM Answer Generation")
            
            # Combine contexts
            combined_context = baseline_context
            if use_embeddings and embedding_context:
                combined_context += "\n\nSemantic Search Results:\n" + embedding_context
            
            # Generate answer
            with st.spinner(f"Generating answer with {model_choice}..."):
                llm = LLMAnswerGenerator(model_choice)
                answer, metrics = llm.generate_answer(
                    combined_context,
                    query,
                    persona="FPL Fantasy Trivia Expert"
                )
                
                # Set retrieval method
                retrieval_method_used = "Baseline Only"
                if use_embeddings:
                    retrieval_method_used = "Baseline + Embeddings"
                    metrics.embedding_strategy = embedding_model_name
                
                metrics.retrieval_method = retrieval_method_used
                
                # Determine retrieval correctness (simplified: assume baseline is correct if results > 0)
                metrics.retrieval_correct = bool(baseline_results)
                metrics.kg_strategy = "embeddings" if use_embeddings else "baseline"
                
                # Add to comparator
                st.session_state.comparator.add_result(metrics)
                
                # Store current answer for evaluation
                st.session_state.current_answer = answer
                st.session_state.current_query = query
                st.session_state.current_model = model_choice
                st.session_state.current_retrieval = retrieval_method_used
                st.session_state.current_embedding = embedding_model_name
                st.session_state.current_baseline_results = baseline_results
                st.session_state.current_embedding_results = embedding_results
            
            # Display answer
            st.markdown("### 💬 Answer")
            st.success(answer)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Response Time", f"{metrics.response_time:.2f}s")
            with col2:
                st.metric("Token Count", metrics.token_count)
            with col3:
                st.metric("Answer Length", f"{metrics.answer_length} chars")
            with col4:
                st.metric("Context Size", f"{metrics.context_size} chars")
            
            st.markdown("---")
            st.subheader("⭐ Rate This Answer (Qualitative Evaluation)")
            
            with st.form(key="qualitative_eval_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    relevance = st.slider("Relevance (1-5)", 1, 5, 3, 
                                        help="Does the answer address the question?")
                    accuracy = st.slider("Accuracy (1-5)", 1, 5, 3,
                                       help="Is the information factually correct?")
                    naturalness = st.slider("Naturalness (1-5)", 1, 5, 3,
                                          help="How natural/human-like is the answer?")
                
                with col2:
                    completeness = st.slider("Completeness (1-5)", 1, 5, 3,
                                           help="Does it fully answer the question?")
                    clarity = st.slider("Clarity (1-5)", 1, 5, 3,
                                      help="Is the answer clear and well-structured?")
                
                notes = st.text_area("Additional Notes (optional)",
                                    placeholder="Any specific observations about this answer...")
                
                submit_eval = st.form_submit_button("Submit Evaluation", use_container_width=True)
                
                if submit_eval:
                    qual_eval = QualitativeEvaluation(
                        query=st.session_state.current_query,
                        answer=st.session_state.current_answer,
                        model_name=st.session_state.current_model,
                        retrieval_method=st.session_state.current_retrieval,
                        embedding_model=st.session_state.current_embedding,
                        relevance=relevance,
                        accuracy=accuracy,
                        completeness=completeness,
                        clarity=clarity,
                        naturalness=naturalness, # Added naturalness
                        notes=notes,
                        timestamp=datetime.now().isoformat()
                    )
                    
                    st.session_state.comparator.add_qualitative_evaluation(qual_eval)
                    
                    overall_score = qual_eval.calculate_overall_score()
                    st.success(f"✅ Evaluation saved! Overall Score: {overall_score:.2f}/5.0")
            
            # Retrieval Comparison Logic
            if compare_retrieval and 'current_baseline_results' in st.session_state and 'current_embedding_results' in st.session_state:
                with st.expander("📊 Retrieval Effectiveness Comparison"):
                    
                    baseline_correct = bool(st.session_state.current_baseline_results)
                    embeddings_correct = bool(st.session_state.current_embedding_results)
                    
                    baseline_retrieval_time = 0.0
                    if st.session_state.current_baseline_results is not None:
                         # Need to actually measure baseline retrieval time separately if possible
                         pass # Placeholder for actual time measurement

                    embeddings_retrieval_time = 0.0
                    if st.session_state.current_embedding_results is not None:
                         # Need to actually measure embedding retrieval time separately if possible
                         pass # Placeholder for actual time measurement

                    retrieval_comp = RetrievalComparison(
                        query=st.session_state.current_query,
                        timestamp=datetime.now().isoformat(),
                        baseline_answer=combined_context if "Semantic Search Results:" not in combined_context else baseline_context, # Approximate context
                        baseline_correct=baseline_correct,
                        baseline_retrieval_time=baseline_retrieval_time,
                        baseline_context_nodes=len(st.session_state.current_baseline_results) if st.session_state.current_baseline_results else 0,
                        embeddings_answer=combined_context if "Semantic Search Results:" in combined_context else "", # Approximate context
                        embeddings_correct=embeddings_correct,
                        embeddings_retrieval_time=embeddings_retrieval_time,
                        embeddings_context_nodes=len(st.session_state.current_embedding_results) if st.session_state.current_embedding_results else 0
                    )
                    
                    retrieval_comp.determine_winner()
                    st.session_state.comparator.add_retrieval_comparison(retrieval_comp)
                    
                    st.info(f"**Winner:** {retrieval_comp.winner.capitalize()} (Confidence: {retrieval_comp.confidence:.2f})")
                    st.write(f"Baseline Correct: {baseline_correct} | Embeddings Correct: {embeddings_correct}")
                    st.write(f"Baseline Time: {baseline_retrieval_time:.3f}s | Embeddings Time: {embeddings_retrieval_time:.3f}s")


            # Show full context
            with st.expander("📋 View Complete Context Sent to LLM"):
                st.text(combined_context)
    
    # -------------------------------------------------------------------------
    # MODEL COMPARISON & EVALUATION RESULTS
    # -------------------------------------------------------------------------
    
    if st.session_state.comparator.results:
        st.markdown("---")
        st.subheader("📊 Quantitative Evaluation Results")
        
        # Summary statistics
        summary = st.session_state.comparator.get_summary_stats()
        
        if summary:
            st.markdown("**📈 Summary Statistics by Model**")
            
            cols = st.columns(len(summary))
            for idx, (model, stats) in enumerate(summary.items()):
                with cols[idx]:
                    st.metric(
                        model,
                        f"{stats['total_queries']} queries"
                    )
                    st.caption(f"Avg Response: {stats['avg_response_time']:.2f}s")
                    st.caption(f"Avg Cost: ${stats['avg_cost']:.4f}")
                    st.caption(f"Retr. Acc.: {stats['retrieval_accuracy']:.1f}%")
        
        # Detailed comparison table
        st.markdown("**🔍 Detailed Quantitative Results**")
        
        import pandas as pd
        comparison_data = st.session_state.comparator.get_comparison_table()
        df = pd.DataFrame(comparison_data)
        
        display_cols = [
            'model_name', 'retrieval_method', 'embedding_strategy',
            'response_time', 'token_count', 'answer_length', 'cost', 'query'
        ]
        
        display_df = df[[col for col in display_cols if col in df.columns]]
        
        if 'response_time' in display_df.columns:
            display_df['response_time'] = display_df['response_time'].apply(lambda x: f"{x:.2f}s")
        if 'cost' in display_df.columns:
            display_df['cost'] = display_df['cost'].apply(lambda x: f"${x:.4f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    if st.session_state.comparator.qualitative_evaluations:
        st.markdown("---")
        st.subheader("⭐ Qualitative Evaluation Results")
        
        qual_summary = st.session_state.comparator.get_qualitative_summary()
        
        if qual_summary:
            st.markdown("**📊 Average Scores by Configuration**")
            
            import pandas as pd
            qual_df_data = []
            for config, scores in qual_summary.items():
                qual_df_data.append({
                    'Configuration': config,
                    'Count': scores['count'],
                    'Avg Relevance': f"{scores['avg_relevance']:.2f}",
                    'Avg Accuracy': f"{scores['avg_accuracy']:.2f}",
                    'Avg Completeness': f"{scores['avg_completeness']:.2f}",
                    'Avg Clarity': f"{scores['avg_clarity']:.2f}",
                    'Avg Naturalness': f"{scores['avg_naturalness']:.2f}", # Added naturalness avg
                    'Overall Score': f"{scores['avg_overall']:.2f}"
                })
            
            qual_df = pd.DataFrame(qual_df_data)
            st.dataframe(qual_df, use_container_width=True, hide_index=True)
        
        # Individual evaluations
        with st.expander("📝 View Individual Evaluations"):
            for idx, eval in enumerate(st.session_state.comparator.qualitative_evaluations, 1):
                st.markdown(f"**Evaluation {idx}**")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.info(f"**Query:** {eval.query}")
                    st.write(f"**Answer:** {eval.answer}")
                    if eval.notes:
                        st.caption(f"Notes: {eval.notes}")
                
                with col2:
                    st.write(f"**Model:** {eval.model_name}")
                    st.write(f"**Retrieval:** {eval.retrieval_method}")
                    st.write(f"**Embedding:** {eval.embedding_model}")
                    st.metric("Overall Score", f"{eval.calculate_overall_score():.2f}/5.0")
                    st.caption(f"R:{eval.relevance} A:{eval.accuracy} C:{eval.completeness} Cl:{eval.clarity} N:{eval.naturalness}") # Included Naturalness
                
                st.markdown("---")

    if compare_retrieval and st.session_state.comparator.retrieval_comparisons:
        st.markdown("---")
        st.subheader("🚀 Retrieval Effectiveness Comparison")
        
        retrieval_summary = st.session_state.comparator.get_retrieval_comparison_summary()
        
        if retrieval_summary:
            st.markdown("**📊 Summary of Retrieval Strategy Performance**")
            
            cols = st.columns(3)
            with cols[0]:
                st.metric("Baseline Wins", retrieval_summary["baseline_wins"])
            with cols[1]:
                st.metric("Embeddings Wins", retrieval_summary["embeddings_wins"])
            with cols[2]:
                st.metric("Ties", retrieval_summary["ties"])
            
            st.caption(f"Total comparisons: {retrieval_summary['total_comparisons']}")

        with st.expander("🔍 Detailed Retrieval Comparisons"):
            for idx, comp in enumerate(st.session_state.comparator.get_retrieval_comparison_list(), 1):
                st.markdown(f"**Comparison {idx}**")
                st.info(f"**Query:** {comp['query']}")
                st.write(f"**Winner:** {comp['winner'].capitalize()} (Confidence: {comp['confidence']:.2f})")
                st.write(f"Baseline Correct: {comp['baseline_correct']} | Embeddings Correct: {comp['embeddings_correct']}")
                st.write(f"Baseline Time: {comp['baseline_retrieval_time']:.3f}s | Embeddings Time: {comp['embeddings_retrieval_time']:.3f}s")
                st.markdown("---")
    
    # Download all results
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("📥 Download All Results (JSON)", use_container_width=True):
            st.session_state.comparator.export_results()
            st.success("✅ Results exported to evaluation_results.json")
    
    with col2:
        # Create CSV for quantitative results
        import pandas as pd
        csv = pd.DataFrame(st.session_state.comparator.get_comparison_table()).to_csv(index=False)
        st.download_button(
            "📥 Download Quantitative (CSV)",
            csv,
            "quantitative_results.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col3:
        st.caption("Download all collected evaluation data for further analysis.")

if __name__ == "__main__":
    main()
