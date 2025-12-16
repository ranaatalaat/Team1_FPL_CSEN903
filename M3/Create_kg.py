from neo4j import GraphDatabase
import csv
import os

class FPLKnowledgeGraph:
    def __init__(self, uri, username, password):
        """Initialize Neo4j driver"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ Database cleared!")

    def create_constraints(self):
        with self.driver.session() as session:
            constraints = [
                """
                CREATE CONSTRAINT season_name IF NOT EXISTS
                FOR (s:Season) REQUIRE s.season_name IS UNIQUE
                """,
                """
                CREATE CONSTRAINT gameweek_id IF NOT EXISTS
                FOR (gw:Gameweek) REQUIRE (gw.season, gw.GW_number) IS UNIQUE
                """,
                """
                CREATE CONSTRAINT fixture_id IF NOT EXISTS
                FOR (f:Fixture) REQUIRE (f.season, f.fixture_number) IS UNIQUE
                """,
                """
                CREATE CONSTRAINT team_name IF NOT EXISTS
                FOR (t:Team) REQUIRE t.name IS UNIQUE
                """,
                """
                CREATE CONSTRAINT player_id IF NOT EXISTS
                FOR (p:Player) REQUIRE (p.player_name, p.player_element) IS UNIQUE
                """,
                """
                CREATE CONSTRAINT position_name IF NOT EXISTS
                FOR (pos:Position) REQUIRE pos.name IS UNIQUE
                """
            ]

            for query in constraints:
                session.run(query)

        print("✓ Constraints created!")

    def load_data(self):
        csv_file = 'fpl_two_seasons.csv'

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"{csv_file} not found!")

        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = list(reader)

        print(f"✓ Loaded {len(data)} rows from {csv_file}")

        with self.driver.session() as session:
            self.create_seasons(session, data)
            self.create_gameweeks(session, data)
            self.create_teams(session, data)
            self.create_positions(session, data)
            self.create_fixtures(session, data)
            self.create_players(session, data)
            self.create_relationships(session, data)


    def create_seasons(self, session, data):
        seasons = set(row['season'] for row in data)

        for season in seasons:
            session.run("""
                MERGE (s:Season {season_name: $season})
            """, season=season)

        print(f"✓ Created {len(seasons)} Season nodes")

    def create_gameweeks(self, session, data):
        gameweeks = set((row['season'], int(row['GW'])) for row in data)

        for season, gw_num in gameweeks:
            session.run("""
                MERGE (gw:Gameweek {season: $season, GW_number: $gw_num})
                WITH gw
                MATCH (s:Season {season_name: $season})
                MERGE (s)-[:HAS_GW]->(gw)
            """, season=season, gw_num=gw_num)

        print(f"✓ Created {len(gameweeks)} Gameweek nodes")

    def create_teams(self, session, data):
        teams = set()

        for row in data:
            teams.add(row['home_team'])
            teams.add(row['away_team'])

        for team in teams:
            session.run("""
                MERGE (t:Team {name: $team})
            """, team=team)

        print(f"✓ Created {len(teams)} Team nodes")

    def create_positions(self, session, data):
        positions = set(row['position'] for row in data)

        for position in positions:
            session.run("""
                MERGE (p:Position {name: $position})
            """, position=position)

        print(f"✓ Created {len(positions)} Position nodes")

    def create_fixtures(self, session, data):
        fixtures = {}

        for row in data:
            key = (row['season'], int(row['fixture']))
            if key not in fixtures:
                fixtures[key] = row

        for (season, fixture_num), row in fixtures.items():
            session.run("""
                MERGE (f:Fixture {season: $season, fixture_number: $fixture_num})
                SET f.kickoff_time = $kickoff_time
                WITH f
                MATCH (gw:Gameweek {season: $season, GW_number: $gw_num})
                MERGE (gw)-[:HAS_FIXTURE]->(f)
                WITH f
                MATCH (home:Team {name: $home_team})
                MERGE (f)-[:HAS_HOME_TEAM]->(home)
                WITH f
                MATCH (away:Team {name: $away_team})
                MERGE (f)-[:HAS_AWAY_TEAM]->(away)
            """,
                season=season,
                fixture_num=fixture_num,
                kickoff_time=row.get('kickoff_time', ''),
                gw_num=int(row['GW']),
                home_team=row['home_team'],
                away_team=row['away_team']
            )

        print(f"✓ Created {len(fixtures)} Fixture nodes")

    def create_players(self, session, data):
        players = {}

        for row in data:
            key = (row['name'], int(row['element']))
            if key not in players:
                players[key] = {
                    'name': row['name'],
                    'element': int(row['element']),
                    'positions': set()
                }
            players[key]['positions'].add(row['position'])

        for (name, element), info in players.items():
            session.run("""
                MERGE (p:Player {player_name: $name, player_element: $element})
            """, name=name, element=element)

            for position in info['positions']:
                session.run("""
                    MATCH (p:Player {player_name: $name, player_element: $element})
                    MATCH (pos:Position {name: $position})
                    MERGE (p)-[:PLAYS_AS]->(pos)
                """, name=name, element=element, position=position)

        print(f"✓ Created {len(players)} Player nodes")

    def create_relationships(self, session, data):
        for i, row in enumerate(data):
            session.run("""
                MATCH (p:Player {player_name: $name, player_element: $element})
                MATCH (f:Fixture {season: $season, fixture_number: $fixture_num})
                MERGE (p)-[r:PLAYED_IN]->(f)
                SET r.minutes = $minutes,
                    r.goals_scored = $goals_scored,
                    r.assists = $assists,
                    r.total_points = $total_points,
                    r.bonus = $bonus,
                    r.clean_sheets = $clean_sheets,
                    r.goals_conceded = $goals_conceded,
                    r.own_goals = $own_goals,
                    r.penalties_saved = $penalties_saved,
                    r.penalties_missed = $penalties_missed,
                    r.yellow_cards = $yellow_cards,
                    r.red_cards = $red_cards,
                    r.saves = $saves,
                    r.bps = $bps,
                    r.influence = $influence,
                    r.creativity = $creativity,
                    r.threat = $threat,
                    r.ict_index = $ict_index,
                    r.form = $form
            """,
                name=row['name'],
                element=int(row['element']),
                season=row['season'],
                fixture_num=int(row['fixture']),
                minutes=int(row.get('minutes', 0)),
                goals_scored=int(row.get('goals_scored', 0)),
                assists=int(row.get('assists', 0)),
                total_points=int(row.get('total_points', 0)),
                bonus=int(row.get('bonus', 0)),
                clean_sheets=int(row.get('clean_sheets', 0)),
                goals_conceded=int(row.get('goals_conceded', 0)),
                own_goals=int(row.get('own_goals', 0)),
                penalties_saved=int(row.get('penalties_saved', 0)),
                penalties_missed=int(row.get('penalties_missed', 0)),
                yellow_cards=int(row.get('yellow_cards', 0)),
                red_cards=int(row.get('red_cards', 0)),
                saves=int(row.get('saves', 0)),
                bps=int(row.get('bps', 0)),
                influence=float(row.get('influence', 0)),
                creativity=float(row.get('creativity', 0)),
                threat=float(row.get('threat', 0)),
                ict_index=float(row.get('ict_index', 0)),
                form=float(row.get('form', 0))
            )

            if (i + 1) % 1000 == 0:
                print(f"  → Processed {i + 1} PLAYED_IN relationships")

        print("✓ Created all PLAYED_IN relationships")

def read_config(config_file='config.txt'):
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key] = value
    return config

def main():
    config = read_config('config.txt')
    kg = FPLKnowledgeGraph(config['URI'], config['USERNAME'], config['PASSWORD'])

    try:
        print("\n Starting Knowledge Graph Construction...\n")
        kg.clear_database()
        kg.create_constraints()
        kg.load_data()
        print("\n Knowledge Graph construction completed successfully!")
    finally:
        kg.close()

if __name__ == "__main__":
    main()
