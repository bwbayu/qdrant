- run neo4j in docker using docker compose
docker-compose -f docker-compose.neo4j.yml up -d

PYTHONPATH=agents/src python3.10 -m agents.tests.app