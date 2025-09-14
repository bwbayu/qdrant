- run neo4j in docker using docker compose
docker-compose -f docker-compose.neo4j.yml up -d

<!-- Run Agents -->
PYTHONPATH=agents/src python3.10 -m agents.tests.app

<!-- Gradio -->
python3.10 -m app.client.app

- run qdrant in docker using docker compose : 
docker-compose -f docker-compose.qdrant.yml up -d

- test
