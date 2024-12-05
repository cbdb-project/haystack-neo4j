# heystack-neo4j
 

## Run Neo4j in Docker

```
docker run --restart always --publish=7474:7474 --publish=7687:7687 neo4j:5.25.1
```

## Remove everything in Neo4j

```
MATCH (n)
DETACH DELETE n;
```