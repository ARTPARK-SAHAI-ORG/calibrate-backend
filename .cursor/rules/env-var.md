---
description: "Handling addition, removal, or modification of environment variables"
alwaysApply: true
---

If any environment variables are added/changed/removed, update all four of these: src/.env.example, docker-compose.yml, the github actions workflows that export them, and ENV.md, which is the reference readers actually consult and the one most often forgotten.
