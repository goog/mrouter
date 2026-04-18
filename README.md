# mrouter

## config
you should set LLM api env first, currently only support openrouter, openai.  
openrouter has been tested lightly.

config model on **openclaw**
```
"providers": {
      "local": {
        "baseUrl": "http://127.0.0.1:8000/v1",
        "api": "openai-completions",
        "models": [
          {
            "id": "my-model",
            "name": "My Local Model",
            "compat": {
              "requiresStringContent": true,
              "supportsTools": true
            },
            "cost": {
              "input": 0,
              "output": 0,
              "cacheRead": 0,
              "cacheWrite": 0
            },
            "contextWindow": 200000,
            "maxTokens": 4096
          }
        ]
      },

```

## usage
start the gateway  
```python gateway.py```

## example
```
# Auto-route (classifier decides difficulty)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 42 times 7?"}]}'

# Force complex model
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Design a microservices architecture"}],"difficulty":"complex"}'

# Privacy mode → always routes to local Ollama
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Summarize this confidential doc"}],"privacy_mode":true}'

# Streaming (SSE)
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Count 1 to 5"}],"stream":true}'
```
