# mrouter

## usage
start the gateway
```python gateway.py```


## config
you should set LLM api env first, support openrouter, openai.  
openrouter has been tested lightly.

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
