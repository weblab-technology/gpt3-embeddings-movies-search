```bash
curl https://api.openai.com/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"input": "Sample document text goes here",
       "model":"text-similarity-babbage-001"}' > out1.txt
```