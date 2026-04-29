"""
Start the sentiment pipeline web app.

Run from the project root:
    python -m pipeline
    python -m pipeline.client.local_app
    python -m pipeline.client.local_app --port 8080

This opens your default browser to http://localhost:5000 where you can:
  1. Select a market (US / China) and asset (commodity / stock)
  2. Fetch news from live sources
  3. Filter headlines for relevance using an LLM
  4. Analyze sentiment using a local LoRA-fine-tuned model
  5. View results in the browser
"""
from pipeline.client.local_app import main

if __name__ == "__main__":
    main()
