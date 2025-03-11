# Sentiment Analysis with LLM API

## Overview

This project implements a **sentiment analysis tool** using the Groq API. It evaluates the sentiment of a given text input using different **prompting strategies** and estimates confidence levels based on **LLM self-evaluation**. The tool is designed to classify text as **Positive, Negative, or Neutral** while providing an **explanation** for the classification.

## Features

- **Three Prompting Strategies**:
  - **Basic**: Directly asks for sentiment classification.
  - **Structured**: Uses a predefined format for structured responses.
  - **Few-Shot**: Provides examples to guide the model’s classification.
- **Confidence Estimation**: Uses a self-evaluation prompt since Groq API does not support logprobs.
- **Threshold Filtering**: Filters results with confidence below the defined threshold.
- **Error Handling**: Handles API failures and missing responses gracefully.

## Installation

### Prerequisites

- Groq API Key
- Required libraries:
  ```
	-openai
	-os
	-dotenv
	-requests
	-re
	-json
	-collections
  ```

### Setup

1. Create a `.env` file in the project directory and add your API credentials:
   ```ini
   API_KEY=your_api_key_here
   BASE_URL=https://api.groq.com
   LLM_MODEL=your_model_here
   ```
2. Run the script:
   ```bash
   python taming_llm.py
   ```

## Usage

### Running Sentiment Analysis

Modify `test_text` in `taming_llm.py` to analyze different texts:

```python
if __name__ == "__main__":
    client = LLMClient(confidence_threshold=0.7)
    test_text = "I’m still sorting out my thoughts on this product. It has a sleek design..."
    results = client.compare_prompt_strategies(test_text)
    print(json.dumps(results, indent=4))
```

This will output sentiment classifications for **Basic, Structured, and Few-Shot** prompts.
