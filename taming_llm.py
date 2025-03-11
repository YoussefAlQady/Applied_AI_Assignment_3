import os
import requests
import re
import json
from collections import Counter
from dotenv import load_dotenv

class LLMClient:
    def __init__(self, confidence_threshold=0.80):
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("LLM_MODEL")
        self.confidence_threshold = confidence_threshold  # Set confidence threshold

        if not self.api_key or not self.base_url or not self.model:
            raise ValueError("Missing API credentials or model information in .env file")

    def complete(self, prompt, max_tokens=500, temperature=0.7):
        """
        Sends a structured prompt to the Groq API and returns the response.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with API: {e}")
            return None

    def create_structured_prompt(self, text):
        """
        Creates a structured prompt that instructs the LLM to analyze sentiment
        in a very specific format.
        """
        prompt = f"""
        # Sentiment Analysis Report
        
        ## Task
        You are an advanced AI trained to analyze the sentiment of a given text.
        Your task is to classify the sentiment as **Positive, Negative, or Neutral**.
        If the text includes happy tones and signs of satisfaction, then this would be positive. The text does not need to explicitly use words like happy, good, or satisfied for it to be classified as positive sentiment.
        If the text includes a sad tone and signs of dissatisfaction, then this would be negative. The text also does not need to explicitly use negative language to be classified as negative.
        If the text demonstrates a lack of emotion and no explicit signs of satisfaction or dissatisfaction, then this would be negative.
        
        ## Guidelines
        - Always output results in a structured format (see below).
        - Consider the **tone, word choice, and context** when determining sentiment.
        - Give a brief explanation for your classification.
        
        ## Input Text
        {text}
        
        ## Response Format
        Sentiment: [Positive | Negative | Neutral]
        Explanation: [Brief explanation of classification]
        
        ## Analysis
        """
        return prompt

    def structured_sentiment_analysis(self, text, num_samples=3):
        """
        Uses structured prompting to generate sentiment analysis with confidence estimation.
        - Runs completions at a fixed temperature (0.5) to ensure consistency.
        - Uses LLM self-evaluation for confidence estimation.
        - Filters results based on confidence threshold.
        """
        prompt = self.create_structured_prompt(text)  # Ensure we use the structured prompt

        # Run sentiment classification
        completion = self.complete(prompt, temperature=0.5)
        if not completion:
            return {
                "Sentiment": "Unknown",
                "Confidence": 0.5,
                "Explanation": "No response from API."
            }

        extracted_data = self.extract_sections(completion)
        most_common_sentiment = extracted_data["Sentiment"]

        # Get LLM self-evaluation confidence, using the actual structured prompt
        llm_confidence = self.get_llm_confidence(prompt, most_common_sentiment)

        # **Threshold Filtering**
        if llm_confidence < self.confidence_threshold:
            return {
                "Sentiment": "Uncertain",
                "Confidence": round(llm_confidence, 2),
                "Explanation": "Confidence too low to determine sentiment reliably."
            }

        return {
            "Sentiment": most_common_sentiment,
            "Confidence": round(llm_confidence, 2),
            "Explanation": extracted_data["Explanation"]
        }

    def get_llm_confidence(self, prompt, sentiment):
        """
        Asks the LLM to self-evaluate its confidence in the classification.
        Now uses the actual prompt used in the classification.
        """
        confidence_prompt = f"""
        You classified the following text as **{sentiment}**.
        On a scale from 0 to 1, how confident are you in this classification? Be frugal with your confidence score. Do not simply say 1.0 or 0.9 when you are confident. Approach it with the mindset that nothing is perfect and everything can be improved.

        ## Prompt Used
        {prompt}
        
        ## Response Format
        Confidence: [Value between 0 and 1]
        """

        completion = self.complete(confidence_prompt, max_tokens=10, temperature=0.0)  # Low temp for consistency

        if not completion:  # Handle None response
            return 0.5  # Default confidence if API fails

        confidence_match = re.search(r"Confidence:\s*([\d.]+)", completion)

        return float(confidence_match.group(1)) if confidence_match else 0.5  # Default to 0.5 if not found

    def extract_sections(self, completion):
        """
        Extracts structured data from the model's response, specifically:
        - Sentiment (Positive, Negative, Neutral)
        - Explanation (A brief text description)
        """
        if not completion:
            return {"Sentiment": "Unknown", "Explanation": "No response from API"}

        sentiment_match = re.search(r"Sentiment:\s*(Positive|Negative|Neutral)", completion)
        explanation_match = re.search(r"Explanation:\s*(.+)", completion)

        return {
            "Sentiment": sentiment_match.group(1) if sentiment_match else "Unknown",
            "Explanation": explanation_match.group(1) if explanation_match else "No explanation provided."
        }

    def compare_prompt_strategies(self, text):
        """
        Runs sentiment analysis using three different prompt strategies (basic, structured, few-shot).
        Also retrieves confidence scores for each classification and enforces the confidence threshold.
        """
        prompts = {
            "Basic": f"Classify the sentiment of the following text as Positive, Negative, or Neutral. Provide a brief explanation for your choice.\n\nText: {text}\n\nSentiment: [Positive | Negative | Neutral]\nExplanation: [Brief explanation]",
            "Structured": self.create_structured_prompt(text),
            "Few-Shot": f"""
            You are an advanced AI trained to analyze the sentiment of a given text. Your task is to classify the sentiment as Positive, Negative, or Neutral. Below are some examples to help you understand the task.
            
            Example 1:
            Text: "I’ve been using this product for a little while now, and it’s been a decent experience overall. It works as expected, and I appreciate how straightforward it is to use. There are a few small quirks, like the materials feeling a bit lightweight, but nothing that’s been a dealbreaker so far. It’s not perfect, but it gets the job done without any major issues."
            Sentiment: Positive
            Explanation: The text uses a decent tone and demonstrates satisfaction."
            
            Example 2:
            Text: "This product has some good qualities, like its simple setup and functional design, but I’m not entirely sold on it. A few things, like the less-than-sturdy feel and the occasional inconvenience, make me question its long-term value. It’s not terrible, but I expected a bit more for the price. It’s okay, but I’m not sure I’d buy it again."
            Sentiment: Negative
            Explanation: The text uses negative phrases like "would not buy again" and demonstrates a lack of satisfaction."
            
            Now, classify the sentiment of the following text:
            
            Text: {text}
            
            Sentiment: [Positive | Negative | Neutral]
            Explanation: [Brief explanation of classification]
            """
        }

        results = {}
        for strategy, prompt in prompts.items():
            completion = self.complete(prompt, temperature=0.5)

            if not completion:
                results[strategy] = {
                    "Sentiment": "Unknown",
                    "Confidence": 0.5,
                    "Explanation": "No response from API."
                }
                continue  # Skip this iteration if no response

            extracted_data = self.extract_sections(completion)

            # Get sentiment classification
            sentiment = extracted_data["Sentiment"]

            # Compute confidence using the API response instead of the original prompt
            confidence = self.get_llm_confidence(completion, sentiment) if sentiment != "Unknown" else 0.5

            # **Enforce Confidence Threshold**
            if confidence < self.confidence_threshold:
                results[strategy] = {
                    "Sentiment": "Uncertain",
                    "Confidence": round(confidence, 2),
                    "Explanation": "Confidence too low to determine sentiment reliably."
                }
            else:
                results[strategy] = {
                    "Sentiment": sentiment,
                    "Confidence": round(confidence, 2),
                    "Explanation": extracted_data["Explanation"]
                }

        return results


if __name__ == "__main__":
    client = LLMClient(confidence_threshold=0.7)

    # Sample test text
    test_text = "I’m still sorting out my thoughts on this product. It has a sleek design, is easy to set up, and works reliably without major issues. However, the build quality feels inconsistent, with some parts seeming less durable, and I’ve noticed minor wear after just a couple of weeks. The battery life is fine but not great. It does what it’s supposed to, but I’m not sure if the positives fully outweigh the flaws. For now, it’s serving its purpose."

    results = client.compare_prompt_strategies(test_text)

    print("\n=== Sentiment Analysis Comparison ===")
    print(json.dumps(results, indent=4))
