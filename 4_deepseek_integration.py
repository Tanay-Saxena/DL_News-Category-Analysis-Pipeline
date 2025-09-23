"""
DeepSeek Integration Module
Uses DeepSeek API for category prediction and natural language explanations
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import os
from datetime import datetime

class DeepSeekReasoning:
    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-reasoner"):
        """
        Initialize DeepSeek Reasoning integration

        Args:
            api_key (str): DeepSeek API key (can also be set via environment variable)
            base_url (str): Base URL for DeepSeek API
            model (str): Model name to use
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.base_url = base_url
        self.model = model

        if not self.api_key:
            print("Warning: No API key provided. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests

        # Cache for responses
        self.response_cache = {}

    def _make_request(self,
                     prompt: str,
                     max_tokens: int = 1000,
                     temperature: float = 0.7) -> Optional[Dict]:
        """
        Make a request to DeepSeek API with rate limiting

        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature

        Returns:
            Dict: API response or None if error
        """
        if not self.api_key:
            print("Error: No API key available")
            return None

        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        # Check cache
        cache_key = f"{prompt}_{max_tokens}_{temperature}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )

            self.last_request_time = time.time()

            if response.status_code == 200:
                result = response.json()
                self.response_cache[cache_key] = result
                return result
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
                return None

        except Exception as e:
            print(f"Error making API request: {e}")
            return None

    def predict_category(self,
                        text: str,
                        available_categories: List[str],
                        context: Optional[str] = None) -> Dict[str, Any]:
        """
        Predict the most likely category for input text

        Args:
            text (str): Input text to classify
            available_categories (List[str]): List of possible categories
            context (str): Optional context about the dataset

        Returns:
            Dict: Prediction results with confidence and explanation
        """
        categories_str = ", ".join(available_categories)

        prompt = f"""
You are an expert news categorization system. Given a news article description, predict the most appropriate category.

Available categories: {categories_str}

News description: "{text}"

Please provide your analysis in the following JSON format:
{{
    "predicted_category": "CATEGORY_NAME",
    "confidence": 0.95,
    "reasoning": "Brief explanation of why this category was chosen",
    "alternative_categories": ["CATEGORY1", "CATEGORY2"],
    "key_indicators": ["indicator1", "indicator2", "indicator3"]
}}

Focus on:
1. Key words and phrases that indicate the category
2. The main subject matter and tone
3. The type of content (news, opinion, analysis, etc.)
4. The target audience and context

Respond only with valid JSON.
"""

        if context:
            prompt += f"\n\nAdditional context: {context}"

        response = self._make_request(prompt, max_tokens=500, temperature=0.3)

        if not response:
            return {
                "predicted_category": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": "API request failed",
                "alternative_categories": [],
                "key_indicators": [],
                "error": "API request failed"
            }

        try:
            # Extract content from response
            content = response['choices'][0]['message']['content']

            # Try to parse JSON
            result = json.loads(content)

            # Validate result
            if 'predicted_category' not in result:
                raise ValueError("Invalid response format")

            return result

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing API response: {e}")
            return {
                "predicted_category": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": f"Failed to parse response: {str(e)}",
                "alternative_categories": [],
                "key_indicators": [],
                "error": "Response parsing failed"
            }

    def generate_explanation(self,
                           text: str,
                           predicted_category: str,
                           ground_truth_category: Optional[str] = None,
                           similar_articles: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Generate a detailed explanation for the prediction

        Args:
            text (str): Input text
            predicted_category (str): Predicted category
            ground_truth_category (str): Actual category (if available)
            similar_articles (List[Dict]): Similar articles for context

        Returns:
            Dict: Detailed explanation
        """
        similar_context = ""
        if similar_articles:
            similar_texts = [article.get('text', '')[:100] + "..." for article in similar_articles[:3]]
            similar_context = f"\n\nSimilar articles found:\n" + "\n".join([f"- {text}" for text in similar_texts])

        ground_truth_context = ""
        if ground_truth_category:
            ground_truth_context = f"\n\nGround truth category: {ground_truth_category}"

        prompt = f"""
You are an expert news analyst. Provide a detailed explanation for why a news article was categorized as "{predicted_category}".

Article text: "{text}"
{ground_truth_context}{similar_context}

Please provide a comprehensive analysis in the following JSON format:
{{
    "explanation": "Detailed explanation of the categorization decision",
    "key_phrases": ["phrase1", "phrase2", "phrase3"],
    "tone_analysis": "Analysis of the article's tone and style",
    "subject_matter": "Main subject matter and themes",
    "category_characteristics": "Why this fits the predicted category",
    "confidence_factors": ["factor1", "factor2", "factor3"],
    "potential_ambiguities": "Any potential ambiguities or edge cases",
    "comparison_with_alternatives": "Why this category over others"
}}

Be thorough and analytical in your explanation.
Respond only with valid JSON.
"""

        response = self._make_request(prompt, max_tokens=800, temperature=0.4)

        if not response:
            return {
                "explanation": "Failed to generate explanation due to API error",
                "key_phrases": [],
                "tone_analysis": "Unable to analyze",
                "subject_matter": "Unable to determine",
                "category_characteristics": "Unable to determine",
                "confidence_factors": [],
                "potential_ambiguities": "API error",
                "comparison_with_alternatives": "Unable to compare",
                "error": "API request failed"
            }

        try:
            content = response['choices'][0]['message']['content']
            result = json.loads(content)
            return result

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing explanation response: {e}")
            return {
                "explanation": f"Failed to parse explanation: {str(e)}",
                "key_phrases": [],
                "tone_analysis": "Unable to analyze",
                "subject_matter": "Unable to determine",
                "category_characteristics": "Unable to determine",
                "confidence_factors": [],
                "potential_ambiguities": "Parsing error",
                "comparison_with_alternatives": "Unable to compare",
                "error": "Response parsing failed"
            }

    def analyze_mismatch(self,
                        text: str,
                        predicted_category: str,
                        ground_truth_category: str,
                        similar_articles: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Analyze why a prediction was incorrect

        Args:
            text (str): Input text
            predicted_category (str): Predicted category
            ground_truth_category (str): Actual category
            similar_articles (List[Dict]): Similar articles for context

        Returns:
            Dict: Mismatch analysis
        """
        similar_context = ""
        if similar_articles:
            similar_texts = [article.get('text', '')[:100] + "..." for article in similar_articles[:3]]
            similar_context = f"\n\nSimilar articles found:\n" + "\n".join([f"- {text}" for text in similar_texts])

        prompt = f"""
You are an expert news analyst. Analyze why a categorization prediction was incorrect.

Article text: "{text}"
Predicted category: {predicted_category}
Actual category: {ground_truth_category}
{similar_context}

Please provide a detailed analysis in the following JSON format:
{{
    "mismatch_reason": "Primary reason for the mismatch",
    "predicted_indicators": ["indicator1", "indicator2"],
    "actual_indicators": ["indicator1", "indicator2"],
    "confusing_elements": "Elements that led to confusion",
    "correct_interpretation": "How the text should be interpreted",
    "category_boundaries": "Analysis of category boundaries and overlaps",
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "similarity_analysis": "Analysis of similarity to other categories"
}}

Be critical and analytical in your assessment.
Respond only with valid JSON.
"""

        response = self._make_request(prompt, max_tokens=700, temperature=0.5)

        if not response:
            return {
                "mismatch_reason": "API error prevented analysis",
                "predicted_indicators": [],
                "actual_indicators": [],
                "confusing_elements": "Unable to analyze",
                "correct_interpretation": "Unable to determine",
                "category_boundaries": "Unable to analyze",
                "improvement_suggestions": [],
                "similarity_analysis": "Unable to analyze",
                "error": "API request failed"
            }

        try:
            content = response['choices'][0]['message']['content']
            result = json.loads(content)
            return result

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing mismatch analysis: {e}")
            return {
                "mismatch_reason": f"Failed to parse analysis: {str(e)}",
                "predicted_indicators": [],
                "actual_indicators": [],
                "confusing_elements": "Parsing error",
                "correct_interpretation": "Unable to determine",
                "category_boundaries": "Unable to analyze",
                "improvement_suggestions": [],
                "similarity_analysis": "Unable to analyze",
                "error": "Response parsing failed"
            }

    def batch_predict(self,
                     texts: List[str],
                     available_categories: List[str],
                     delay_between_requests: float = 1.0) -> List[Dict[str, Any]]:
        """
        Predict categories for multiple texts with rate limiting

        Args:
            texts (List[str]): List of texts to classify
            available_categories (List[str]): Available categories
            delay_between_requests (float): Delay between requests in seconds

        Returns:
            List[Dict]: List of prediction results
        """
        results = []

        for i, text in enumerate(texts):
            print(f"Processing text {i+1}/{len(texts)}")

            result = self.predict_category(text, available_categories)
            results.append(result)

            # Rate limiting
            if i < len(texts) - 1:
                time.sleep(delay_between_requests)

        return results

    def save_responses(self,
                      responses: List[Dict],
                      filename: str = "deepseek_responses.json") -> bool:
        """
        Save API responses to file

        Args:
            responses (List[Dict]): List of responses to save
            filename (str): Output filename

        Returns:
            bool: Success status
        """
        try:
            with open(filename, 'w') as f:
                json.dump(responses, f, indent=2)

            print(f"Responses saved to {filename}")
            return True

        except Exception as e:
            print(f"Error saving responses: {e}")
            return False

    def load_responses(self,
                      filename: str = "deepseek_responses.json") -> List[Dict]:
        """
        Load API responses from file

        Args:
            filename (str): Input filename

        Returns:
            List[Dict]: Loaded responses
        """
        try:
            with open(filename, 'r') as f:
                responses = json.load(f)

            print(f"Responses loaded from {filename}")
            return responses

        except Exception as e:
            print(f"Error loading responses: {e}")
            return []

def main():
    """Main function to demonstrate DeepSeek integration"""
    print("DeepSeekReasoning class created successfully!")
    print("This class provides category prediction and explanation capabilities.")

    # Example usage (commented out since we need API key):
    """
    # Initialize DeepSeek
    deepseek = DeepSeekReasoning(api_key="your_api_key_here")

    # Predict category
    text = "The president announced new economic policies today..."
    categories = ["POLITICS", "BUSINESS", "WORLD NEWS", "SPORTS"]

    prediction = deepseek.predict_category(text, categories)
    print(prediction)

    # Generate explanation
    explanation = deepseek.generate_explanation(text, prediction['predicted_category'])
    print(explanation)
    """

if __name__ == "__main__":
    main()
