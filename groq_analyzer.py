# groq_analyzer.py - Improved implementation for Groq sentiment analysis

import os
import json
import requests
import time
import logging
import re
import random

class GroqSentimentAnalyzer:
    def __init__(self):
        """Initialize the Groq sentiment analyzer"""
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        
        # Check if we have a Groq API key
        if self.api_key:
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Using Groq's LLM for sentiment analysis
            self.model = "llama3-8b-8192"
            self.using_real_api = True
            logging.info(f"Initialized Groq Sentiment Analyzer with model: {self.model}")
        else:
            self.using_real_api = False
            logging.warning("No GROQ_API_KEY found, using fallback sentiment analysis")
    
    def analyze(self, text):
        """Analyze sentiment using Groq API or fallback method"""
        if self.using_real_api:
            return self._analyze_with_groq_api(text)
        else:
            return self._analyze_with_fallback(text)
    
    def _analyze_with_groq_api(self, text):
        """Analyze sentiment using the actual Groq API"""
        prompt = f"""
        Analyze the sentiment of this tweet and respond with ONLY a JSON object 
        having two fields: "sentiment" (values: "positive", "neutral", or "negative") 
        and "confidence" (a float value between 0.1 and 0.99).

        Tweet: "{text}"
        
        JSON response:
        """
        
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a sentiment analysis expert. Analyze the sentiment of tweets and respond with the specified JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 100
            }
            
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload
            )
            
            response.raise_for_status()
            response_data = response.json()
            result_text = response_data["choices"][0]["message"]["content"].strip()
            
            # Extract the JSON portion from the response
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(result_text)
            
            # Validate the response format
            if "sentiment" not in result or "confidence" not in result:
                raise ValueError("Invalid response format from Groq API")
                
            return result
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error calling Groq API: {str(e)}")
            # Fallback to local method if API call fails
            return self._analyze_with_fallback(text)
        
        except (ValueError, json.JSONDecodeError) as e:
            logging.error(f"Error parsing Groq API response: {str(e)}")
            return self._analyze_with_fallback(text)
    
    def _analyze_with_fallback(self, text):
        """Use a rule-based fallback method for sentiment analysis"""
        # Word lists for sentiment analysis
        positive_words = [
            'amazing', 'awesome', 'excellent', 'good', 'great', 'love', 'fantastic',
            'wonderful', 'brilliant', 'outstanding', 'superb', 'incredible', 'impressive',
            'exceptional', 'perfect', 'happy', 'excited', 'enjoy', 'pleased', 'delighted',
            'thrilled', 'revolutionary', 'game-changing', 'innovative', 'fast', 'faster',
            'best', 'better', 'improved', 'efficient', 'effective', 'seamless', 'smooth'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'hate', 'disappointed', 'issues', 'error', 'bug',
            'problem', 'slow', 'crash', 'fail', 'failure', 'poor', 'useless', 'waste',
            'broken', 'unusable', 'frustrating', 'annoying', 'difficult', 'complicated',
            'expensive', 'overpriced', 'unstable', 'unreliable', 'inconsistent', 'glitch',
            'disappointing', 'worse', 'worst', 'sucks', 'horrible', 'mediocre', 'subpar'
        ]
        
        # Negation words that flip sentiment
        negation_words = ['not', 'no', "don't", "doesn't", "didn't", "wasn't", "aren't", 
                          "isn't", "haven't", "hasn't", "won't", "wouldn't", "couldn't",
                          "shouldn't", "can't", "cannot", "never"]
        
        # Convert text to lowercase for easier matching
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Calculate positive and negative scores
        pos_score = 0
        neg_score = 0
        
        for i, word in enumerate(words):
            # Check for negation context (simple window of 3 words before current word)
            negated = False
            for j in range(max(0, i-3), i):
                if words[j] in negation_words:
                    negated = True
                    break
            
            # Count sentiment words with negation handling
            if word in positive_words:
                if negated:
                    neg_score += 1
                else:
                    pos_score += 1
            
            if word in negative_words:
                if negated:
                    pos_score += 1
                else:
                    neg_score += 1
        
        # Determine sentiment based on scores
        if pos_score > neg_score:
            sentiment = "positive"
            # Base confidence on margin of difference
            base_confidence = 0.6 + min(0.3, (pos_score - neg_score) * 0.1)
        elif neg_score > pos_score:
            sentiment = "negative"
            base_confidence = 0.6 + min(0.3, (neg_score - pos_score) * 0.1)
        else:
            # Check if there were any sentiment words at all
            if pos_score == 0 and neg_score == 0:
                sentiment = "neutral"
                base_confidence = 0.7
            else:
                # If tied but some sentiment words exist, slightly random
                if random.random() < 0.5:
                    sentiment = "neutral"
                    base_confidence = 0.6
                else:
                    # Slightly favor positive sentiment when tied
                    sentiment = "positive" if random. random() < 0.6 else "negative"
                    base_confidence = 0.55
        
        # Add some randomness to confidence for more realistic results
        confidence = min(0.98, max(0.5, base_confidence + random.uniform(-0.05, 0.05)))
        
        return {"sentiment": sentiment, "confidence": round(confidence, 2)}