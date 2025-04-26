# tweet_simulator.py - Enhanced tweet simulator for demo purposes

import random
from datetime import datetime
import time
import re

class TweetSimulator:
    def __init__(self, hashtag="#GrowWithGroqHack"):
        """Initialize the tweet simulator with the target hashtag"""
        # Store hashtag without forcing the # prefix
        self.hashtag = hashtag if hashtag.startswith("#") else f"#{hashtag}"
        
        # Templates for generating more realistic tweets
        self.templates = [
            "Just tried {product} and it's {opinion}! {hashtag}",
            "{opinion} experience with {product} today. {hashtag}",
            "Has anyone else {question} with {product}? {hashtag}",
            "I'm {feeling} about {product}'s new features. {hashtag}",
            "{product} is absolutely {opinion}. {hashtag} #Tech",
            "My {timeframe} using {product}: {opinion}! {hashtag}",
            "Need help with {product}. Getting {issue} when I try to {action}. {hashtag}",
            "{opinion} that {product} just announced their new {feature}! {hashtag}",
            "Comparing {product} with competitors - {opinion} difference in {metric}. {hashtag}",
            "{action} with {product} and the results are {opinion}. {hashtag}"
        ]
        
        # Word banks for generating variations
        self.products = ["Groq", "GroqCloud", "GroqAPI", "the Groq platform", "Groq's LLM API", 
                        "the Groq inference engine", "Groq's transformer models"]
        
        self.positive_opinions = ["amazing", "incredible", "outstanding", "game-changing", "impressive", 
                                "revolutionary", "top-notch", "excellent", "fantastic", "mind-blowing"]
        
        self.neutral_opinions = ["interesting", "decent", "reasonable", "adequate", "acceptable", 
                               "standard", "moderate", "fair", "ordinary", "typical"]
        
        self.negative_opinions = ["disappointing", "frustrating", "underwhelming", "problematic", 
                                "concerning", "subpar", "mediocre", "buggy", "flawed", "unstable"]
        
        self.feelings = ["excited", "optimistic", "unsure", "concerned", "impressed", "disappointed", 
                        "thrilled", "confused", "skeptical", "hopeful"]
        
        self.questions = ["noticed issues", "had trouble", "seen performance gains", "found bugs", 
                        "experienced latency", "gotten good results", "had success"]
        
        self.issues = ["errors", "timeouts", "crashes", "inconsistent results", "authentication issues", 
                      "rate limiting", "unexpected responses", "low performance"]
        
        self.actions = ["implementing", "deploying", "testing", "benchmarking", "integrating", 
                       "optimizing", "debugging", "configuring"]
        
        self.timeframes = ["first week", "two months", "experience", "trial run", "benchmark", 
                          "integration project", "performance test"]
        
        self.features = ["pricing model", "latency improvements", "API enhancements", "model upgrades", 
                        "inference optimization", "developer tools", "documentation"]
        
        self.metrics = ["speed", "cost", "accuracy", "reliability", "ease of use", "documentation", 
                       "customer support", "integration options"]
        
        # User details for simulating different authors
        self.user_prefixes = ["dev", "ai", "ml", "tech", "data", "cloud", "code", "neural", "prompt", "llm"]
        self.user_suffixes = ["guru", "ninja", "expert", "hacker", "wizard", "engineer", "scientist", "enthusiast", "developer", "pro"]
        
        # Last generated time to ensure timestamps move forward
        self.last_time = datetime.now().timestamp()
    
    def _get_sentiment_type(self):
        """Randomly choose a sentiment type with weights"""
        # 50% positive, 30% neutral, 20% negative
        r = random.random()
        if r < 0.5:
            return "positive"
        elif r < 0.8:
            return "neutral"
        else:
            return "negative"
    
    def _get_opinion_by_sentiment(self, sentiment_type):
        """Get an opinion word based on the sentiment type"""
        if sentiment_type == "positive":
            return random.choice(self.positive_opinions)
        elif sentiment_type == "neutral":
            return random.choice(self.neutral_opinions)
        else:
            return random.choice(self.negative_opinions)
    
    def _generate_username(self):
        """Generate a random Twitter-like username"""
        prefix = random.choice(self.user_prefixes)
        suffix = random.choice(self.user_suffixes)
        number = random.randint(10, 9999)
        return f"@{prefix}_{suffix}{number}"
    
    def _generate_timestamp(self):
        """Generate a timestamp that's always moving forward"""
        # Ensure each tweet is 1-5 seconds newer than the previous one
        current_time = self.last_time + random.uniform(1, 5)
        self.last_time = current_time
        timestamp = datetime.fromtimestamp(current_time)
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    def _generate_tweet_text(self, sentiment_type):
        """Generate tweet text based on a template and sentiment type"""
        template = random.choice(self.templates)
        product = random.choice(self.products)
        opinion = self._get_opinion_by_sentiment(sentiment_type)
        feeling = random.choice(self.feelings)
        question = random.choice(self.questions)
        issue = random.choice(self.issues)
        action = random.choice(self.actions)
        timeframe = random.choice(self.timeframes)
        feature = random.choice(self.features)
        metric = random.choice(self.metrics)
        
        # Format the tweet text with the template
        tweet_text = template.format(
            product=product,
            opinion=opinion,
            hashtag=self.hashtag,
            feeling=feeling,
            question=question,
            issue=issue,
            action=action,
            timeframe=timeframe,
            feature=feature,
            metric=metric
        )
        
        return tweet_text
    
    def get_random_tweet(self, sentiment_type=None):
        """Generate a random tweet with timestamp and user"""
        # Determine sentiment if not specified
        if sentiment_type is None:
            sentiment_type = self._get_sentiment_type()
        
        # Generate tweet components
        username = self._generate_username()
        text = self._generate_tweet_text(sentiment_type)
        timestamp = self._generate_timestamp()
        
        # Return tweet in standardized format
        return {
            "id": f"sim_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            "username": username,
            "text": text,
            "timestamp": timestamp
        }
    
    def get_stream_sample(self):
        """Match the interface of TwitterStream"""
        return self.get_random_tweet()