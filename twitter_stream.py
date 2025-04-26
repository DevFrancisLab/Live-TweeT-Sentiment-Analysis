# twitter_stream.py - Twitter API v2 integration with proper implementation

import tweepy
import os
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TwitterStream:
    def __init__(self, hashtag="#GrowWithGroqHack"):
        """Initialize the Twitter stream with API credentials and target hashtag"""
        
        # Ensure hashtag has # prefix
        self.hashtag = hashtag if hashtag.startswith("#") else f"#{hashtag}"
        
        # Get credentials from environment variables for security
        self.bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
        
        # Validate credentials are available
        if not self.bearer_token:
            raise ValueError("Missing required Twitter API credential: TWITTER_BEARER_TOKEN")
        
        # Initialize Twitter client
        self.client = tweepy.Client(bearer_token=self.bearer_token)
        logging.info(f"TwitterStream initialized for hashtag: {self.hashtag}")
        
        # Search parameters
        self.query = f"{self.hashtag} -is:retweet"  # Exclude retweets
    
    def get_recent_tweets(self, count=10):
        """Fetch recent tweets matching the hashtag using Twitter API v2"""
        try:
            # Define tweet fields we want to retrieve
            tweet_fields = ['created_at', 'public_metrics', 'text']
            user_fields = ['username', 'name', 'profile_image_url']
            expansions = ['author_id']
            
            # Search for tweets
            response = self.client.search_recent_tweets(
                query=self.query,
                max_results=count,
                tweet_fields=tweet_fields,
                user_fields=user_fields,
                expansions=expansions
            )
            
            if not response.data:
                logging.warning(f"No tweets found matching query: {self.query}")
                return []
            
            # Process tweets into standardized format
            processed_tweets = []
            
            # Create a dict of users for quick lookup
            users = {user.id: user for user in response.includes['users']} if 'users' in response.includes else {}
            
            for tweet in response.data:
                # Get user info
                user = users.get(tweet.author_id)
                username = f"@{user.username}" if user else "Unknown User"
                
                # Format timestamp
                timestamp = tweet.created_at.strftime("%Y-%m-%d %H:%M:%S") if tweet.created_at else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                processed_tweet = {
                    "id": tweet.id,
                    "username": username,
                    "text": tweet.text,
                    "timestamp": timestamp
                }
                processed_tweets.append(processed_tweet)
            
            logging.info(f"Fetched {len(processed_tweets)} tweets with hashtag {self.hashtag}")
            return processed_tweets
            
        except Exception as e:
            logging.error(f"Error fetching tweets: {str(e)}")
            raise  # Re-raise to let main app handle the error properly
    
    def get_stream_sample(self):
        """Get a single tweet from the stream - can be called repeatedly for streaming effect"""
        tweets = self.get_recent_tweets(count=5)
        
        if not tweets:
            logging.warning(f"No tweets found for {self.hashtag}, returning None")
            return None
        
        # Return most recent tweet first
        return tweets[0] if tweets else None