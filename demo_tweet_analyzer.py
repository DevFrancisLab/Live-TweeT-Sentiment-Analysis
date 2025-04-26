# demo_tweet_analyzer.py - Simplified version for demo purposes

import streamlit as st
import pandas as pd
import plotly.express as px
import time
import random
import re
import threading
from collections import deque
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Tweet simulator class - simplified version
class TweetSimulator:
    def __init__(self, hashtag="#GrowWithGroqHack"):
        self.hashtag = hashtag if hashtag.startswith("#") else f"#{hashtag}"
        
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
        
        # Word banks based on the hashtag topic (will extract from the hashtag)
        self.hashtag_topic = self.hashtag.replace("#", "").lower()
        self.products = self._generate_product_list()
        
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
        
        self.user_prefixes = ["dev", "ai", "ml", "tech", "data", "cloud", "code", "neural", "prompt", "llm"]
        self.user_suffixes = ["guru", "ninja", "expert", "hacker", "wizard", "engineer", "scientist", "enthusiast", "developer", "pro"]
        
        self.last_time = datetime.now().timestamp()
    
    def _generate_product_list(self):
        """Generate product names based on the hashtag topic"""
        topic = self.hashtag_topic
        if not topic or len(topic) < 3:
            topic = "tech"
            
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', topic)
        topic_base = words[0] if words else "tech"
        
        return [
            f"{topic_base.capitalize()}", 
            f"{topic_base.capitalize()} Cloud", 
            f"{topic_base.capitalize()} Platform",
            f"The {topic_base.capitalize()} API", 
            f"{topic_base.capitalize()} Pro", 
            f"{topic_base.capitalize()} Enterprise",
            f"{topic_base.capitalize()} Suite"
        ]
    
    def _get_sentiment_type(self):
        """Randomly choose a sentiment type with weights"""
        r = random.random()
        if r < 0.5:  # 50% positive
            return "positive"
        elif r < 0.8:  # 30% neutral
            return "neutral"
        else:  # 20% negative
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
        if sentiment_type is None:
            sentiment_type = self._get_sentiment_type()
        
        username = self._generate_username()
        text = self._generate_tweet_text(sentiment_type)
        timestamp = self._generate_timestamp()
        
        return {
            "id": f"sim_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            "username": username,
            "text": text,
            "timestamp": timestamp,
            "sentiment": sentiment_type,
            "confidence": round(random.uniform(0.7, 0.98), 2)
        }
    
    def get_stream_sample(self):
        """Get a single tweet from the stream"""
        return self.get_random_tweet()

# Simplified sentiment analyzer - for demo, just returns sentiment directly from simulator
class SimpleAnalyzer:
    def analyze(self, text):
        """Simple random sentiment analysis for demo purposes"""
        r = random.random()
        if r < 0.5:
            sentiment = "positive"
        elif r < 0.8:
            sentiment = "neutral"
        else:
            sentiment = "negative"
        
        confidence = round(random.uniform(0.6, 0.95), 2)
        
        return {"sentiment": sentiment, "confidence": confidence}

# Function to update word counts from tweet text
def update_word_counts(text, word_counts):
    """Update word counts from tweet text for word cloud"""
    text = re.sub(r'http\S+', '', text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    
    stop_words = {'the', 'a', 'an', 'and', 'is', 'in', 'it', 'to', 'with', 'for', 'of', 'on', 'at', 
                 'by', 'from', 'up', 'about', 'into', 'over', 'after', 'this', 'that', 'these', 
                 'those', 'just', 'but', 'very', 'not', 'has', 'have', 'had', 'was', 'were', 'been'}
    
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    return word_counts

# Function to process a single tweet
def process_tweet(stream_active, tweet_source, analyzer, tweets, sentiment_counts, recent_negative, word_counts):
    """Process a single tweet from the stream"""
    if not stream_active:
        return None
    
    try:
        # Get a tweet
        tweet = tweet_source.get_stream_sample()
        
        if not tweet:
            return None
            
        # For demo, sentiment is already in the generated tweet
        sentiment = tweet["sentiment"]
        confidence = tweet["confidence"]
        
        # Update sentiment counts
        sentiment_counts[sentiment] += 1
        
        # Track negative tweets for alert condition
        if sentiment == "negative":
            recent_negative.append(time.time())
        
        # Update word counts for word cloud
        update_word_counts(tweet["text"], word_counts)
        
        # Return the processed tweet
        return tweet
    
    except Exception as e:
        st.error(f"Error processing tweet: {str(e)}")
        return None

# Main function
def main():
    st.set_page_config(
        page_title="Live Tweet Sentiment Meter",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state
    if 'tweets' not in st.session_state:
        st.session_state.tweets = []
    
    if 'sentiment_counts' not in st.session_state:
        st.session_state.sentiment_counts = {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }
    
    if 'recent_negative' not in st.session_state:
        st.session_state.recent_negative = deque(maxlen=20)
    
    if 'word_counts' not in st.session_state:
        st.session_state.word_counts = {}
    
    if 'stream_active' not in st.session_state:
        st.session_state.stream_active = False
    
    if 'hashtag' not in st.session_state:
        st.session_state.hashtag = "#GrowWithGroqHack"
    
    if 'tweet_gen_rate' not in st.session_state:
        st.session_state.tweet_gen_rate = 1.5  # Generate a tweet every 1.5 seconds by default
    
    if 'tweet_source' not in st.session_state:
        st.session_state.tweet_source = TweetSimulator(hashtag=st.session_state.hashtag)
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SimpleAnalyzer()
    
    # Title and description
    st.title("📊 Live Tweet Sentiment Meter")
    st.markdown(f"Real-time sentiment analysis of tweets with **{st.session_state.hashtag}** hashtag")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Hashtag input
        new_hashtag = st.text_input(
            "Hashtag to track",
            value=st.session_state.hashtag
        )
        
        # Ensure hashtag has # prefix
        if not new_hashtag.startswith("#") and new_hashtag.strip() != "":
            new_hashtag = f"#{new_hashtag}"
        
        if new_hashtag != st.session_state.hashtag and new_hashtag.strip() != "":
            # Update session state
            st.session_state.hashtag = new_hashtag
            
            # Reset tweet source based on the new hashtag
            st.session_state.tweet_source = TweetSimulator(hashtag=new_hashtag)
            
            # Clear existing data when changing hashtag
            st.session_state.tweets = []
            st.session_state.sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            st.session_state.recent_negative.clear()
            st.session_state.word_counts = {}
            
            st.success(f"Now tracking {new_hashtag} tweets")
            
            # Force stream to stop so user can restart it
            if st.session_state.stream_active:
                st.session_state.stream_active = False
                st.warning("Stream stopped due to hashtag change. Please restart the stream.")
        
        # Tweet generation rate slider for demo
        new_rate = st.slider(
            "Tweet generation rate (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.tweet_gen_rate,
            step=0.5
        )
        
        if new_rate != st.session_state.tweet_gen_rate:
            st.session_state.tweet_gen_rate = new_rate
        
        st.header("Controls")
        
        # Start/Stop button
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.stream_active:
                if st.button("▶️ Start Stream"):
                    st.session_state.stream_active = True
                    st.success(f"Stream started for {st.session_state.hashtag}")
            else:
                if st.button("⏹️ Stop Stream"):
                    st.session_state.stream_active = False
                    st.info("Stream stopped")
        
        with col2:
            # Reset button
            if st.button("🔄 Reset Data"):
                st.session_state.tweets = []
                st.session_state.sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
                st.session_state.recent_negative.clear()
                st.session_state.word_counts = {}
                st.success("Data has been reset")
        
        # Demo tweet generators
        st.subheader("Demo Controls")
        
        test_cols = st.columns(3)
        
        with test_cols[0]:
            if st.button("➕ Check Positive"):
                tweet = st.session_state.tweet_source.get_random_tweet(sentiment_type="positive")
                st.session_state.tweets.insert(0, tweet)
                st.session_state.sentiment_counts["positive"] += 1
                update_word_counts(tweet["text"], st.session_state.word_counts)
        
        with test_cols[1]:
            if st.button("⚪ Check Neutral"):
                tweet = st.session_state.tweet_source.get_random_tweet(sentiment_type="neutral")
                st.session_state.tweets.insert(0, tweet)
                st.session_state.sentiment_counts["neutral"] += 1
                update_word_counts(tweet["text"], st.session_state.word_counts)
        
        with test_cols[2]:
            if st.button("➖ Check Negative"):
                tweet = st.session_state.tweet_source.get_random_tweet(sentiment_type="negative")
                st.session_state.tweets.insert(0, tweet)
                st.session_state.sentiment_counts["negative"] += 1
                st.session_state.recent_negative.append(time.time())
                update_word_counts(tweet["text"], st.session_state.word_counts)
    
    # Main content area - split into columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Process a tweet if stream is active
        if st.session_state.stream_active:
            new_tweet = process_tweet(
                st.session_state.stream_active,
                st.session_state.tweet_source,
                st.session_state.analyzer,
                st.session_state.tweets,
                st.session_state.sentiment_counts,
                st.session_state.recent_negative,
                st.session_state.word_counts
            )
            
            if new_tweet:
                st.session_state.tweets.insert(0, new_tweet)
                if len(st.session_state.tweets) > 100:
                    st.session_state.tweets = st.session_state.tweets[:100]
        
        # Overall mood emoji
        total_tweets = sum(st.session_state.sentiment_counts.values())
        if total_tweets > 0:
            pos_percent = st.session_state.sentiment_counts["positive"] / total_tweets
            neg_percent = st.session_state.sentiment_counts["negative"] / total_tweets
            
            mood_emoji = "😐"  # Default neutral
            if pos_percent > 0.5:
                mood_emoji = "😊"
            elif neg_percent > 0.4:
                mood_emoji = "😠"
            
            overall_mood = f"Overall Mood: {mood_emoji}"
            st.subheader(overall_mood)
        else:
            st.subheader("Overall Mood: Waiting for data...")
        
        # Sentiment distribution chart
        st.subheader("Sentiment Distribution")
        sentiment_df = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [
                st.session_state.sentiment_counts["positive"],
                st.session_state.sentiment_counts["neutral"],
                st.session_state.sentiment_counts["negative"]
            ]
        })
        
        colors = {'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}
        
        # Create pie chart
        if total_tweets > 0:
            fig = px.pie(
                sentiment_df, 
                values='Count', 
                names='Sentiment',
                color='Sentiment',
                color_discrete_map=colors,
                hole=0.4
            )
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Start the stream to see sentiment distribution")
        
        # Word Cloud visualization
        st.subheader("Trending Words")
        if st.session_state.word_counts:
            # Generate word cloud
            wordcloud = WordCloud(
                width=600, 
                height=300, 
                background_color='white',
                colormap='viridis',
                max_words=50
            ).generate_from_frequencies(st.session_state.word_counts)
            
            # Display word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("Start the stream to see trending words")
    
    with col2:
        # Latest tweets with sentiment labels
        st.subheader("Latest Tweets")
        
        # Add CSS for tweet styling with improved color contrast
        st.markdown("""
        <style>
        .tweet-container-positive {
            background-color: #e6f4ea;
            border-left: 4px solid #4CAF50;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
            color: #333333; /* Darker text color for better readability */
        }
        .tweet-container-neutral {
            background-color: #fff9e6;
            border-left: 4px solid #FFC107;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
            color: #333333; /* Darker text color for better readability */
        }
        .tweet-container-negative {
            background-color: #fdedeb;
            border-left: 4px solid #F44336;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
            color: #333333; /* Darker text color for better readability */
        }
        .tweet-header {
            font-weight: bold;
            margin-bottom: 5px;
            color: #000000; /* Even darker for usernames */
        }
        .tweet-timestamp {
            color: #505050; /* Darker gray for better contrast */
            font-size: 0.8em;
        }
        .tweet-sentiment-positive {
            font-weight: bold;
            color: #2E7D32; /* Darker green for better contrast */
            text-transform: uppercase;
            font-size: 0.8em;
        }
        .tweet-sentiment-neutral {
            font-weight: bold;
            color: #B86E00; /* Darker yellow/amber for better contrast */
            text-transform: uppercase;
            font-size: 0.8em;
        }
        .tweet-sentiment-negative {
            font-weight: bold;
            color: #C62828; /* Darker red for better contrast */
            text-transform: uppercase;
            font-size: 0.8em;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.session_state.tweets:
            # Use a container for all tweets
            tweets_container = st.container()
            
            # Display tweets in the container
            for tweet in st.session_state.tweets[:10]:  # Show only 10 most recent
                sentiment = tweet["sentiment"]
                container_class = f"tweet-container-{sentiment}"
                sentiment_class = f"tweet-sentiment-{sentiment}"
                
                # Format confidence to 2 decimal places
                confidence = round(float(tweet["confidence"]), 2)
                
                # Display tweet content with styling
                tweet_html = f"""
                <div class="{container_class}">
                    <div class="tweet-header">{tweet["username"]}</div>
                    <div>{tweet["text"]}</div>
                    <div class="tweet-timestamp">{tweet["timestamp"]} · 
                    <span class="{sentiment_class}">{sentiment.upper()} ({confidence:.2f})</span></div>
                </div>
                """
                
                tweets_container.markdown(tweet_html, unsafe_allow_html=True)
        else:
            st.info("Start the stream to see tweets")

if __name__ == "__main__":
    main()