# app.py - Main Streamlit application with improved Twitter simulation and Groq integration

import streamlit as st
import pandas as pd
import plotly.express as px
import time
import requests
from datetime import datetime
import threading
from collections import deque
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Import Twitter stream and Groq analyzer
from twitter_stream import TwitterStream
from tweet_simulator import TweetSimulator
from groq_analyzer import GroqSentimentAnalyzer

# Slack integration for alerts
class SlackAlerter:
    def __init__(self, webhook_url=None):
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        
    def send_alert(self, message):
        """Send alert to Slack channel"""
        if not self.webhook_url:
            st.warning(f"SLACK ALERT (NO WEBHOOK CONFIGURED): {message}")
            return {"status": "no_webhook", "message": "No webhook URL configured"}
        
        payload = {"text": message}
        
        try:
            response = requests.post(self.webhook_url, json=payload)
            if response.status_code == 200:
                st.success("Alert sent to Slack successfully")
            else:
                st.error(f"Failed to send alert to Slack: {response.status_code}")
            return {"status": "sent", "response": response.text}
        except Exception as e:
            st.error(f"Error sending alert to Slack: {str(e)}")
            return {"status": "error", "message": str(e)}

# Initialize session state variables if they don't exist
def initialize_session_state():
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
    
    # Always try to use real Twitter API first, with graceful fallback
    if 'tweet_source' not in st.session_state or 'using_real_twitter' not in st.session_state:
        try:
            # Check for Twitter API credentials
            if not os.environ.get("TWITTER_BEARER_TOKEN"):
                st.session_state.tweet_source = TweetSimulator(hashtag=st.session_state.hashtag)
                st.session_state.using_real_twitter = False
            else:
                st.session_state.tweet_source = TwitterStream(hashtag=st.session_state.hashtag)
                st.session_state.using_real_twitter = True
        except Exception as e:
            st.error(f"Error initializing Twitter API: {str(e)}. Falling back to simulation.")
            st.session_state.tweet_source = TweetSimulator(hashtag=st.session_state.hashtag)
            st.session_state.using_real_twitter = False
    
    # Initialize Groq Sentiment Analyzer
    if 'analyzer' not in st.session_state:
        try:
            st.session_state.analyzer = GroqSentimentAnalyzer()
            st.session_state.using_real_groq = st.session_state.analyzer.using_real_api
        except Exception as e:
            st.error(f"Error initializing Groq API: {str(e)}. Using fallback sentiment analysis.")
            st.session_state.analyzer = GroqSentimentAnalyzer()  # This will use fallback
            st.session_state.using_real_groq = False
    
    # Initialize Slack Alerter
    if 'slack_alerter' not in st.session_state:
        st.session_state.slack_alerter = SlackAlerter()

# Function to check if we need to send a slack alert
def check_for_alert_condition():
    # Get negative tweets from the last 10 seconds
    current_time = time.time()
    recent_negative_count = sum(1 for t in st.session_state.recent_negative 
                               if current_time - t <= 10)
    
    if recent_negative_count >= 3:
        message = f"⚠️ Negative sentiment spike detected! Check tweets for {st.session_state.hashtag}."
        st.session_state.slack_alerter.send_alert(message)
        # Clear the queue to avoid repeated alerts
        st.session_state.recent_negative.clear()

# Function to update word counts from tweet text
def update_word_counts(text):
    # Clean and tokenize the text
    text = re.sub(r'http\S+', '', text.lower())
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    
    # Filter out common stop words (simplified version)
    stop_words = {'the', 'a', 'an', 'and', 'is', 'in', 'it', 'to', 'with', 'for', 'of', 'on', 'at', 
                 'by', 'from', 'up', 'about', 'into', 'over', 'after', 'this', 'that', 'these', 
                 'those', 'just', 'but', 'very', 'not', 'has', 'have', 'had', 'was', 'were', 'been'}
    
    # Also filter out the hashtag itself from the word counts
    hashtag_text = st.session_state.hashtag.lower().replace('#', '')
    words = [word for word in words if word not in stop_words and len(word) > 2 and word != hashtag_text]
    
    # Update word counts
    for word in words:
        if word in st.session_state.word_counts:
            st.session_state.word_counts[word] += 1
        else:
            st.session_state.word_counts[word] = 1

# Function to process a single tweet
def process_tweet():
    if not st.session_state.stream_active:
        return
    
    try:
        # Get a tweet - either from Twitter API or simulator depending on state
        tweet = st.session_state.tweet_source.get_stream_sample()
        
        if not tweet:
            # No tweet available - either retry or wait
            return
        
        # Analyze sentiment using our Groq-powered analyzer
        result = st.session_state.analyzer.analyze(tweet["text"])
        sentiment = result["sentiment"]
        confidence = result["confidence"]
        
        # Update sentiment counts
        st.session_state.sentiment_counts[sentiment] += 1
        
        # Track negative tweets for alert condition
        if sentiment == "negative":
            st.session_state.recent_negative.append(time.time())
        
        # Check if we need to send an alert
        check_for_alert_condition()
        
        # Update word counts for word cloud
        update_word_counts(tweet["text"])
        
        # Add processed tweet to the list
        tweet_with_sentiment = {
            **tweet,
            "sentiment": sentiment,
            "confidence": confidence
        }
        st.session_state.tweets.insert(0, tweet_with_sentiment)
        
        # Keep only the 100 most recent tweets
        if len(st.session_state.tweets) > 100:
            st.session_state.tweets = st.session_state.tweets[:100]
    
    except Exception as e:
        # Don't crash, just log the error
        st.error(f"Error processing tweet: {str(e)}")

# Function to continuously process tweets
def tweet_stream():
    while st.session_state.get("stream_active", False):
        process_tweet()
        time.sleep(st.session_state.tweet_gen_rate)  # Process tweets at the configured rate

# Main function to render the Streamlit app
def main():
    st.set_page_config(
        page_title="Live Tweet Sentiment Meter",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Title and description
    st.title("📊 Live Tweet Sentiment Meter")
    st.markdown(f"Real-time sentiment analysis of tweets with **{st.session_state.hashtag}** hashtag")
    
    # Data source indicators
    col_status1, col_status2 = st.columns(2)
    
    with col_status1:
        if st.session_state.using_real_twitter:
            st.success("✅ Using real Twitter API data")
        else:
            st.warning("⚠️ Using simulated tweet data (Twitter API not configured)")
    
    with col_status2:
        if st.session_state.using_real_groq:
            st.success("✅ Using Groq API for sentiment analysis")
        else:
            st.warning("⚠️ Using fallback sentiment analysis (Groq API not configured)")
    
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
            try:
                if st.session_state.using_real_twitter:
                    st.session_state.tweet_source = TwitterStream(hashtag=new_hashtag)
                else:
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
            except Exception as e:
                st.error(f"Error updating tweet source: {str(e)}")
        
        # Tweet generation rate slider (only for simulated mode)
        if not st.session_state.using_real_twitter:
            new_rate = st.slider(
                "Tweet generation rate (seconds)",
                min_value=0.5,
                max_value=5.0,
                value=st.session_state.tweet_gen_rate,
                step=0.5
            )
            
            if new_rate != st.session_state.tweet_gen_rate:
                st.session_state.tweet_gen_rate = new_rate
                st.success(f"Tweet generation rate updated to {new_rate} seconds")
        
        # Webhook URL input
        webhook_url = st.text_input(
            "Slack Webhook URL",
            value=st.session_state.slack_alerter.webhook_url or "",
            type="password"
        )
        
        if webhook_url != (st.session_state.slack_alerter.webhook_url or ""):
            st.session_state.slack_alerter = SlackAlerter(webhook_url)
            st.success("Slack webhook URL updated")
        
        st.header("Controls")
        
        # Start/Stop button
        if not st.session_state.stream_active:
            if st.button("▶️ Start Stream"):
                st.session_state.stream_active = True
                # Start the stream in a background thread
                thread = threading.Thread(target=tweet_stream)
                thread.daemon = True
                thread.start()
                st.success(f"Stream started for {st.session_state.hashtag}")
        else:
            if st.button("⏹️ Stop Stream"):
                st.session_state.stream_active = False
                st.info("Stream stopped")
        
        # Reset button
        if st.button("🔄 Reset Data"):
            st.session_state.tweets = []
            st.session_state.sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            st.session_state.recent_negative.clear()
            st.session_state.word_counts = {}
            st.success("Data has been reset")
        
        # Test buttons for generating specific sentiment tweets (only in simulation mode)
        if not st.session_state.using_real_twitter:
            st.subheader("Simulation Controls")
            test_cols = st.columns(3)
            
            with test_cols[0]:
                if st.button("➕ Add Positive"):
                    # Generate and process a positive tweet
                    simulator = TweetSimulator(st.session_state.hashtag)
                    tweet = simulator.get_random_tweet(sentiment_type="positive")
                    
                    # Add tweet to state and update counts
                    result = st.session_state.analyzer.analyze(tweet["text"])
                    tweet_with_sentiment = {**tweet, **result}
                    st.session_state.tweets.insert(0, tweet_with_sentiment)
                    st.session_state.sentiment_counts[result["sentiment"]] += 1
                    update_word_counts(tweet["text"])
            
            with test_cols[1]:
                if st.button("⚪ Add Neutral"):
                    # Generate and process a neutral tweet
                    simulator = TweetSimulator(st.session_state.hashtag)
                    tweet = simulator.get_random_tweet(sentiment_type="neutral")
                    
                    # Add tweet to state and update counts
                    result = st.session_state.analyzer.analyze(tweet["text"])
                    tweet_with_sentiment = {**tweet, **result}
                    st.session_state.tweets.insert(0, tweet_with_sentiment)
                    st.session_state.sentiment_counts[result["sentiment"]] += 1
                    update_word_counts(tweet["text"])
            
            with test_cols[2]:
                if st.button("➖ Add Negative"):
                    # Generate and process a negative tweet
                    simulator = TweetSimulator(st.session_state.hashtag)
                    tweet = simulator.get_random_tweet(sentiment_type="negative")
                    
                    # Add tweet to state and update counts
                    result = st.session_state.analyzer.analyze(tweet["text"])
                    tweet_with_sentiment = {**tweet, **result}
                    st.session_state.tweets.insert(0, tweet_with_sentiment)
                    st.session_state.sentiment_counts[result["sentiment"]] += 1
                    
                    # Also track for alert condition
                    if result["sentiment"] == "negative":
                        st.session_state.recent_negative.append(time.time())
                        # Check if need to send alert
                        check_for_alert_condition()
                    
                    update_word_counts(tweet["text"])
        
        # Display alert rules
        st.subheader("Alert Rules")
        st.info("⚠️ Alert will be sent to Slack if 3 or more negative tweets are detected within 10 seconds")
        
        # Display credits
        st.markdown("---")
        st.markdown("### About")
        st.markdown("Live Tweet Sentiment Meter powered by:")
        st.markdown("- 🚀 Groq for sentiment analysis")
        st.markdown("- 🐦 Twitter API for real-time data")
        st.markdown("- 📊 Streamlit for visualization")
        st.markdown("- 📱 Slack for alerts")

    # Main content area - split into columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
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
        
        # Add CSS for tweet styling once at the top level
        st.markdown("""
        <style>
        .tweet-container-positive {
            background-color: #e6f4ea;
            border-left: 4px solid #4CAF50;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .tweet-container-neutral {
            background-color: #fff9e6;
            border-left: 4px solid #FFC107;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .tweet-container-negative {
            background-color: #fdedeb;
            border-left: 4px solid #F44336;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .tweet-header {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .tweet-timestamp {
            color: #657786;
            font-size: 0.8em;
        }
        .tweet-sentiment-positive {
            font-weight: bold;
            color: #4CAF50;
            text-transform: uppercase;
            font-size: 0.8em;
        }
        .tweet-sentiment-neutral {
            font-weight: bold;
            color: #FFC107;
            text-transform: uppercase;
            font-size: 0.8em;
        }
        .tweet-sentiment-negative {
            font-weight: bold;
            color: #F44336;
            text-transform: uppercase;
            font-size: 0.8em;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.session_state.tweets:
            # Use a container for all tweets
            tweets_container = st.container()
            
            # Display tweets in the container
            for i, tweet in enumerate(st.session_state.tweets[:10]):  # Show only 10 most recent
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