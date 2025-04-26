# Live Tweet Sentiment Meter

Real-time sentiment analysis dashboard for tracking and visualizing tweet sentiment.

![Tweet Sentiment Dashboard](https://i.imgur.com/5X8wK9H.png)

## 📋 Overview

This project is a real-time Twitter sentiment analysis dashboard that processes tweets with a specific hashtag (e.g., `#GrowWithGroqHack`), analyzes their sentiment using Groq-accelerated transformer models, and visualizes the results in a Streamlit dashboard.

### ✨ Features

- **Real-time Tweet Processing**: Simulates or fetches tweets with configurable hashtags
- **Groq-Powered Sentiment Analysis**: Uses transformer models for accurate sentiment classification
- **Interactive Dashboard**: Visualizes sentiment distribution, trending words, and latest tweets
- **Alert System**: Sends Slack notifications when negative sentiment spikes are detected
- **Word Cloud Visualization**: Shows trending words from processed tweets
- **Color-Coded Tweets**: Visual indicators for positive, neutral, and negative sentiment

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Groq API access (optional for real deployment)
- Slack Webhook URL (for alert functionality)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/tweet-sentiment-meter.git
   cd tweet-sentiment-meter
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## 🔧 Configuration

- **Slack Webhook URL**: Add your own webhook URL in the sidebar to enable real Slack alerts
- **Tweet Source**: For production, replace the `TweetSimulator` class with real Twitter API integration
- **Groq API**: Replace the mock `GroqSentimentAnalyzer` with actual Groq API integration

## 💻 Usage

1. Open the app in your browser (typically http://localhost:8501)
2. Click "Start Stream" to begin processing tweets
3. Monitor sentiment distribution and trending words in real-time
4. Receive Slack alerts when negative sentiment spikes are detected

## 🧠 How It Works

1. **Tweet Collection**: Simulates or fetches tweets with the target hashtag
2. **Sentiment Analysis**: Processes each tweet using Groq-accelerated transformer models
3. **Visualization**: Updates dashboard in real-time with sentiment trends and statistics
4. **Alert Monitoring**: Tracks negative sentiment patterns and sends alerts when thresholds are reached

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Built using Streamlit for dashboard visualization
- Powered by Groq for fast transformer model inference
- Slack for alerts integration