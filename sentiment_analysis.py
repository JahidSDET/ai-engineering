import os

import dotenv
from openai import OpenAI

# Load values from the .env file into the process environment.
dotenv.load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        api_key = os.getenv("my-groq-api-key")
        if not api_key:
            raise RuntimeError("Missing my-groq-api-key in .env")

        self.llm = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def analyze_sentiment(self, text):
        response = self.llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the sentiment as positive, negative, or neutral. "
                        "Reply with only the label."
                    ),
                },
                # These examples show the model the expected text-to-label pattern.
                {"role": "user", "content": "I love how helpful the team was."},
                {"role": "assistant", "content": "positive"},
                {"role": "user", "content": "The package arrived broken."},
                {"role": "assistant", "content": "negative"},
                {"role": "user", "content": "The package arrived on Tuesday."},
                {"role": "assistant", "content": "neutral"},
                # This is the text the user wants classified.
                {
                    "role": "user",
                    "content": f"Analyze the sentiment of this text:\n{text}",
                },
            ],
        )

        return response.choices[0].message.content.strip()

    def main(self):
        while True:
            user_input = input(
                "Enter text to analyze sentiment (or 'exit' to quit): "
            )

            if user_input.lower() == "exit":
                break

            sentiment = self.analyze_sentiment(user_input)
            print(f"Sentiment: {sentiment}\n")


# Start the interactive prompt only when this file is run directly.
if __name__ == "__main__":
    SentimentAnalyzer().main()