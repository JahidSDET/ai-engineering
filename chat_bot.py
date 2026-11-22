import os

import dotenv
from openai import OpenAI

dotenv.load_dotenv()


class ChatBot:
    def __init__(self):
        api_key = os.getenv("my-groq-api-key")
        if not api_key:
            raise RuntimeError("Missing my-groq-api-key in .env")

        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Give short and concise answers.",
            }
        ]
        self.total_tokens = 0
        self.llm = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def _trim_messages(self):
        system_message = self.messages[0]
        recent_messages = self.messages[1:][-5:]
        self.messages = [system_message, *recent_messages]

    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        self._trim_messages()

        stream = self.llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=self.messages,
            stream=True,
            stream_options={"include_usage": True},
        )

        print("Bot: ", end="", flush=True)
        response_parts = []
        prompt_tokens = None
        completion_tokens = None
        request_total_tokens = None

        for chunk in stream:
            if chunk.choices:
                text = chunk.choices[0].delta.content
                if text:
                    print(text, end="", flush=True)
                    response_parts.append(text)

            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
                request_total_tokens = chunk.usage.total_tokens

        print()
        assistant_message = "".join(response_parts)
        self.messages.append(
            {"role": "assistant", "content": assistant_message}
        )
        self._trim_messages()

        if request_total_tokens is not None:
            self.total_tokens += request_total_tokens
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Completion tokens: {completion_tokens}")
            print(f"Tokens this request: {request_total_tokens}")
        else:
            print("Token usage was not included in the response.")

        return assistant_message

    def chat_loop(self):
        print("Welcome to the Chat Bot! Type 'exit' to quit.")

        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Exiting the chat bot. Goodbye!")
                break

            self.chat(user_input)
            print(f"Total tokens used: {self.total_tokens}")


def main():
    bot = ChatBot()
    bot.chat_loop()


if __name__ == "__main__":
    main()