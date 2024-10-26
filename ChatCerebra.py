from langchain_cerebras import ChatCerebras
from livekit.agents.llm import LLM, ChatContext

class CerebrasLLM(LLM):
    def __init__(self, model: str, api_key: str):
        self.llm = ChatCerebras(
            model=model,
            api_key=api_key
        )

    async def chat(self, chat_ctx: ChatContext):
        """
        Accepts a ChatContext and returns an async generator that yields the assistant's responses.
        """
        # Convert the chat context into the format expected by ChatCerebras
        messages = []
        for msg in chat_ctx.messages:
            content = msg.content
            print("Initial content: ", content)
            if isinstance(content, list):
                # Concatenate list elements into a single string
                content = ''.join([c if isinstance(c, str) else '' for c in content])
                print("Concatenated content: ", content)

            messages.append({
                'role': msg.role,
                'content': content
            })

        # Since ChatCerebras may not support async methods, we'll use an executor
        import asyncio
        loop = asyncio.get_event_loop()
        print("before we call the function, we're going to print the message: ", messages)
        # Function to call the LLM synchronously
        def generate_response():
            response = self.llm(messages)
            return response

        response = await loop.run_in_executor(None, generate_response)

        # Yield the response as an async generator
        async def response_generator():
            yield response

        return response_generator()