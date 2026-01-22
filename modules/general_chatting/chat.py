from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage

class ChatBot:
    def __init__(self, model_name="llama3.1"):
        self.model = ChatOllama(model=model_name)
        self.chat_history = []  # Store HumanMessage and AIMessage objects

    def chat(self, prompt):
        # Append the user's message
        self.chat_history.append(HumanMessage(content=prompt))

        # Get response from the model
        response = self.model.invoke(self.chat_history)

        # Append AI's response to chat history
        self.chat_history.append(response)

        # Return the latest response
        return response.content.strip()

def return_chat(prompt):
    # Initialize the model
    c = ChatBot()
    # Get the response
    response = c.chat(prompt)
    return response

# # Example usage
# bot = ChatBot()


# print(bot.chat("Tell me a joke."))
# print(bot.chat("Now explain that joke to me."))
