# Multi-Modal, Multi-Purpose AI Agent

This project implements a modular multi-purpose AI assistant that understands natural language user queries and dynamically routes them to task-specific modules. A Transformer-based (BERT) intent classifier serves as the core decision layer, enabling accurate intent detection and scalable module routing.

The system is designed with a cascading architecture that separates intent prediction from task execution, allowing easy extensibility and maintainability.



## Architecture

The system consists of two layers:

  Intent Classification Layer
        Fine-tuned BERT model for multi-class intent classification

  Task Execution Layer
        Independent modules triggered based on predicted intent

Flow: User Input → Intent Classification → Module Routing → Task Execution → Output




## Features
- **General Chatting**
    Conversational interaction using a locally hosted language model.

- **Notes Maker**  
  Extracts text from images and generates summarized notes.

- **Intent Classifier**  
  Recognizes user queries and triggers the appropriate module.

- **Gmail Operations**  
  Sends, drafts, and manages emails through the AI agent.

- **NL2SQL**  
  Converts natural language queries into SQL commands for database interaction.

- **Multi-Modal AI**  
  Handles text, audio, and image inputs.

- **Streamlit Frontend**  
  Provides a web-based interface for interacting with the agent.

>  **Note:** Some features are not uploaded due to space and compatibility constraints.

