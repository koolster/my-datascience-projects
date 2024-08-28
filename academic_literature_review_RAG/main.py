import os
import yaml
from src.chatbot import run_chatbot

def load_config():
    """
    Load the configuration from the YAML file.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Check for API key
    if 'OPENAI_API_KEY' not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables. Please set it before running the script.")
    
    # Load configuration and run the chatbot
    config = load_config()
    run_chatbot(config)

if __name__ == "__main__":
    main()
