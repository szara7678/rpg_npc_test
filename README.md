<<<<<<< HEAD
<<<<<<< HEAD
# rpg_npc_test
=======
# RPG AI System
=======
# RPG AI System!
>>>>>>> 7ab7424 (first2)

An advanced AI system for role-playing games with memory management, NPC interactions, and autonomous decision making capabilities.

## Features

- **Memory Management**
  - Short-term and long-term memory storage
  - Memory decay and consolidation
  - Context-aware memory retrieval
  - Persistent memory through game sessions

- **NPC Interactions**
  - Realistic NPC-to-NPC conversations
  - Dynamic dialogue based on personality and memories
  - Relationship tracking between NPCs

- **Autonomous Decision Making**
  - NPCs make decisions based on personality, memories, and current situation
  - Environmental awareness and response
  - Goal-oriented behavior

- **Environmental Simulation**
  - Time and weather systems
  - Location-based event handling
  - Object and NPC presence tracking

## System Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers library 4.21.0+
- Hugging Face account for model access

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install torch transformers huggingface_hub
```

3. Configure your Hugging Face token in the code or as an environment variable:

```python
import os
from huggingface_hub import login

# Either set in code
login(token="your_huggingface_token")

# Or use environment variable
# export HUGGINGFACE_TOKEN="your_huggingface_token"
# login(token=os.environ.get("HUGGINGFACE_TOKEN"))
```

## Quick Start

Run the test script to see the system in action:

```bash
python RPG_AI_Test.py
```

## Usage Examples

### Creating an NPC

```python
from RPG_AI_System import generate_ai_character_data, NPCBrain

# Generate random character data
character_data = generate_ai_character_data()

# Or create custom character data
character_data = {
    "instincts": {
        "species": "Human",
        "basic_instincts": {
            "survival_instinct": 0.8,
            "social_instinct": 0.7,
            "fear_factor": 0.3
        }
    },
    "goals": {
        "current_goal": "find_valuable_items",
        "goal_priority": 0.8,
        "long_term_goal": "become_master_craftsman",
        "goal_progress": 0.2
    }
}

# Create NPC brain
npc = NPCBrain("blacksmith_john", character_data)

# Update NPC state
npc.update_state({
    "location": "blacksmith",
    "time_of_day": "morning",
    "weather": "sunny",
    "emotion": "happy"
})
```

### Decision Making

```python
# NPC makes a decision based on a situation
situation = "A customer wants to haggle over the price of a sword"
decision = npc.make_decision(situation)
print(f"NPC decides to: {decision}")

# Decision with options
options = [
    "Lower the price slightly",
    "Refuse to negotiate",
    "Offer an alternative item",
    "Include a free item to maintain the price"
]
decision = npc.make_decision(situation, options)
print(f"NPC decides to: {decision}")
```

### NPC Conversation

```python
# Create two NPCs
merchant = NPCBrain("merchant_anna", merchant_data)
blacksmith = NPCBrain("blacksmith_john", blacksmith_data)

# Initiate conversation
initial_statement = "Hello John, I need ten iron daggers by next week. Can you make them?"
response = blacksmith.interact_with_npc(merchant, "business deal", initial_statement)
print(f"Blacksmith responds: {response}")

# Merchant replies
counter_response = merchant.interact_with_npc(blacksmith, "business deal", response)
print(f"Merchant responds: {counter_response}")
```

### Environmental Observation

```python
# Create game environment
game_env = GameEnvironment()

# Add NPCs to environment
game_env.add_npc(merchant)
game_env.add_npc(blacksmith)

# Add an event
game_env.add_event({
    "description": "A loud explosion heard from the mine",
    "location": "marketplace",
    "importance": 0.9
})

# NPC observes environment
env_data = game_env.get_environment_data(merchant.current_state["location"])
reaction = merchant.observe_environment(env_data)
print(f"Merchant's reaction: {reaction}")
```

## Customizing AI Models

You can configure different models in the `model_config.py` file:

```python
from model_config import get_model_config, ModelType

# Use a more powerful model for important NPCs
model_config = get_model_config(ModelType.LLAMA3, "8B")
important_npc = NPCBrain("quest_giver", character_data, model_id=model_config["model_id"])

# Use a lightweight model for background NPCs
light_model_config = get_model_config(ModelType.GPT_NEO, "1.3B")
background_npc = NPCBrain("villager", character_data, model_id=light_model_config["model_id"])
```

## Memory System

The memory system stores NPC experiences and allows for realistic recall:

```python
# Add a memory
npc.memory.add_memory({
    "type": "conversation",
    "partner": "player",
    "topic": "rare sword",
    "content": "The player asked about the legendary blade of the mountains",
    "importance": 0.8
})

# Retrieve relevant memories
memories = npc.memory.retrieve_relevant_memories("legendary blade mountain")

# Memories affect decisions
situation = "The player has returned asking about mountain treasures"
decision = npc.make_decision(situation)  # Will be influenced by stored memories
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

Developed for advanced NPC AI in role-playing games.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
>>>>>>> 8797545 (first)
