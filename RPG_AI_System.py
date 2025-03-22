import json
import random
import time
import os
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# 모델 설정 및 로그인
login(token="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")

class MemorySystem:
    """
    Memory management system for NPCs with short-term and long-term memory
    """
    def __init__(self, npc_id, memory_path="memories"):
        self.npc_id = npc_id
        self.memory_path = memory_path
        self.short_term_memory = []
        self.long_term_memory = []
        self.memory_retention_threshold = 0.6  # Memories above this threshold move to long-term
        self.max_short_term_memories = 20
        
        # Create memory directory if it doesn't exist
        os.makedirs(memory_path, exist_ok=True)
        
        # Load existing memories if available
        self._load_memories()
    
    def add_memory(self, memory_data):
        """Add a new memory to short-term memory"""
        # Add timestamp to memory
        memory_data["timestamp"] = datetime.now().isoformat()
        memory_data["decay_factor"] = 1.0  # Initial strength of memory
        
        # Add to short-term memory
        self.short_term_memory.append(memory_data)
        
        # Consolidate important memories to long-term if needed
        self._consolidate_memories()
        
        # Save memories
        self._save_memories()
    
    def _consolidate_memories(self):
        """Move important memories to long-term and manage memory capacity"""
        # Check if short-term memory is getting full
        if len(self.short_term_memory) > self.max_short_term_memories:
            # Sort by importance/relevance
            memories_to_process = self.short_term_memory[:len(self.short_term_memory) - self.max_short_term_memories]
            self.short_term_memory = self.short_term_memory[len(self.short_term_memory) - self.max_short_term_memories:]
            
            for memory in memories_to_process:
                # Check if memory is important enough to move to long-term
                if memory.get("importance", 0) > self.memory_retention_threshold:
                    self.long_term_memory.append(memory)
    
    def update_memory_strength(self):
        """Update memory decay over time"""
        current_time = datetime.now()
        
        # Update short-term memories
        for memory in self.short_term_memory:
            memory_time = datetime.fromisoformat(memory["timestamp"])
            time_diff = (current_time - memory_time).total_seconds() / 3600  # hours
            
            # Exponential decay formula
            memory["decay_factor"] = max(0.1, memory["decay_factor"] * (0.9 ** time_diff))
    
    def retrieve_relevant_memories(self, context, max_memories=5):
        """Retrieve memories most relevant to the current context"""
        # Update memory strengths before retrieval
        self.update_memory_strength()
        
        # Combine short and long-term memories for retrieval
        all_memories = self.short_term_memory + self.long_term_memory
        
        # Score memories by relevance to context and decay factor
        scored_memories = []
        for memory in all_memories:
            relevance_score = self._calculate_relevance(memory, context)
            final_score = relevance_score * memory["decay_factor"]
            scored_memories.append((memory, final_score))
        
        # Sort by score and return top memories
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, score in scored_memories[:max_memories]]
    
    def _calculate_relevance(self, memory, context):
        """Calculate how relevant a memory is to the current context"""
        # Simple keyword matching for relevance (can be enhanced with embeddings)
        relevance = 0.1  # Base relevance
        
        memory_content = json.dumps(memory).lower()
        context_lower = context.lower()
        
        # Check for keyword matches
        context_keywords = context_lower.split()
        for keyword in context_keywords:
            if len(keyword) > 3 and keyword in memory_content:
                relevance += 0.2
        
        # Check for topic match
        if "topic" in memory and memory["topic"].lower() in context_lower:
            relevance += 0.4
            
        # Boost relevance of emotional memories
        if "emotion_impact" in memory and memory["emotion_impact"] == "strong":
            relevance += 0.3
            
        return min(1.0, relevance)
    
    def _save_memories(self):
        """Save memories to disk"""
        memory_data = {
            "short_term": self.short_term_memory,
            "long_term": self.long_term_memory
        }
        
        with open(f"{self.memory_path}/{self.npc_id}_memories.json", "w") as f:
            json.dump(memory_data, f, indent=2)
    
    def _load_memories(self):
        """Load memories from disk"""
        memory_file = f"{self.memory_path}/{self.npc_id}_memories.json"
        if os.path.exists(memory_file):
            try:
                with open(memory_file, "r") as f:
                    memory_data = json.load(f)
                    self.short_term_memory = memory_data.get("short_term", [])
                    self.long_term_memory = memory_data.get("long_term", [])
            except Exception as e:
                print(f"Error loading memories: {e}")


class NPCBrain:
    """
    AI brain for NPCs that handles decision making and interactions
    """
    def __init__(self, npc_id, npc_data, model_id="meta-llama/Llama-3.2-3B-Instruct"):
        self.npc_id = npc_id
        self.npc_data = npc_data
        self.memory = MemorySystem(npc_id)
        self.model_id = model_id
        
        # Load AI model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        
        # Current state
        self.current_state = {
            "location": "town_square",
            "time_of_day": "morning",
            "weather": "sunny",
            "nearby_npcs": [],
            "emotion": "neutral",
            "energy": 100,
            "health": 100
        }
        
    def update_state(self, new_state_data):
        """Update NPC's current state"""
        self.current_state.update(new_state_data)
        
    def make_decision(self, situation, options=None):
        """Make a decision based on the current situation and NPC's personality/memories"""
        # Get relevant memories
        relevant_memories = self.memory.retrieve_relevant_memories(situation)
        memory_context = self._format_memories_for_prompt(relevant_memories)
        
        # Format personality traits
        personality = self._format_personality_for_prompt()
        
        # Create prompt for the AI model
        prompt = self._create_decision_prompt(situation, personality, memory_context, options)
        
        # Generate response from the model
        response = self.pipe(prompt, max_new_tokens=150, temperature=0.7)
        
        # Process and return the decision
        decision = self._extract_decision(response[0]["generated_text"])
        
        # Save the decision as a memory
        self.memory.add_memory({
            "type": "decision",
            "situation": situation,
            "decision": decision,
            "importance": 0.5  # Medium importance by default
        })
        
        return decision
    
    def interact_with_npc(self, other_npc, topic, initial_statement=None):
        """Handle interaction with another NPC"""
        # Get relevant memories about the other NPC
        memories_about_npc = self.memory.retrieve_relevant_memories(f"interaction with {other_npc.npc_id}")
        memory_context = self._format_memories_for_prompt(memories_about_npc)
        
        # Format conversation prompt
        personality = self._format_personality_for_prompt()
        prompt = self._create_conversation_prompt(other_npc, topic, personality, memory_context, initial_statement)
        
        # Generate response
        response = self.pipe(prompt, max_new_tokens=200, temperature=0.8)
        conversation_text = self._extract_conversation(response[0]["generated_text"])
        
        # Save the interaction as a memory
        self.memory.add_memory({
            "type": "conversation",
            "partner": other_npc.npc_id,
            "topic": topic,
            "content": conversation_text,
            "importance": 0.7  # Conversations are usually important
        })
        
        return conversation_text
    
    def observe_environment(self, environment_data):
        """Process and react to environment observations"""
        # Extract relevant information from environment
        location = environment_data.get("location", "unknown")
        nearby_objects = environment_data.get("nearby_objects", [])
        events = environment_data.get("events", [])
        
        # Update NPC state
        self.update_state({
            "location": location,
            "nearby_objects": nearby_objects
        })
        
        # Create observation memory
        for event in events:
            importance = event.get("importance", 0.4)
            self.memory.add_memory({
                "type": "observation",
                "event": event.get("description", ""),
                "location": location,
                "importance": importance
            })
        
        # Decide if any action is needed based on observations
        if events:
            situation = f"Observed: {', '.join([e.get('description', '') for e in events])} at {location}"
            return self.make_decision(situation)
        
        return None
    
    def _format_memories_for_prompt(self, memories):
        """Format memories for inclusion in prompt"""
        if not memories:
            return "No relevant memories."
        
        memory_texts = []
        for memory in memories:
            memory_type = memory.get("type", "general")
            
            if memory_type == "conversation":
                memory_texts.append(f"You talked with {memory.get('partner', 'someone')} about {memory.get('topic', 'something')}. Content: {memory.get('content', '')}")
            
            elif memory_type == "observation":
                memory_texts.append(f"You observed: {memory.get('event', '')} at {memory.get('location', 'somewhere')}")
            
            elif memory_type == "decision":
                memory_texts.append(f"When faced with '{memory.get('situation', '')}', you decided to {memory.get('decision', 'do something')}")
            
            else:
                memory_texts.append(f"Memory: {json.dumps(memory)}")
        
        return "Relevant memories:\n" + "\n".join(memory_texts)
    
    def _format_personality_for_prompt(self):
        """Format NPC personality traits for prompt"""
        instincts = self.npc_data.get("instincts", {})
        goals = self.npc_data.get("goals", {})
        
        personality_text = f"You are a {instincts.get('species', 'being')} with the following traits:\n"
        
        if "basic_instincts" in instincts:
            basic = instincts["basic_instincts"]
            personality_text += f"- Survival instinct: {basic.get('survival_instinct', 0.5):.1f}/1.0\n"
            personality_text += f"- Social instinct: {basic.get('social_instinct', 0.5):.1f}/1.0\n"
            personality_text += f"- Fear factor: {basic.get('fear_factor', 0.5):.1f}/1.0\n"
        
        personality_text += f"Your current goal is: {goals.get('current_goal', 'survive')}\n"
        personality_text += f"Your long-term goal is: {goals.get('long_term_goal', 'thrive')}\n"
        
        return personality_text
    
    def _create_decision_prompt(self, situation, personality, memory_context, options=None):
        """Create a prompt for decision making"""
        prompt = f"""<|system|>
You are an autonomous NPC in a fantasy RPG world. You make decisions based on your character traits, memories, and the current situation.

{personality}

Current state:
- Location: {self.current_state.get('location', 'unknown')}
- Time: {self.current_state.get('time_of_day', 'daytime')}
- Weather: {self.current_state.get('weather', 'clear')}
- Current emotion: {self.current_state.get('emotion', 'neutral')}
- Energy level: {self.current_state.get('energy', 100)}/100
- Health: {self.current_state.get('health', 100)}/100

{memory_context}

Current situation:
{situation}
"""

        if options:
            prompt += "\nYou can choose from the following options:\n"
            for i, option in enumerate(options, 1):
                prompt += f"{i}. {option}\n"
            prompt += "\nThink through each option and its consequences before deciding."
        else:
            prompt += "\nDecide what action to take based on your personality, memories, and the current situation."

        prompt += "\nYou should decide on a specific action and return it in the format: ACTION: [your chosen action]"
        
        prompt += "\n<|user|>\nWhat will you do in this situation?\n<|assistant|>"
        
        return prompt
    
    def _create_conversation_prompt(self, other_npc, topic, personality, memory_context, initial_statement=None):
        """Create a prompt for conversation with another NPC"""
        other_personality = f"a {other_npc.npc_data.get('instincts', {}).get('species', 'being')}"
        
        prompt = f"""<|system|>
You are an autonomous NPC in a fantasy RPG world engaged in a conversation with another character.

{personality}

Current state:
- Location: {self.current_state.get('location', 'unknown')}
- Time: {self.current_state.get('time_of_day', 'daytime')}
- Current emotion: {self.current_state.get('emotion', 'neutral')}

You are talking to {other_personality} about {topic}.

{memory_context}

You should respond in a conversational manner, expressing your character's thoughts, feelings, and desires based on your personality and memories.
<|user|>
"""
        
        if initial_statement:
            prompt += f"The other character says: \"{initial_statement}\"\n\nHow do you respond?\n<|assistant|>"
        else:
            prompt += f"Start a conversation with the other character about {topic}.\n<|assistant|>"
        
        return prompt
    
    def _extract_decision(self, generated_text):
        """Extract the decision from the generated text"""
        # Look for the ACTION: format first
        if "ACTION:" in generated_text:
            action_part = generated_text.split("ACTION:")[1].strip()
            # Get the first line only
            action = action_part.split("\n")[0].strip()
            return action
            
        # If no specific format is found, use the last paragraph as the decision
        paragraphs = generated_text.split("\n\n")
        for paragraph in reversed(paragraphs):
            if paragraph.strip():
                return paragraph.strip()
                
        # Fallback
        return generated_text.strip()
    
    def _extract_conversation(self, generated_text):
        """Extract conversation response from generated text"""
        # Remove any system or user parts if included in the response
        if "<|assistant|>" in generated_text:
            response_part = generated_text.split("<|assistant|>")[1].strip()
        else:
            response_part = generated_text.strip()
            
        return response_part


class GameEnvironment:
    """
    Simulates the game environment for testing AI interactions
    """
    def __init__(self):
        self.locations = {
            "town_square": {
                "description": "The central gathering place of the town",
                "connected_to": ["marketplace", "tavern", "blacksmith"],
                "objects": ["fountain", "bench", "notice_board"]
            },
            "marketplace": {
                "description": "Busy trading area with various stalls",
                "connected_to": ["town_square", "northern_gate"],
                "objects": ["food_stall", "weapon_stall", "jewelry_stall"]
            },
            "tavern": {
                "description": "A warm, lively place to drink and socialize",
                "connected_to": ["town_square"],
                "objects": ["bar_counter", "fireplace", "tables", "chairs"]
            },
            "blacksmith": {
                "description": "Hot workshop where weapons and tools are forged",
                "connected_to": ["town_square"],
                "objects": ["anvil", "forge", "weapons_rack", "water_barrel"]
            },
            "northern_gate": {
                "description": "Main entrance to the town from the north",
                "connected_to": ["marketplace", "forest_path"],
                "objects": ["guard_post", "wooden_gate", "torch"]
            },
            "forest_path": {
                "description": "Winding trail through the dense forest",
                "connected_to": ["northern_gate", "forest_clearing"],
                "objects": ["tall_trees", "mushrooms", "fallen_log"]
            },
            "forest_clearing": {
                "description": "Open space within the forest with a small pond",
                "connected_to": ["forest_path", "cave_entrance"],
                "objects": ["pond", "wild_flowers", "large_rock"]
            },
            "cave_entrance": {
                "description": "Dark opening leading into an ancient cave system",
                "connected_to": ["forest_clearing", "cave_interior"],
                "objects": ["stalagmites", "bat_nest", "old_torch"]
            }
        }
        
        self.time_of_day = "morning"
        self.weather = "clear"
        self.npcs = {}
        self.events = []
        
    def add_npc(self, npc):
        """Add an NPC to the environment"""
        self.npcs[npc.npc_id] = npc
        
    def update_time(self, new_time):
        """Update time of day"""
        self.time_of_day = new_time
        
        # Notify NPCs about time change
        for npc_id, npc in self.npcs.items():
            npc.update_state({"time_of_day": new_time})
            
    def update_weather(self, new_weather):
        """Update weather conditions"""
        self.weather = new_weather
        
        # Notify NPCs about weather change
        for npc_id, npc in self.npcs.items():
            npc.update_state({"weather": new_weather})
    
    def get_environment_data(self, location):
        """Get environment data for a specific location"""
        if location not in self.locations:
            return {"error": "Invalid location"}
            
        location_data = self.locations[location]
        
        # Find NPCs at this location
        npcs_here = []
        for npc_id, npc in self.npcs.items():
            if npc.current_state.get("location") == location:
                npcs_here.append(npc_id)
                
        return {
            "location": location,
            "description": location_data["description"],
            "connected_to": location_data["connected_to"],
            "nearby_objects": location_data["objects"],
            "nearby_npcs": npcs_here,
            "time_of_day": self.time_of_day,
            "weather": self.weather,
            "events": self.get_current_events(location)
        }
    
    def add_event(self, event_data):
        """Add an event to the environment"""
        event_data["timestamp"] = time.time()
        self.events.append(event_data)
    
    def get_current_events(self, location=None):
        """Get current events, optionally filtered by location"""
        # Remove old events (older than 15 minutes)
        current_time = time.time()
        self.events = [e for e in self.events if current_time - e["timestamp"] < 900]
        
        if location:
            return [e for e in self.events if e.get("location") == location]
        return self.events


# 기본 캐릭터 데이터 생성 함수
def generate_ai_character_data():
    """Generate random character data for an NPC"""
    species = random.choice(["Human", "Elf", "Dwarf", "Orc", "Halfling"])
    
    # Personality traits on a scale of 0-1
    personality = {
        "aggressiveness": random.uniform(0.0, 1.0),
        "curiosity": random.uniform(0.0, 1.0),
        "sociability": random.uniform(0.0, 1.0),
        "bravery": random.uniform(0.0, 1.0),
        "loyalty": random.uniform(0.0, 1.0),
        "honesty": random.uniform(0.0, 1.0)
    }
    
    # Determine dominant traits
    dominant_traits = []
    for trait, value in personality.items():
        if value > 0.7:
            dominant_traits.append(trait)
    
    # Basic instincts
    instincts = {
        "species": species,
        "basic_instincts": {
            "survival_instinct": random.uniform(0.5, 1.0),
            "social_instinct": random.uniform(0.3, 1.0),
            "fear_factor": random.uniform(0.0, 0.8)
        },
        "dominant_traits": dominant_traits
    }

    # Knowledge and skills
    knowledge = []
    possible_skills = {
        "Human": ["Farming", "Blacksmithing", "Cooking", "Hunting", "Fishing", "Trading", "Medicine"],
        "Elf": ["Archery", "Magic", "Herbalism", "Music", "Crafting", "History", "Tracking"],
        "Dwarf": ["Mining", "Metallurgy", "Stonework", "Brewing", "Engineering", "Combat", "Forging"],
        "Orc": ["Warfare", "Intimidation", "Survival", "Weapon-making", "Hunting", "Tribal Lore", "Shamanism"],
        "Halfling": ["Cooking", "Stealth", "Gardening", "Brewing", "Trading", "Storytelling", "Pickpocketing"]
    }
    
    # Add species-specific skills
    for _ in range(3):
        skill = random.choice(possible_skills[species])
        knowledge.append({
            "skill": skill,
            "proficiency": random.uniform(0.3, 1.0),
            "experience_years": random.randint(1, 20)
        })
    
    # Add some general knowledge
    for _ in range(2):
        topic = random.choice(["Local Geography", "Ancient Legends", "Common Herbs", "Trade Routes", "Political Situation"])
        knowledge.append({
            "topic": topic,
            "confidence": random.uniform(0.3, 0.9),
            "verified": random.choice([True, False])
        })

    # Current emotional state
    emotions = {
        "current_emotion": random.choice(["happy", "neutral", "curious", "anxious", "angry", "sad"]),
        "emotional_stability": random.uniform(0.3, 0.9),
        "stress_level": random.uniform(0.0, 0.7)
    }

    # Goals system
    goals = {
        "current_goal": random.choice([
            "find_valuable_items", 
            "make_new_friends", 
            "learn_new_skill", 
            "explore_new_areas",
            "complete_current_task",
            "earn_money",
            "improve_reputation"
        ]),
        "goal_priority": random.uniform(0.5, 1.0),
        "long_term_goal": random.choice([
            "become_master_craftsman", 
            "find_true_love", 
            "achieve_wealth", 
            "gain_power",
            "discover_ancient_knowledge",
            "establish_business",
            "find_peaceful_life"
        ]),
        "goal_progress": random.uniform(0.0, 0.3)
    }

    # Memory initialization (empty - will be filled during gameplay)
    recent_conversations = []
    long_term_memory = []

    return {
        "instincts": instincts,
        "personality": personality,
        "knowledge": knowledge,
        "emotions": emotions,
        "goals": goals,
        "recent_conversations": recent_conversations,
        "long_term_memory": long_term_memory
    } 