"""
Enhanced memory retrieval system using embeddings
This augments the basic memory system with semantic search capabilities
"""

import os
import json
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

class EmbeddingMemorySystem:
    """
    Memory system that uses embeddings for more accurate memory retrieval
    """
    def __init__(self, npc_id, memory_path="memories", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.npc_id = npc_id
        self.memory_path = memory_path
        self.embedding_model_name = embedding_model
        
        # Memory storage
        self.short_term_memory = []
        self.long_term_memory = []
        self.embeddings_cache = {}
        
        # Configuration
        self.memory_retention_threshold = 0.6
        self.max_short_term_memories = 20
        self.relevance_threshold = 0.7
        
        # Create memory directory
        os.makedirs(memory_path, exist_ok=True)
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        
        # Load existing memories
        self._load_memories()
    
    def add_memory(self, memory_data):
        """Add a new memory and generate its embedding"""
        # Add timestamp to memory
        memory_data["timestamp"] = datetime.now().isoformat()
        memory_data["decay_factor"] = 1.0
        
        # Generate memory key for embedding cache
        memory_key = f"{len(self.short_term_memory)}_{datetime.now().timestamp()}"
        memory_data["memory_key"] = memory_key
        
        # Generate embedding
        memory_text = self._memory_to_text(memory_data)
        embedding = self._generate_embedding(memory_text)
        self.embeddings_cache[memory_key] = embedding
        
        # Add to short-term memory
        self.short_term_memory.append(memory_data)
        
        # Consolidate important memories to long-term if needed
        self._consolidate_memories()
        
        # Save memories
        self._save_memories()
    
    def retrieve_relevant_memories(self, context, max_memories=5):
        """Retrieve memories most semantically similar to the context"""
        # Update memory strengths
        self.update_memory_strength()
        
        # Generate embedding for the context
        context_embedding = self._generate_embedding(context)
        
        # Combine short and long-term memories for retrieval
        all_memories = self.short_term_memory + self.long_term_memory
        
        # Calculate similarity scores
        scored_memories = []
        for memory in all_memories:
            memory_key = memory.get("memory_key")
            
            # Generate embedding if not in cache
            if memory_key not in self.embeddings_cache:
                memory_text = self._memory_to_text(memory)
                self.embeddings_cache[memory_key] = self._generate_embedding(memory_text)
            
            # Calculate similarity
            similarity = 1 - cosine(context_embedding, self.embeddings_cache[memory_key])
            
            # Apply decay factor to score
            final_score = similarity * memory.get("decay_factor", 0.5)
            scored_memories.append((memory, final_score))
        
        # Sort by score and return top memories
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = [memory for memory, score in scored_memories[:max_memories] if score > self.relevance_threshold]
        
        # If no memories meet the threshold but we have results, return at least one
        if not top_memories and scored_memories:
            top_memories = [scored_memories[0][0]]
            
        return top_memories
    
    def _generate_embedding(self, text):
        """Generate embedding for a piece of text"""
        # Tokenize and prepare for model
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**inputs)
        
        # Use mean pooling to get sentence embedding
        token_embeddings = model_output.last_hidden_state
        attention_mask = inputs["attention_mask"]
        
        # Mask padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings along sequence dimension
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Sum attention mask for averaging
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Average embeddings
        embeddings = sum_embeddings / sum_mask
        
        # Convert to numpy array
        return embeddings[0].numpy()
    
    def _memory_to_text(self, memory):
        """Convert memory object to text representation for embedding"""
        memory_type = memory.get("type", "general")
        memory_text = ""
        
        if memory_type == "conversation":
            memory_text = f"Conversation with {memory.get('partner', 'someone')} about {memory.get('topic', 'something')}. {memory.get('content', '')}"
        
        elif memory_type == "observation":
            memory_text = f"Observed: {memory.get('event', '')} at {memory.get('location', 'somewhere')}"
        
        elif memory_type == "decision":
            memory_text = f"Decision: When faced with '{memory.get('situation', '')}', decided to {memory.get('decision', 'do something')}"
        
        else:
            # Generic handling for other memory types
            memory_text = json.dumps(memory)
        
        return memory_text
    
    def update_memory_strength(self):
        """Update memory decay over time"""
        current_time = datetime.now()
        
        # Update short-term memories
        for memory in self.short_term_memory:
            if "timestamp" in memory:
                memory_time = datetime.fromisoformat(memory["timestamp"])
                time_diff = (current_time - memory_time).total_seconds() / 3600  # hours
                
                # Exponential decay formula
                memory["decay_factor"] = max(0.1, memory.get("decay_factor", 1.0) * (0.9 ** time_diff))
    
    def _consolidate_memories(self):
        """Move important memories to long-term and manage memory capacity"""
        # Check if short-term memory is getting full
        if len(self.short_term_memory) > self.max_short_term_memories:
            # Sort by importance/relevance
            self.short_term_memory.sort(key=lambda x: x.get("importance", 0.5) * x.get("decay_factor", 1.0), reverse=True)
            
            # Keep top memories in short-term
            memories_to_process = self.short_term_memory[self.max_short_term_memories:]
            self.short_term_memory = self.short_term_memory[:self.max_short_term_memories]
            
            for memory in memories_to_process:
                # Check if memory is important enough to move to long-term
                if memory.get("importance", 0) > self.memory_retention_threshold:
                    self.long_term_memory.append(memory)
    
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
                    
                    # Rebuild embeddings cache
                    for memory in self.short_term_memory + self.long_term_memory:
                        if "memory_key" in memory:
                            memory_text = self._memory_to_text(memory)
                            self.embeddings_cache[memory["memory_key"]] = self._generate_embedding(memory_text)
            except Exception as e:
                print(f"Error loading memories: {e}")


# Example usage of embedding-based memory system
def demo_embedding_memory():
    """Demonstrate embedding-based memory retrieval"""
    # Create memory system for an NPC
    memory_system = EmbeddingMemorySystem("tavern_keeper")
    
    # Add sample memories
    memory_system.add_memory({
        "type": "conversation",
        "partner": "adventurer",
        "topic": "dragon sightings",
        "content": "The adventurer mentioned seeing a red dragon near the mountains to the north.",
        "importance": 0.8
    })
    
    memory_system.add_memory({
        "type": "observation",
        "event": "A group of suspicious mercenaries entered the tavern and were whispering about a hidden treasure.",
        "location": "tavern",
        "importance": 0.7
    })
    
    memory_system.add_memory({
        "type": "conversation",
        "partner": "village_elder",
        "topic": "upcoming festival",
        "content": "The elder asked to prepare extra food for the harvest festival next week.",
        "importance": 0.6
    })
    
    # Query for different contexts
    print("Querying for dragon-related memories:")
    dragon_memories = memory_system.retrieve_relevant_memories("Have you heard anything about dragons recently?")
    for memory in dragon_memories:
        print(f"- {memory.get('type')}: {memory.get('content', memory.get('event', ''))}")
    
    print("\nQuerying for treasure-related memories:")
    treasure_memories = memory_system.retrieve_relevant_memories("I'm looking for valuable treasures in this area.")
    for memory in treasure_memories:
        print(f"- {memory.get('type')}: {memory.get('content', memory.get('event', ''))}")
    
    print("\nQuerying for festival-related memories:")
    festival_memories = memory_system.retrieve_relevant_memories("When is the next town celebration?")
    for memory in festival_memories:
        print(f"- {memory.get('type')}: {memory.get('content', memory.get('event', ''))}")


if __name__ == "__main__":
    demo_embedding_memory() 