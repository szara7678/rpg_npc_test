import time
import random
from RPG_AI_System import generate_ai_character_data, NPCBrain, GameEnvironment

# 테스트 시나리오 실행
def run_test_scenario():
    """
    Run a test scenario to demonstrate the AI system capabilities
    """
    print("=== RPG AI System Test ===")
    
    # Initialize game environment
    print("\nInitializing game environment...")
    game_env = GameEnvironment()
    
    # Create NPCs
    print("\nCreating NPCs...")
    npc_data = []
    
    # Create a merchant NPC
    merchant_data = generate_ai_character_data()
    merchant_data["instincts"]["species"] = "Human"
    merchant_data["goals"]["current_goal"] = "earn_money"
    merchant_data["goals"]["long_term_goal"] = "establish_business"
    merchant_data["knowledge"].append({
        "skill": "Trading",
        "proficiency": 0.9,
        "experience_years": 15
    })
    merchant = NPCBrain("merchant_thomas", merchant_data)
    merchant.update_state({
        "location": "marketplace",
        "emotion": "happy"
    })
    npc_data.append(("Merchant Thomas", merchant))
    game_env.add_npc(merchant)
    
    # Create a guard NPC
    guard_data = generate_ai_character_data()
    guard_data["instincts"]["species"] = "Human"
    guard_data["personality"]["bravery"] = 0.9
    guard_data["personality"]["loyalty"] = 0.95
    guard_data["goals"]["current_goal"] = "protect_town"
    guard_data["goals"]["long_term_goal"] = "become_captain"
    guard_data["knowledge"].append({
        "skill": "Combat",
        "proficiency": 0.85,
        "experience_years": 8
    })
    guard = NPCBrain("guard_roland", guard_data)
    guard.update_state({
        "location": "northern_gate",
        "emotion": "alert"
    })
    npc_data.append(("Guard Roland", guard))
    game_env.add_npc(guard)
    
    # Create a mage NPC
    mage_data = generate_ai_character_data()
    mage_data["instincts"]["species"] = "Elf"
    mage_data["personality"]["curiosity"] = 0.95
    mage_data["goals"]["current_goal"] = "research_magic"
    mage_data["goals"]["long_term_goal"] = "discover_ancient_knowledge"
    mage_data["knowledge"].append({
        "skill": "Magic",
        "proficiency": 0.8,
        "experience_years": 120
    })
    mage = NPCBrain("mage_elindra", mage_data)
    mage.update_state({
        "location": "tavern",
        "emotion": "curious"
    })
    npc_data.append(("Mage Elindra", mage))
    game_env.add_npc(mage)
    
    print("\nNPCs created:")
    for name, npc in npc_data:
        species = npc.npc_data["instincts"]["species"]
        goal = npc.npc_data["goals"]["current_goal"]
        location = npc.current_state["location"]
        print(f"- {name} ({species}) at {location}, goal: {goal}")
    
    # Run test scenarios
    print("\n=== Test Scenario 1: Environmental Response ===")
    print("\nChanging weather to 'rainy'...")
    game_env.update_weather("rainy")
    
    print("\nNPC responses to weather change:")
    for name, npc in npc_data:
        situation = "It has started raining heavily. The streets are getting muddy and visibility is reduced."
        decision = npc.make_decision(situation)
        print(f"- {name}: {decision}")
    
    # Add an event
    print("\n=== Test Scenario 2: Event Response ===")
    print("\nAdding event: Thief spotted in marketplace...")
    game_env.add_event({
        "description": "A thief was spotted stealing from a stall",
        "location": "marketplace",
        "importance": 0.8
    })
    
    # NPCs observe environment
    for name, npc in npc_data:
        env_data = game_env.get_environment_data(npc.current_state["location"])
        response = npc.observe_environment(env_data)
        
        if response:
            print(f"- {name} observed events and decided: {response}")
        else:
            print(f"- {name} observed nothing unusual in their location")
    
    # Test memory system
    print("\n=== Test Scenario 3: Memory and Recall ===")
    
    # Add memories to the merchant
    merchant.memory.add_memory({
        "type": "observation",
        "event": "Guard Roland chased away a thief",
        "location": "marketplace",
        "importance": 0.7,
        "emotion_impact": "strong"
    })
    
    merchant.memory.add_memory({
        "type": "conversation",
        "partner": "mage_elindra",
        "topic": "rare magical ingredients",
        "content": "Elindra mentioned needing rare herbs from the forest",
        "importance": 0.6
    })
    
    # Time passes (simulate memory decay)
    print("\nSimulating time passing...")
    merchant.memory.update_memory_strength()
    
    # Test memory recall
    print("\nMerchant Thomas tries to recall information about Guard Roland:")
    memories = merchant.memory.retrieve_relevant_memories("Guard Roland thief")
    if memories:
        print(f"Retrieved {len(memories)} memories:")
        for memory in memories:
            if "event" in memory:
                print(f"- Remembered: {memory['event']}")
            elif "content" in memory:
                print(f"- Remembered: {memory['content']}")
    else:
        print("No relevant memories found")
    
    # Test NPC conversation
    print("\n=== Test Scenario 4: NPC Conversation ===")
    
    # Move mage to marketplace
    print("\nMage Elindra moves to marketplace to meet Merchant Thomas...")
    mage.update_state({"location": "marketplace"})
    
    # Conversation
    print("\nConversation about magical ingredients:")
    initial_statement = "Greetings, Thomas. I've been looking for moonflower petals for a spell. Do you have any in stock?"
    response = merchant.interact_with_npc(mage, "magical ingredients", initial_statement)
    print(f"Merchant Thomas: {response}")
    
    # Now mage responds
    response = mage.interact_with_npc(merchant, "magical ingredients", response)
    print(f"Mage Elindra: {response}")
    
    # Test complex decision making
    print("\n=== Test Scenario 5: Complex Decision Making ===")
    
    situation = """
    A stranger has arrived at the northern gate. He claims to be a traveling merchant, but his clothes are worn and he seems nervous.
    He wants to enter the town to sell his wares, but cannot produce proper merchant credentials.
    The town has had problems with spies and thieves recently.
    """
    
    options = [
        "Allow him to enter but assign someone to watch him",
        "Deny entry completely",
        "Allow entry after confiscating his bags for inspection",
        "Detain him for questioning"
    ]
    
    print("\nGuard Roland faces a security decision at the gate...")
    print(f"Situation: {situation}")
    
    decision = guard.make_decision(situation, options)
    print(f"Guard Roland decides: {decision}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    run_test_scenario() 