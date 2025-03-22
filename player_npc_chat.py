import time
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from RPG_AI_System import NPCBrain, generate_ai_character_data, GameEnvironment

# Hugging Face 토큰으로 로그인
login(token="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")

class PlayerInteraction:
    """
    플레이어와 NPC 간 대화 시뮬레이션을 위한 클래스
    """
    def __init__(self, player_name="Player", model_id="meta-llama/Llama-3.2-3B-Instruct"):
        """
        플레이어 상호작용 시스템 초기화
        
        Args:
            player_name (str): 플레이어 이름
            model_id (str): 사용할 AI 모델 ID
        """
        self.player_name = player_name
        self.model_id = model_id
        
        # 게임 환경 초기화
        self.game_env = GameEnvironment()
        
        # 플레이어 정보 초기화
        self.player_info = {
            "name": player_name,
            "location": "town_square",
            "reputation": {
                "general": 0.5,  # 0.0: 최악, 1.0: 최상
                "factions": {}
            },
            "conversation_history": []
        }
        
        # NPC 목록 초기화
        self.npcs = {}
    
    def create_npc(self, npc_id, npc_name, species=None, location=None, custom_data=None):
        """
        새로운 NPC 생성
        
        Args:
            npc_id (str): NPC의 고유 ID
            npc_name (str): NPC의 이름
            species (str, optional): NPC의 종족
            location (str, optional): NPC의 위치
            custom_data (dict, optional): 사용자 정의 NPC 데이터
            
        Returns:
            NPCBrain: 생성된 NPC 객체
        """
        # NPC 데이터 생성 또는 사용자 정의 데이터 사용
        if custom_data:
            npc_data = custom_data
        else:
            npc_data = generate_ai_character_data()
            
        # 종족과 위치 설정
        if species:
            npc_data["instincts"]["species"] = species
            
        # NPC 객체 생성
        npc = NPCBrain(npc_id, npc_data, model_id=self.model_id)
        
        # 초기 상태 설정
        initial_state = {
            "name": npc_name
        }
        
        if location:
            initial_state["location"] = location
            
        npc.update_state(initial_state)
        
        # 게임 환경에 NPC 추가
        self.npcs[npc_id] = npc
        self.game_env.add_npc(npc)
        
        return npc
    
    def chat_with_npc(self, npc_id, player_message):
        """
        NPC와 대화 시뮬레이션
        
        Args:
            npc_id (str): 대화할 NPC의 ID
            player_message (str): 플레이어의 메시지
            
        Returns:
            str: NPC의 응답
        """
        if npc_id not in self.npcs:
            return f"오류: {npc_id} ID를 가진 NPC를 찾을 수 없습니다."
        
        npc = self.npcs[npc_id]
        
        # 플레이어 메시지 기록
        conversation_record = {
            "timestamp": time.time(),
            "player_message": player_message,
            "npc_id": npc_id,
            "npc_name": npc.current_state.get("name", npc_id)
        }
        
        # NPC에게 플레이어 정보 제공 및 대화 컨텍스트 생성
        player_context = self._create_player_context()
        
        # 플레이어 메시지에 기반한 주제 추측
        topic = self._guess_topic_from_message(player_message)
        
        # NPC의 메모리에 플레이어와의 대화 저장
        npc.memory.add_memory({
            "type": "conversation",
            "partner": f"player_{self.player_name}",
            "topic": topic,
            "content": player_message,
            "importance": 0.7,  # 플레이어와의 대화는 중요도 높게 설정
            "emotion_impact": "strong"
        })
        
        # NPC의 응답 생성 (특별 처리)
        response = self._generate_npc_response_to_player(npc, player_message, player_context, topic)
        
        # 응답 기록
        conversation_record["npc_response"] = response
        self.player_info["conversation_history"].append(conversation_record)
        
        return response
    
    def _create_player_context(self):
        """플레이어 컨텍스트 생성"""
        return {
            "name": self.player_name,
            "location": self.player_info["location"],
            "reputation": self.player_info["reputation"]["general"]
        }
    
    def _guess_topic_from_message(self, message):
        """메시지에서 주제 추측"""
        # 간단한 키워드 기반 주제 추측
        topic_keywords = {
            "quest": ["quest", "mission", "task", "help", "need", "request"],
            "trade": ["buy", "sell", "trade", "price", "cost", "money", "gold", "item"],
            "information": ["where", "what", "who", "when", "how", "why", "tell", "know", "learn"],
            "greeting": ["hello", "hi", "greetings", "good", "morning", "afternoon", "evening"],
            "farewell": ["goodbye", "bye", "farewell", "see you", "later"]
        }
        
        message_lower = message.lower()
        
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return topic
        
        # 기본값
        return "general_conversation"
    
    def _generate_npc_response_to_player(self, npc, player_message, player_context, topic):
        """
        플레이어 메시지에 대한 NPC 응답 생성
        특별 프롬프트 포맷 사용
        """
        # NPC 성격 정보 가져오기
        personality = npc._format_personality_for_prompt()
        
        # 관련 메모리 검색
        memories_about_player = npc.memory.retrieve_relevant_memories(f"conversation with {self.player_name}")
        memory_context = npc._format_memories_for_prompt(memories_about_player)
        
        # 플레이어 전용 프롬프트 생성
        prompt = f"""<|system|>
You are an autonomous NPC in a fantasy RPG world interacting with a player character.

{personality}

Current state:
- Location: {npc.current_state.get('location', 'unknown')}
- Time: {npc.current_state.get('time_of_day', 'daytime')}
- Weather: {npc.current_state.get('weather', 'clear')}
- Current emotion: {npc.current_state.get('emotion', 'neutral')}

You are talking with a player named {player_context["name"]} about {topic}.

{memory_context}

Player reputation: {player_context["reputation"]:.1f}/1.0 (higher means better reputation)

Respond in a conversational manner, expressing your character's thoughts, feelings, and personality based on your traits and memories. Be immersive and consistent with your character's background and goals.
<|user|>
The player says: "{player_message}"

How do you respond?
<|assistant|>
"""
        
        # 응답 생성
        response = npc.pipe(prompt, max_new_tokens=200, temperature=0.8)
        
        # 응답 추출
        response_text = npc._extract_conversation(response[0]["generated_text"])
        
        return response_text
    
    def move_player(self, new_location):
        """
        플레이어 위치 이동
        
        Args:
            new_location (str): 이동할 위치
            
        Returns:
            dict: 새 위치의 환경 데이터
        """
        if new_location in self.game_env.locations:
            self.player_info["location"] = new_location
            return self.get_current_location_info()
        else:
            return {"error": f"위치를 찾을 수 없습니다: {new_location}"}
    
    def get_current_location_info(self):
        """
        현재 플레이어 위치 정보 가져오기
        
        Returns:
            dict: 현재 위치 환경 데이터
        """
        location = self.player_info["location"]
        env_data = self.game_env.get_environment_data(location)
        
        # NPC 이름 추가
        npcs_at_location = []
        for npc_id in env_data.get("nearby_npcs", []):
            if npc_id in self.npcs:
                npc = self.npcs[npc_id]
                npcs_at_location.append({
                    "id": npc_id,
                    "name": npc.current_state.get("name", npc_id),
                    "species": npc.npc_data["instincts"]["species"]
                })
        
        env_data["nearby_npcs"] = npcs_at_location
        
        return env_data
    
    def get_available_locations(self):
        """
        이동 가능한 위치 목록 가져오기
        
        Returns:
            list: 이동 가능한 위치 목록
        """
        current_location = self.player_info["location"]
        if current_location in self.game_env.locations:
            return self.game_env.locations[current_location]["connected_to"]
        return []
    
    def save_game_state(self, filename="game_state.json"):
        """
        게임 상태 저장
        
        Args:
            filename (str): 저장할 파일명
        """
        # 저장할 데이터 준비
        game_state = {
            "player": self.player_info,
            "environment": {
                "time_of_day": self.game_env.time_of_day,
                "weather": self.game_env.weather
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(game_state, f, indent=2)
            
        print(f"게임 상태가 {filename}에 저장되었습니다.")
    
    def load_game_state(self, filename="game_state.json"):
        """
        게임 상태 불러오기
        
        Args:
            filename (str): 불러올 파일명
            
        Returns:
            bool: 불러오기 성공 여부
        """
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    game_state = json.load(f)
                
                # 플레이어 정보 복원
                if "player" in game_state:
                    self.player_info = game_state["player"]
                
                # 환경 정보 복원
                if "environment" in game_state:
                    env = game_state["environment"]
                    if "time_of_day" in env:
                        self.game_env.update_time(env["time_of_day"])
                    if "weather" in env:
                        self.game_env.update_weather(env["weather"])
                
                print(f"게임 상태가 {filename}에서 불러와졌습니다.")
                return True
                
            except Exception as e:
                print(f"게임 상태 불러오기 오류: {e}")
                return False
        else:
            print(f"파일을 찾을 수 없습니다: {filename}")
            return False


# 대화 시뮬레이션 예제
def run_chat_simulation():
    """
    플레이어와 NPC 간 대화 시뮬레이션 실행
    """
    print("=== RPG Player-NPC 대화 시뮬레이션 ===")
    
    # 플레이어 이름 입력
    player_name = input("당신의 이름을 입력하세요: ")
    
    # 상호작용 시스템 초기화
    interaction = PlayerInteraction(player_name)
    
    print("\n게임 환경 초기화 중...")
    
    # NPC 생성
    print("\nNPC 생성 중...")
    
    # 상인 NPC 생성
    merchant = interaction.create_npc(
        "merchant_anna", 
        "Anna", 
        species="Human", 
        location="marketplace"
    )
    merchant.update_state({
        "emotion": "happy"
    })
    
    # 마법사 NPC 생성
    mage = interaction.create_npc(
        "mage_elindra",
        "Elindra",
        species="Elf",
        location="tavern"
    )
    mage.update_state({
        "emotion": "curious"
    })
    
    # 경비병 NPC 생성
    guard = interaction.create_npc(
        "guard_roland",
        "Roland",
        species="Human",
        location="northern_gate"
    )
    guard.update_state({
        "emotion": "alert"
    })
    
    # 플레이어 시작 위치 설정
    interaction.move_player("town_square")
    location_info = interaction.get_current_location_info()
    
    print(f"\n당신은 {location_info['description']}에 있습니다.")
    print(f"시간: {location_info['time_of_day']}, 날씨: {location_info['weather']}")
    print("주변 물건:", ", ".join(location_info['nearby_objects']))
    
    # 이동 가능한 위치 표시
    available_locations = interaction.get_available_locations()
    print("이동 가능한 장소:", ", ".join(available_locations))
    
    # 명령어 안내
    print("\n=== 명령어 안내 ===")
    print("이동: move <장소>")
    print("대화: talk <NPC ID>")
    print("위치 정보: look")
    print("종료: exit")
    
    # 메인 루프
    chatting_with = None  # 현재 대화 중인 NPC
    
    while True:
        try:
            if chatting_with:
                # 대화 모드
                prompt = f"{chatting_with.current_state.get('name', chatting_with.npc_id)}와 대화 중 (종료: /exit): "
                player_input = input(prompt)
                
                if player_input.lower() == "/exit":
                    print(f"{chatting_with.current_state.get('name', chatting_with.npc_id)}와의 대화를 종료합니다.")
                    chatting_with = None
                else:
                    # NPC와 대화
                    response = interaction.chat_with_npc(chatting_with.npc_id, player_input)
                    print(f"{chatting_with.current_state.get('name', chatting_with.npc_id)}: {response}")
            else:
                # 일반 명령 모드
                command = input("\n명령어 입력: ").strip()
                
                if command.lower() == "exit":
                    print("게임을 종료합니다.")
                    break
                    
                elif command.lower() == "look":
                    # 현재 위치 정보
                    location_info = interaction.get_current_location_info()
                    print(f"\n당신은 {location_info['description']}에 있습니다.")
                    print(f"시간: {location_info['time_of_day']}, 날씨: {location_info['weather']}")
                    print("주변 물건:", ", ".join(location_info['nearby_objects']))
                    
                    # 근처 NPC 표시
                    if location_info['nearby_npcs']:
                        print("주변 NPC:")
                        for npc in location_info['nearby_npcs']:
                            print(f"- {npc['name']} ({npc['species']})")
                    else:
                        print("주변에 NPC가 없습니다.")
                    
                    # 이동 가능한 위치 표시
                    available_locations = interaction.get_available_locations()
                    print("이동 가능한 장소:", ", ".join(available_locations))
                    
                elif command.lower().startswith("move "):
                    # 다른 위치로 이동
                    location = command[5:].strip()
                    available_locations = interaction.get_available_locations()
                    
                    if location in available_locations:
                        location_info = interaction.move_player(location)
                        print(f"\n당신은 {location_info['description']}(으)로 이동했습니다.")
                        
                        # 이동 후 위치 정보 표시
                        print(f"시간: {location_info['time_of_day']}, 날씨: {location_info['weather']}")
                        print("주변 물건:", ", ".join(location_info['nearby_objects']))
                        
                        # 근처 NPC 표시
                        if location_info['nearby_npcs']:
                            print("주변 NPC:")
                            for npc in location_info['nearby_npcs']:
                                print(f"- {npc['name']} ({npc['species']})")
                        else:
                            print("주변에 NPC가 없습니다.")
                        
                        # 이동 가능한 위치 표시
                        available_locations = interaction.get_available_locations()
                        print("이동 가능한 장소:", ", ".join(available_locations))
                    else:
                        print(f"'{location}'(으)로 이동할 수 없습니다.")
                        print("이동 가능한 장소:", ", ".join(available_locations))
                        
                elif command.lower().startswith("talk "):
                    # NPC와 대화 시작
                    npc_id = command[5:].strip()
                    
                    if npc_id in interaction.npcs:
                        chatting_with = interaction.npcs[npc_id]
                        
                        # 대화 시작 메시지
                        npc_name = chatting_with.current_state.get("name", npc_id)
                        print(f"\n{npc_name}와 대화를 시작합니다. (대화 종료: /exit)")
                        
                        # NPC가 다른 위치에 있는 경우
                        if chatting_with.current_state.get('location') != interaction.player_info['location']:
                            print(f"(참고: {npc_name}는 현재 {chatting_with.current_state.get('location')}에 있어 원격으로 대화하고 있습니다.)")
                    else:
                        print(f"'{npc_id}' ID를 가진 NPC를 찾을 수 없습니다.")
                        
                        # 현재 위치의 NPC 목록
                        location_info = interaction.get_current_location_info()
                        if location_info['nearby_npcs']:
                            print("주변 NPC:")
                            for npc in location_info['nearby_npcs']:
                                print(f"- {npc['name']} (ID: {npc['id']})")
                else:
                    print("알 수 없는 명령어입니다.")
                    print("사용 가능한 명령어: move <장소>, talk <NPC ID>, look, exit")
                    
        except KeyboardInterrupt:
            print("\n게임을 종료합니다.")
            break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    run_chat_simulation() 