import time
import json
import os
from transformers import AutoTokenizer, pipeline
import importlib.util
import pickle
from datetime import datetime
import random
from huggingface_hub import login

# PyTorch 설치 확인
pytorch_installed = importlib.util.find_spec("torch") is not None

if pytorch_installed:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    
    # CUDA 가용성 확인
    cuda_available = torch.cuda.is_available()
    
    # GPU 메모리 정리
    if cuda_available:
        torch.cuda.empty_cache()
        print(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"가용 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
else:
    print("경고: PyTorch가 설치되어 있지 않습니다. AI 모델 기능이 제한됩니다.")
    print("PyTorch 설치 방법: https://pytorch.org/get-started/locally/")

# Hugging Face 토큰으로 로그인
login(token="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")

# model_config 모듈 가져오기 시도
try:
    from model_config import model_config
    print(f"모델 설정 로드: {model_config}")
    use_model_config = True
except ImportError as e:
    print(f"model_config 모듈을 가져오는 데 문제가 발생했습니다: {e}")
    use_model_config = False
except Exception as e:
    print(f"model_config 로드 중 오류: {e}")
    print("기본 모델 설정을 사용합니다.")
    use_model_config = False

# 기존 import 시도
try:
    from RPG_AI_System import NPCBrain, generate_ai_character_data, GameEnvironment
except ImportError as e:
    print(f"RPG_AI_System 모듈을 가져오는 데 문제가 발생했습니다: {e}")
    # 임시 대체 클래스 정의
    class GameEnvironment:
        def __init__(self):
            self.locations = {
                "town_square": {
                    "description": "마을 광장",
                    "connected_to": ["marketplace", "tavern", "northern_gate"],
                    "nearby_objects": ["분수대", "벤치", "게시판"],
                    "nearby_npcs": []
                }
            }
            self.time_of_day = "낮"
            self.weather = "맑음"
        
        def add_npc(self, npc):
            pass
            
        def get_environment_data(self, location):
            if location in self.locations:
                data = self.locations[location].copy()
                data["time_of_day"] = self.time_of_day
                data["weather"] = self.weather
                return data
            return {"error": f"위치를 찾을 수 없습니다: {location}"}
            
        def update_time(self, new_time):
            self.time_of_day = new_time
            
        def update_weather(self, new_weather):
            self.weather = new_weather
    
    class NPCBrain:
        def __init__(self, npc_id, npc_data, model_id=None):
            self.npc_id = npc_id
            self.npc_data = npc_data
            self.current_state = {"name": npc_id}
            self.memory = DummyMemory()
            
        def update_state(self, state_updates):
            self.current_state.update(state_updates)
            
        def _format_personality_for_prompt(self):
            return "Personality: 임시 성격 데이터"
            
        def _format_memories_for_prompt(self, memories):
            return "Memories: 임시 기억 데이터"
            
        def pipe(self, prompt, max_new_tokens=200, temperature=0.8):
            return [{"generated_text": "안녕하세요! 제가 답변해 드리겠습니다."}]
            
        def _extract_conversation(self, text):
            return "죄송합니다, PyTorch가 설치되어 있지 않아 AI 응답을 생성할 수 없습니다."
    
    class DummyMemory:
        def add_memory(self, memory):
            pass
            
        def retrieve_relevant_memories(self, query):
            return []
    
    def generate_ai_character_data():
        return {
            "instincts": {
                "species": "Human",
                "goals": ["살아남기", "친구 만들기"]
            }
        }

class PlayerInteraction:
    """
    플레이어와 NPC 간 대화 시뮬레이션을 위한 클래스
    """
    def __init__(self, player_name="모험가"):
        """PlayerInteraction 클래스 초기화"""
        self.npcs = {}
        self.player_info = {
            "name": player_name,
            "location": "town_square",
            "inventory": [],
            "quests": [],
            "conversation_history": []
        }
        
        # 모델 설정 저장 - 이 부분을 추가해야 합니다
        self.model_id = None
        self.use_cpu = False
        self.use_4bit = False
        self.use_config_model = False  # 추가된 속성
        
        # 게임 환경 초기화
        self.game_env = GameEnvironment()
    
    @property
    def player_name(self):
        """플레이어 이름 속성 (편의를 위한 프로퍼티)"""
        return self.player_info.get("name", "모험가")

    @player_name.setter
    def player_name(self, value):
        """플레이어 이름 설정 프로퍼티"""
        self.player_info["name"] = value
    
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
        try:
            # GPU 메모리 정리
            if pytorch_installed and torch.cuda.is_available() and not self.use_cpu:
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"GPU 메모리 정리 중 오류 발생: {e}")
            
            # NPC 데이터 생성 또는 사용자 정의 데이터 사용
            if custom_data:
                npc_data = custom_data
            else:
                npc_data = generate_ai_character_data()
            
            # 종족과 위치 설정
            if species:
                npc_data["instincts"]["species"] = species
            
            # 모델 타입 설정
            model_type = 'gemma'
            if not self.use_config_model and self.model_id:
                if '/' in self.model_id:
                    model_type = self.model_id.split('/')[-1]
                else:
                    model_type = self.model_id
            
            # 디버깅 정보 출력
            print(f"[DEBUG] {npc_id} 생성 중, 사용할 모델 타입: {model_type}")
            
            # NPC 객체 생성 - 새로운 초기화 방식 적용
            npc = NPCBrain(
                npc_id=npc_id, 
                npc_data=npc_data,
                model_type=model_type,
                use_cpu_mode=self.use_cpu,
                quantization='4bit' if self.use_4bit else None,
                name=npc_name,
                location=location or "town_square"  # 기본 위치는 마을 광장
            )
        
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
        except Exception as e:
            print(f"NPC 생성 중 오류 발생: {e}")
            print("GPU 메모리 부족 오류가 발생했습니다. 다음 해결 방법을 시도해보세요:")
            print("1. CPU 모드로 실행하기")
            print("2. 메모리 최적화 사용하기")
            print("3. 더 작은 모델 선택하기")
            
            # 디버그용 간단한 NPC 객체 생성 - 새로운 초기화 방식 적용
            npc = NPCBrain(
                npc_id=npc_id, 
                npc_data={"instincts": {"species": species or "Human"}},
                name=npc_name, 
                location=location
            )
            self.npcs[npc_id] = npc
            return npc
    
    def chat_with_npc(self, npc_id, player_input):
        """
        플레이어의 입력을 처리하여 NPC와 대화 진행
        """
        try:
            # 플레이어 컨텍스트 생성
            player_context = self._create_player_context()
            
            # 대화 주제 추론
            topic = self._infer_conversation_topic(self.npcs[npc_id], player_input)
            
            # NPC에게 메시지 전달하고 응답 받기 (pipe 대신 chat 메소드 사용)
            response = self.npcs[npc_id].chat(player_input, player_context)
            
            # 대화 기록 저장
            self.player_info["conversation_history"].append({
                "timestamp": time.time(),
                "npc_id": npc_id,
                "player_input": player_input,
                "npc_response": response
            })
            
            # 자동 저장 추가
            self.save_npc(npc_id)
            
            return response
        except Exception as e:
            print(f"NPC 대화 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return "죄송합니다. 대화 처리 중 오류가 발생했습니다."
    
    def _create_player_context(self):
        """플레이어 컨텍스트 생성"""
        # reputation 키가 없는 경우 기본값 설정
        if "reputation" not in self.player_info:
            self.player_info["reputation"] = {"general": 0.5}
        
        # location 키가 없는 경우 기본값 설정
        if "location" not in self.player_info:
            self.player_info["location"] = "town_square"
            
        return {
            "name": self.player_info["name"],
            "location": self.player_info["location"],
            "reputation": self.player_info.get("reputation", {}).get("general", 0.5)
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
    
    def _extract_keywords(self, text):
        """텍스트에서 주요 키워드 추출"""
        # 불용어 목록
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                    "being", "to", "of", "and", "or", "but", "in", "on", "at", 
                    "by", "for", "with", "about", "against", "between", "into", 
                    "through", "during", "before", "after", "above", "below", 
                    "from", "up", "down", "out", "off", "over", "under", "again", 
                    "further", "then", "once", "here", "there", "when", "where", 
                    "why", "how", "all", "any", "both", "each", "few", "more", 
                    "most", "other", "some", "such", "no", "nor", "not", "only", 
                    "own", "same", "so", "than", "too", "very", "s", "t", "can", 
                    "will", "just", "don", "should", "now", "이", "그", "저", "것", 
                    "수", "등", "및", "에", "를", "은", "는", "이다", "있다", "하다", 
                    "그리고", "그러나", "또는", "그래서", "그러면", "고"}
        
        # 텍스트 정규화 및 토큰화
        words = text.lower().replace(".", " ").replace(",", " ").replace("?", " ").replace("!", " ").split()
        
        # 불용어 제거 및 길이 2 이상인 단어만 선택
        keywords = [word for word in words if word not in stopwords and len(word) > 1]
        
        # 중복 제거하고 최대 10개 키워드 반환
        return list(set(keywords))[:10]

    def _infer_conversation_topic(self, npc, player_message):
        """대화 주제 추론"""
        # 주제 후보 (한국어 + 영어)
        topics = {
            "인사": ["안녕", "반가", "처음", "만나", "hello", "hi", "greet", "meet"],
            "거래": ["살", "팔", "가격", "돈", "거래", "물건", "buy", "sell", "price", "money", "trade", "item"],
            "정보": ["어디", "무엇", "누구", "어떻게", "정보", "알려", "where", "what", "who", "how", "info", "tell"],
            "요청": ["도와", "부탁", "해줘", "해주세요", "필요", "help", "please", "need", "request", "favor"],
            "위협": ["공격", "죽", "해치", "위험", "attack", "kill", "hurt", "danger", "threat"],
            "칭찬": ["감사", "고마", "멋지", "좋아", "잘했", "thanks", "thank", "good", "great", "nice"],
            "일상": ["날씨", "기분", "하루", "오늘", "weather", "feel", "day", "today"]
        }
        
        # 메시지에서 주제 키워드 검색
        message_lower = player_message.lower()
        detected_topics = []
        
        for topic, keywords in topics.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_topics.append(topic)
        
        # 주제가 감지되지 않으면 최근 대화 주제 또는 일반 대화 반환
        if not detected_topics:
            if hasattr(npc, 'conversation_context') and npc.conversation_context.get("last_topics"):
                return npc.conversation_context["last_topics"][0]  # 최근 주제 유지
            return "일반 대화"
        
        # 감지된 주제 중 첫 번째 반환 (여러 주제가 있을 경우)
        topic = detected_topics[0]
        
        # 대화 주제 기록 업데이트
        if hasattr(npc, 'conversation_context'):
            npc.conversation_context["last_topics"] = [topic] + npc.conversation_context.get("last_topics", [])[:2]
        
        return topic

    def _update_relationship_with_player(self, npc, player_message, npc_response):
        """플레이어와 NPC 간의 관계 업데이트"""
        # 현재 관계 상태 가져오기
        relationship = npc.current_state.get("relationship_with_player", "neutral")
        reputation = npc.current_state.get("player_reputation", 50)  # 0-100 범위
        
        # 긍정/부정 단어 목록
        positive_words = ["고마", "감사", "좋아", "멋져", "훌륭", "도와", "친구", "thanks", "thank", "good", 
                         "great", "help", "friend", "appreciate", "kind"]
        negative_words = ["미워", "싫어", "못된", "나쁜", "짜증", "화나", "공격", "거짓", "hate", "dislike", 
                         "bad", "annoying", "angry", "attack", "lie"]
        
        # 대화 분석
        combined_text = (player_message + " " + npc_response).lower()
        
        # 긍정/부정 점수 계산
        positive_score = sum(1 for word in positive_words if word in combined_text)
        negative_score = sum(1 for word in negative_words if word in combined_text)
        
        # 점수 차이에 따른 평판 조정
        reputation_change = (positive_score - negative_score) * 2
        
        # 관계 변화가 있는 경우만 처리
        if reputation_change != 0:
            new_reputation = max(0, min(100, reputation + reputation_change))
            
            # 평판에 따른 관계 상태 결정
            if new_reputation >= 80:
                new_relationship = "friendly"
            elif new_reputation >= 60:
                new_relationship = "positive"
            elif new_reputation >= 40:
                new_relationship = "neutral"
            elif new_reputation >= 20:
                new_relationship = "negative"
            else:
                new_relationship = "hostile"
            
            # 관계가 변했을 때만 업데이트
            if new_reputation != reputation or new_relationship != relationship:
                npc.current_state["player_reputation"] = new_reputation
                
                if new_relationship != relationship:
                    npc.current_state["relationship_with_player"] = new_relationship
                    
                    # 관계 변화를 메모리에 기록
                    npc.memory.add_memory({
                        "type": "relationship_change",
                        "from": relationship,
                        "to": new_relationship,
                        "importance": 0.8,
                        "timestamp": time.time(),
                        "content": f"플레이어와의 관계가 {relationship}에서 {new_relationship}로 변화했습니다."
                    })
                    
                    return True
        
        return False

    def _update_npc_state_over_time(self, npc):
        """시간 경과에 따른 NPC 상태 업데이트"""
        current_time = time.time()
        last_interaction = npc.current_state.get("last_interaction_time", current_time)
        time_diff = current_time - last_interaction
        
        # 30분 이상 경과했으면 상태 변화
        if time_diff > 30 * 60:  # 30분
            # 감정 상태 약화 (중립으로 회귀)
            if npc.current_state.get("emotion") != "neutral":
                current_strength = npc.current_state.get("emotion_strength", 0.5)
                # 시간에 따른 감정 약화
                reduced_strength = max(0.1, current_strength - (time_diff / (60 * 60 * 24)))
                
                if reduced_strength < 0.3:  # 임계값 이하면 중립으로
                    npc.current_state["emotion"] = "neutral"
                    npc.current_state["emotion_strength"] = 0.1
                else:
                    npc.current_state["emotion_strength"] = reduced_strength
            
            # 에너지 및 체력 회복
            if "energy" in npc.current_state and npc.current_state["energy"] < 100:
                recovery_rate = time_diff / (60 * 60 * 2)  # 2시간당 100% 회복
                npc.current_state["energy"] = min(100, npc.current_state["energy"] + recovery_rate)
            
            if "health" in npc.current_state and npc.current_state["health"] < 100:
                recovery_rate = time_diff / (60 * 60 * 12)  # 12시간당 100% 회복
                npc.current_state["health"] = min(100, npc.current_state["health"] + recovery_rate)
            
            # 상호작용 시간 업데이트
            npc.current_state["last_interaction_time"] = current_time
    
    def _generate_npc_response_to_player(self, npc, player_message, player_context, topic):
        """NPC 응답 생성 - 개선된 버전"""
        # 메모리 컨텍스트 가져오기
        memories = npc.memory.retrieve_relevant_memories(player_message, k=7)
        memory_context = npc._format_memories_for_prompt(memories)
        
        # 성격 정보 가져오기
        personality = npc._format_personality_for_prompt()
        
        # 최근 대화 컨텍스트 추적
        player_keywords = self._extract_keywords(player_message)
        if hasattr(npc, 'conversation_context'):
            for keyword in player_keywords:
                npc.conversation_context["player_interests"].add(keyword)
        
        # 질문 추적 (다음 응답에 반영)
        is_question = any(q in player_message for q in ["?", "뭐", "어떻", "언제", "누구", "왜", "where", "what", "how", "when", "who", "why"])
        if is_question and len(player_message) > 3:  # 짧은 메시지는 제외
            npc.conversation_context["unresolved_questions"].append(player_message)
            if len(npc.conversation_context["unresolved_questions"]) > 3:
                npc.conversation_context["unresolved_questions"].pop(0)
        
        # 프롬프트 개선
        prompt = f"""<|system|>
당신은 '{npc.current_state.get('name', npc.npc_id)}'(이)라는 이름의 판타지 RPG 세계의 NPC입니다.
당신은 플레이어가 아니라 게임 속 캐릭터이며, 항상 '{npc.current_state.get('name', npc.npc_id)}' 역할을 연기해야 합니다.

기본 정보:
- 이름: {npc.current_state.get('name', npc.npc_id)}
- 종족: {npc.npc_data['instincts'].get('species', 'Human')}
- 직업: {npc.npc_data['instincts'].get('occupation', '알 수 없음')}

{personality}

현재 상태:
- 위치: {npc.current_state.get('location', 'unknown')}
- 시간: {npc.current_state.get('time_of_day', 'daytime')}
- 감정: {npc.current_state.get('emotion', 'neutral')} (강도: {npc.current_state.get('emotion_strength', 0.5)})
- 체력: {npc.current_state.get('health', 100)}/100
- 에너지: {npc.current_state.get('energy', 100)}/100

당신은 '{self.player_info['name']}'(이)라는 이름의 플레이어와 대화 중입니다.
플레이어와의 관계: {npc.current_state.get('relationship_with_player', '중립적')}
{player_context}

{memory_context}

대화 지침:
1. 항상 '{npc.current_state.get('name', npc.npc_id)}'로서 말하고 행동하세요.
2. 간결하고 자연스럽게 응답하세요.
3. 현재 감정 상태({npc.current_state.get('emotion', 'neutral')})를 반영하여 대화하세요.
4. 플레이어의 질문에 집중하여 답변하세요.
5. 대화가 자연스럽게 이어지도록 하세요.
6. 절대 플레이어의 역할을 맡지 마세요.

대화 주제: {topic}
"""

        # 플레이어의 관심사 추가 (있을 경우)
        if hasattr(npc, 'conversation_context') and npc.conversation_context.get("player_interests"):
            player_interests = list(npc.conversation_context["player_interests"])[:5]
            if player_interests:
                prompt += f"플레이어의 관심사: {', '.join(player_interests)}\n"

        prompt += f"""
<|user|>
{player_message}
<|assistant|>"""

        # 조건부 도움말 프롬프트 (AI가 대화를 이어가기 어려울 때)
        if len(player_message) < 5:  # 매우 짧은 메시지
            prompt += "\n(짧은 질문이지만 최대한 자연스럽게 대화를 이어가세요. 간단해도 당신의 성격을 드러내는 답변을 해주세요.)"
        
        if player_message.lower() in ["hi", "hello", "안녕", "ㅎㅇ"]:  # 인사
            prompt += f"\n(첫 인사에는 자신을 소개하고 {npc.current_state.get('name', npc.npc_id)}로서 플레이어를 반겨주세요.)"

        # AI 응답 생성
        try:
            response = npc.pipe(prompt, max_new_tokens=200, temperature=0.7, do_sample=True)
            
            # 응답 처리
            if isinstance(response, list):
                generated_text = response[0]["generated_text"]
            elif isinstance(response, dict):
                generated_text = response.get("generated_text", "죄송합니다, 응답을 생성할 수 없습니다.")
            elif isinstance(response, str):
                generated_text = response
            else:
                generated_text = "죄송합니다, 응답 형식을 처리할 수 없습니다."
            
            # 실제 응답 부분만 추출
            if "<|assistant|>" in generated_text:
                clean_response = generated_text.split("<|assistant|>")[1].strip()
            else:
                clean_response = generated_text.replace(prompt, "").strip()
            
            # 대화 기록을 NPC 메모리에 추가
            npc.memory.add_memory({
                "type": "conversation",
                "content": f"Player: {player_message}\nNPC: {clean_response}",
                "importance": 0.7,
                "keywords": self._extract_keywords(player_message + " " + clean_response),
                "with_player": True,
                "timestamp": time.time()
            })
            
            return clean_response
            
        except Exception as e:
            print(f"응답 생성 중 오류: {e}")
            return "죄송합니다, 대화를 이해하는 데 문제가 있습니다."

    def _format_npc_response_with_emotion(self, npc, response):
        """NPC 응답을 감정 상태에 맞게 포맷팅"""
        # 감정별 이모티콘
        emotion_icons = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "afraid": "😨",
            "surprised": "😲",
            "disgusted": "😖",
            "curious": "🤔",
            "confused": "😕",
            "tired": "😴",
            "relaxed": "😌",
            "neutral": "😐"
        }
        
        # 현재 감정과 강도
        emotion = npc.current_state.get("emotion", "neutral")
        strength = npc.current_state.get("emotion_strength", 0.5)
        
        # 감정 강도가 낮으면 중립적 아이콘 사용
        if strength < 0.3:
            emotion = "neutral"
        
        # 이모티콘 선택
        icon = emotion_icons.get(emotion, "😐")
        
        # 이모티콘을 응답 앞에 추가 (강도에 따라 중복)
        if strength > 0.7:
            return f"{icon} {icon} {response}"
        elif strength > 0.4:
            return f"{icon} {response}"
        else:
            return response
    
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
                    "species": npc.npc_data['instincts']['species']
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

    def save_npc(self, npc_id):
        """
        NPC 상태 저장하기
        
        Args:
            npc_id (str): 저장할 NPC의 ID
        
        Returns:
            bool: 저장 성공 여부
        """
        if npc_id not in self.npcs:
            print(f"오류: {npc_id} ID를 가진 NPC를 찾을 수 없습니다.")
            return False
        
        # NPC 저장 디렉토리 생성
        save_dir = "saved_npcs"
        os.makedirs(save_dir, exist_ok=True)
        
        # 저장 파일 경로
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"{npc_id}_{timestamp}.pkl")
        
        try:
            # NPC 객체 저장
            npc = self.npcs[npc_id]
            
            # 저장할 데이터 준비
            save_data = {
                "npc_id": npc_id,
                "timestamp": timestamp,
                "current_state": npc.current_state,
                "npc_data": npc.npc_data,
                "memory": npc.memory,
                "metadata": {
                    "last_location": npc.current_state.get("location", "unknown"),
                    "last_interaction": datetime.now().isoformat()
                }
            }
            
            # 파일에 저장
            with open(save_path, "wb") as f:
                pickle.dump(save_data, f)
            
            print(f"\n{npc.current_state.get('name', npc_id)}의 상태가 저장되었습니다.")
            return True
        
        except Exception as e:
            print(f"NPC 저장 중 오류 발생: {e}")
            return False

    def load_all_saved_npcs(self):
        """모든 저장된 NPC 정보 로드 - 개선된 버전"""
        print("저장된 NPC 찾는 중...")
        save_dir = "saved_npcs"
        saved_npcs = {}
        
        # 저장 디렉토리가 없으면 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"{save_dir} 디렉토리 생성됨")
            return saved_npcs
        
        # 저장된 모든 .pkl 파일 탐색
        for file in os.listdir(save_dir):
            if file.endswith(".pkl"):
                try:
                    file_path = os.path.join(save_dir, file)
                    # 파일명에서 NPC ID 추출 (예: merchant_anna.pkl -> merchant_anna)
                    npc_id = file.split(".")[0]
                    saved_npcs[npc_id] = {
                        "file_path": file_path,
                        "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M")
                    }
                except Exception as e:
                    print(f"파일 '{file}' 처리 중 오류: {e}")
        
        print(f"발견된 NPC: {len(saved_npcs)}")
        return saved_npcs

    def load_npc(self, npc_id=None, file_path=None):
        """저장된 NPC 로드 - 수정된 버전"""
        if not npc_id and not file_path:
            print("오류: NPC ID 또는 파일 경로를 제공해야 합니다.")
            return None
        
        try:
            load_path = None
            
            # 파일 경로로 로드
            if file_path and os.path.exists(file_path):
                load_path = file_path
            # NPC ID로 로드
            elif npc_id:
                save_dir = "saved_npcs"
                # 정확한 파일명 확인
                potential_file = os.path.join(save_dir, f"{npc_id}.pkl")
                if os.path.exists(potential_file):
                    load_path = potential_file
                else:
                    # 모든 파일 검색
                    for file in os.listdir(save_dir):
                        if file.startswith(f"{npc_id}_") or file == f"{npc_id}.pkl":
                            load_path = os.path.join(save_dir, file)
                            break
            
            if load_path:
                with open(load_path, "rb") as f:
                    try:
                        npc = pickle.load(f)
                        self.npcs[npc_id] = npc
                        print(f"{npc.current_state.get('name', npc.npc_id)} NPC를 불러왔습니다.")
                        return npc
                    except Exception as e:
                        print(f"NPC 객체 로드 중 오류: {e}")
                        return None
            else:
                print(f"오류: {npc_id}에 해당하는 NPC 파일을 찾을 수 없습니다.")
                return None
            
        except Exception as e:
            print(f"NPC 로드 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

    def list_saved_npcs(self):
        """
        저장된 모든 NPC 파일 목록 표시
        
        Returns:
            list: 저장된 NPC 정보 목록
        """
        save_dir = "saved_npcs"
        
        if not os.path.exists(save_dir):
            print("저장된 NPC가 없습니다.")
            return []
        
        saved_files = os.listdir(save_dir)
        if not saved_files:
            print("저장된 NPC가 없습니다.")
            return []
        
        # NPC 정보 수집
        npcs_info = []
        for file_name in saved_files:
            if not file_name.endswith(".pkl"):
                continue
            
            try:
                # 파일에서 NPC ID와 저장 시간 추출
                parts = file_name.split("_")
                npc_id = parts[0]
                timestamp = "_".join(parts[1:]).replace(".pkl", "")
                
                # 파일 내용 읽기 시도
                file_path = os.path.join(save_dir, file_name)
                with open(file_path, "rb") as f:
                    save_data = pickle.load(f)
                
                # NPC 정보 수집
                npc_name = save_data["current_state"].get("name", npc_id)
                location = save_data["current_state"].get("location", "알 수 없음")
                emotion = save_data["current_state"].get("emotion", "neutral")
                
                npcs_info.append({
                    "file_name": file_name,
                    "file_path": file_path,
                    "npc_id": npc_id,
                    "npc_name": npc_name,
                    "timestamp": timestamp,
                    "location": location,
                    "emotion": emotion
                })
                
            except Exception as e:
                print(f"파일 {file_name} 처리 중 오류: {e}")
        
        # 결과 표시
        if npcs_info:
            print("\n=== 저장된 NPC 목록 ===")
            for i, info in enumerate(npcs_info, 1):
                save_time = datetime.strptime(info["timestamp"], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                print(f"{i}. {info['npc_name']} (ID: {info['npc_id']}) - 저장일시: {save_time}")
                print(f"   위치: {info['location']}, 감정: {info['emotion']}")
        
        return npcs_info

    def auto_save_npc(self, npc_id):
        """NPC 상태 자동 저장"""
        if npc_id not in self.npcs:
            return False
        
        npc = self.npcs[npc_id]
        
        # 마지막 저장 시간 확인
        current_time = time.time()
        last_save_time = npc.current_state.get("last_save_time", 0)
        
        # 최소 1분 간격으로 저장 (너무 자주 저장하지 않도록)
        if current_time - last_save_time < 60:  # 60초
            return False
        
        # 저장 디렉토리 확인
        save_dir = "saved_npcs"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 저장 파일명 (NPC ID + 타임스탬프)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{npc_id}_{timestamp}.pkl"
        file_path = os.path.join(save_dir, file_name)
        
        # 저장 시간 업데이트
        npc.current_state["last_save_time"] = current_time
        
        try:
            # pickle로 NPC 객체 저장
            with open(file_path, "wb") as f:
                pickle.dump(npc, f)
            
            # 오래된 백업 파일 정리 (최대 5개 유지)
            self._cleanup_old_saves(npc_id)
            
            return True
        except Exception as e:
            print(f"NPC 저장 중 오류: {e}")
            return False

    def _cleanup_old_saves(self, npc_id, max_saves=5):
        """NPC별 오래된 저장 파일 정리"""
        save_dir = "saved_npcs"
        if not os.path.exists(save_dir):
            return
        
        # NPC ID로 시작하는 모든 파일 검색
        npc_files = [f for f in os.listdir(save_dir) if f.startswith(f"{npc_id}_") and f.endswith(".pkl")]
        
        # 파일이 max_saves보다 많으면 오래된 것부터 삭제
        if len(npc_files) > max_saves:
            # 생성일 기준 정렬
            npc_files.sort()
            # 가장 오래된 파일부터 삭제
            for file_to_delete in npc_files[:-max_saves]:
                try:
                    os.remove(os.path.join(save_dir, file_to_delete))
                except:
                    pass

    def create_new_npc(self, npc_id=None):
        """새로운 NPC 생성 - 오류 처리 개선"""
        print("\n새 NPC 생성 중...")
        try:
            if npc_id is None:
                # 기본 NPC 유형
                npc_types = {
                    "1": "merchant",
                    "2": "guard", 
                    "3": "mage",
                    "4": "innkeeper",
                    "5": "blacksmith"
                }
                
                print("\nNPC 유형 선택:")
                for key, npc_type in npc_types.items():
                    print(f"{key}: {npc_type}")
                    
                choice = input("유형 번호 선택 (기본: 1): ")
                npc_type = npc_types.get(choice, "merchant")
                
                # 랜덤 이름 생성
                name = input(f"{npc_type}의 이름 입력 (Enter 시 자동 생성): ")
                if not name:
                    names = {
                        "merchant": ["Anna", "Brom", "Clara", "Dorn"],
                        "guard": ["Roland", "Sigmund", "Talia", "Vance"],
                        "mage": ["Elindra", "Firion", "Gwendolyn", "Hector"],
                        "innkeeper": ["Isolde", "Jareth", "Katarina", "Lionel"],
                        "blacksmith": ["Morn", "Norissa", "Osric", "Phaedra"]
                    }
                    import random
                    name = random.choice(names.get(npc_type, ["Unknown"]))
                    print(f"이름이 자동 생성되었습니다: {name}")
                    
                npc_id = f"{npc_type}_{name.lower()}"
            
            # 기본 NPC 데이터 생성
            npc_data = generate_ai_character_data()
            npc_data["name"] = name
            npc_data["type"] = npc_type
            npc_data["location"] = "town_square"  # 기본 위치
            
            # 직업 정보 추가 (initialize_npc에서 사용됨)
            if "instincts" in npc_data:
                npc_data["instincts"]["occupation"] = npc_type
            else:
                npc_data["instincts"] = {"occupation": npc_type, "species": "Human"}
            
            print(f"\n{name} NPC 생성 중...")
            
            # NPCBrain 인스턴스 생성 시 예외 처리 강화
            try:
                # 변경된 NPCBrain.__init__ 메서드에 맞게 키워드 인자로 전달
                npc = NPCBrain(
                    npc_id=npc_id, 
                    npc_data=npc_data,
                    model_type=self.model_id.split('/')[-1] if '/' in self.model_id else 'gemma',
                    use_cpu_mode=self.use_cpu,
                    quantization='4bit' if self.use_4bit else None,
                    name=name,
                    location="town_square"  # 기본 위치
                )
                self.npcs[npc_id] = npc
            
                # 초기 저장
                self.save_npc(npc_id)
            
                print(f"{name} NPC가 성공적으로 생성되었습니다!")
                return npc_id
            except Exception as e:
                print(f"NPC 생성 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                return None
            
        except Exception as e:
            print(f"NPC 생성 준비 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

def select_model():
    """사용할 AI 모델 선택"""
    print("\n=== 모델 설정 ===")
    model_id = None
    
    try:
        # model_config.py에서 MODEL_CONFIGS 가져오기
        from model_config import MODEL_CONFIGS, ModelType
        
        # 사용 가능한 모델 타입 목록 표시
        print("\n사용 가능한 모델 타입:")
        model_types = {}
        for i, model_type in enumerate(ModelType, 1):
            model_types[str(i)] = model_type
            print(f"{i}: {model_type.value}")
        
        # 모델 타입 선택
        type_choice = input("\n모델 타입 선택 (기본: 6-gemma): ")
        selected_type = model_types.get(type_choice, ModelType.GEMMA)
        
        print(f"\n{selected_type.value} 모델 크기 선택:")
        sizes = {}
        # 해당 모델 타입의 사용 가능한 크기 표시
        if selected_type in MODEL_CONFIGS:
            for i, size in enumerate(MODEL_CONFIGS[selected_type].keys(), 1):
                sizes[str(i)] = size
                
                # 모델 설명 가져오기
                model_info = MODEL_CONFIGS[selected_type][size]
                if isinstance(model_info, dict) and "model_id" in model_info:
                    model_name = model_info["model_id"]
                    desc = model_info.get("recommended_for", ["기본 모델"])
                    desc = ", ".join(desc[:2]) if desc else "기본 모델"
                else:
                    model_name = model_info
                    desc = "기본 모델"
                
                print(f"{i}: {size} - {model_name} ({desc})")
            
            # 크기 선택
            size_choice = input("\n모델 크기 선택 (기본: 1): ")
            selected_size = sizes.get(size_choice, list(MODEL_CONFIGS[selected_type].keys())[0])
            
            # 모델 ID 가져오기
            model_config = MODEL_CONFIGS[selected_type][selected_size]
            if isinstance(model_config, dict) and "model_id" in model_config:
                model_id = model_config["model_id"]
            else:
                model_id = model_config
                
            print(f"\n선택된 모델: {model_id}")
        else:
            print(f"오류: {selected_type.value} 모델 타입에 대한 설정이 없습니다.")
            model_id = "google/gemma-3-1b-it"  # 기본값
    except Exception as e:
        print(f"모델 설정 로드 중 오류 발생: {e}")
        print("기본 모델을 사용합니다.")
        
        # 기본 모델 목록 제공
        model_options = {
            "1": {"id": "google/gemma-3-1b-it", "name": "Gemma 3 1B", "desc": "빠르고 가벼운 모델 (2GB 메모리)"},
            "2": {"id": "google/gemma-3-4b-it", "name": "Gemma 3 4B", "desc": "균형 잡힌 성능 (4GB 메모리)"},
            "3": {"id": "meta-llama/Llama-3.2-1B-Instruct", "name": "Llama 3.2 1B", "desc": "빠른 응답 (2GB 메모리)"},
            "4": {"id": "meta-llama/Llama-3.2-3B-Instruct", "name": "Llama 3.2 3B", "desc": "향상된 품질 (3GB 메모리)"}
        }
        
        print("\n사용 가능한 모델:")
        print(f"{'번호':<4}{'모델명':<15}{'설명':<30}")
        print("-" * 50)
        for key, model in model_options.items():
            print(f"{key:<4}{model['name']:<15}{model['desc']:<30}")
        
        model_choice = input("\n모델 번호를 선택하세요 (기본: 1): ")
        model_data = model_options.get(model_choice, model_options["1"])
        model_id = model_data["id"]
        print(f"선택된 모델: {model_data['name']} ({model_id})")
    
    return model_id

# 대화 시뮬레이션 예제
def run_chat_simulation():
    """
    플레이어와 NPC 간 대화 시뮬레이션 실행
    """
    print("=== RPG Player-NPC 대화 시뮬레이션 ===")
    
    # 플레이어 이름 입력
    player_name = input("당신의 이름을 입력하세요: ")
    if not player_name:
        player_name = "모험가"
        print(f"이름이 입력되지 않아 '{player_name}'(으)로 지정됩니다.")
    
    # 하드웨어 설정
    print("\n=== 하드웨어 설정 ===")
    
    # CPU 모드 선택
    use_cpu = False
    if pytorch_installed and torch.cuda.is_available():
        print(f"감지된 GPU: {torch.cuda.get_device_name(0)}")
        use_cpu_input = input("CPU 모드로 실행하시겠습니까? (y/n, 기본: n): ").lower()
        use_cpu = use_cpu_input.startswith('y')
    else:
        use_cpu = True
        print("GPU가 감지되지 않아 CPU 모드로 실행됩니다.")
        
    # 4비트 양자화 선택 (GPU를 사용하는 경우만)
    use_4bit = False
    if not use_cpu:
        use_4bit_input = input("4비트 양자화를 사용하시겠습니까? (적은 메모리 사용) (y/n, 기본: y): ").lower()
        use_4bit = not use_4bit_input.startswith('n')  # 기본값 True
    
    # 모델 선택
    model_id = select_model()
    
    try:
        # NPC 데이터 로드
        print("\n=== RPG NPC 대화 시스템 시작 ===")
        interaction = PlayerInteraction()
        
        # 플레이어 이름과 모델 설정 전달
        interaction.player_info["name"] = player_name
        interaction.model_id = model_id
        interaction.use_cpu = use_cpu
        interaction.use_4bit = use_4bit
        
        npc_choice = select_npc_to_chat(interaction)
        if npc_choice:
            start_chat_session(interaction, npc_choice)
        else:
            print("NPC 선택에 실패했습니다.")
    except Exception as e:
        print(f"\n시스템 시작 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()  # 자세한 오류 추적 정보 출력

def select_npc_to_chat(interaction):
    """대화할 NPC 선택 - 개선된 버전"""
    print("\nNPC 선택 메뉴")
    print("1. 저장된 NPC 불러오기")
    print("2. 새 NPC 생성하기")
    choice = input("선택 (1/2): ")
    
    if choice == "1":
        # 저장된 NPC 목록 불러오기
        saved_npcs = interaction.load_all_saved_npcs()
        if not saved_npcs:
            print("저장된 NPC가 없습니다. 새 NPC를 생성합니다.")
            return interaction.create_new_npc()
            
        # NPC 목록 표시
        print("\n=== 저장된 NPC 목록 ===")
        print(f"{'번호':<4}{'NPC ID':<20}{'수정일시':<20}")
        print("-" * 50)
        
        npc_list = list(saved_npcs.items())
        for i, (npc_id, npc_info) in enumerate(npc_list, 1):
            modified = npc_info.get("modified", "Unknown")
            print(f"{i:<4}{npc_id:<20}{modified:<20}")
            
        # NPC 선택
        try:
            select_num = int(input("\n선택할 NPC 번호 (0: 새로 만들기): "))
            if select_num == 0:
                return interaction.create_new_npc()
                
            if 1 <= select_num <= len(npc_list):
                selected_npc_id = npc_list[select_num-1][0]
                file_path = npc_list[select_num-1][1].get("file_path")
                
                loaded_npc = interaction.load_npc(npc_id=selected_npc_id, file_path=file_path)
                if loaded_npc:
                    return loaded_npc.npc_id
                else:
                    print("NPC 로드에 실패했습니다. 새 NPC를 생성합니다.")
                    return interaction.create_new_npc()
            else:
                print("잘못된 번호입니다. 새 NPC를 생성합니다.")
                return interaction.create_new_npc()
        except Exception as e:
            print(f"입력 오류: {e}. 새 NPC를 생성합니다.")
            return interaction.create_new_npc()
    else:
        # 새 NPC 생성
        return interaction.create_new_npc()

def start_chat_session(interaction, npc_id):
    """선택한 NPC와 대화 시작"""
    if npc_id not in interaction.npcs:
        print(f"오류: NPC {npc_id}를 찾을 수 없습니다.")
        return
        
    npc = interaction.npcs[npc_id]
    npc_name = npc.current_state.get('name', npc_id)
    print(f"\n{npc_name}와(과) 대화를 시작합니다.")
    print("대화를 종료하려면 '/exit'를 입력하세요.")
    
    chatting = True
    while chatting:
        player_input = input(f"\n당신: ")
        
        if player_input.lower() == "/exit":
            print("대화를 종료합니다.")
            break

        response = interaction.chat_with_npc(npc_id, player_input)
        print(f"{npc_name}: {response}")

if __name__ == "__main__":
    # 시스템 설정
    print("\n=== RPG NPC 대화 시스템 ===")
    
    # 플레이어 이름 입력
    player_name = input("당신의 이름을 입력하세요: ")
    if not player_name:
        player_name = "모험가"
        print(f"이름이 입력되지 않아 '{player_name}'(으)로 지정됩니다.")
    
    # 하드웨어 설정
    print("\n=== 하드웨어 설정 ===")
    
    # CPU 모드 선택
    use_cpu = False
    if pytorch_installed and torch.cuda.is_available():
        print(f"감지된 GPU: {torch.cuda.get_device_name(0)}")
        use_cpu_input = input("CPU 모드로 실행하시겠습니까? (y/n, 기본: n): ").lower()
        use_cpu = use_cpu_input.startswith('y')
    else:
        use_cpu = True
        print("GPU가 감지되지 않아 CPU 모드로 실행됩니다.")
        
    # 4비트 양자화 선택 (GPU를 사용하는 경우만)
    use_4bit = False
    if not use_cpu:
        use_4bit_input = input("4비트 양자화를 사용하시겠습니까? (적은 메모리 사용) (y/n, 기본: y): ").lower()
        use_4bit = not use_4bit_input.startswith('n')  # 기본값 True
    
    # 모델 선택
    model_id = select_model()
    
    try:
        # NPC 데이터 로드
        print("\n=== RPG NPC 대화 시스템 시작 ===")
        interaction = PlayerInteraction()
        
        # 플레이어 이름과 모델 설정 전달
        interaction.player_info["name"] = player_name
        interaction.model_id = model_id
        interaction.use_cpu = use_cpu
        interaction.use_4bit = use_4bit
        
        npc_choice = select_npc_to_chat(interaction)
        if npc_choice:
            start_chat_session(interaction, npc_choice)
        else:
            print("NPC 선택에 실패했습니다.")
    except Exception as e:
        print(f"\n시스템 시작 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()  # 자세한 오류 추적 정보 출력 