import json
import random
import time
import os
import sys
import re
import logging
from datetime import datetime, timedelta
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 모델 설정 및 로그인
login(token="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")

class MemorySystem:
    """
    향상된 메모리 관리 시스템 - 장기 기억과 단기 기억 구분 및 세계관 지식 포함
    """
    def __init__(self, owner_id, max_short_term=30, max_long_term=200):
        self.owner_id = owner_id
        
        # 메모리 시스템
        self.short_term_memories = []  # 최근 대화 및 이벤트 (휘발성)
        self.long_term_memories = []   # 영구적 기억 (중요 정보)
        self.world_knowledge = []      # 세계관 지식 (배경, 역사, 지리 등)
        self.personal_experiences = [] # 캐릭터의 개인적 경험
        
        # 메모리 제한
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        
        # 메모리 관리 도구
        self.memory_embeddings = {}    # 메모리 임베딩 캐시
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.conversation_summary = ""  # 대화 요약
        self.identity_info = {}  # 정체성 정보
        self.last_summary_time = time.time()  # 마지막 요약 시간
        
        # 감정 상태 추적
        self.emotional_state = {
            "current_emotion": "neutral",
            "emotion_strength": 0.5,
            "emotion_history": []
        }
        
        # 시간 인식
        self.last_interaction_time = time.time()
        
    def add_memory(self, memory):
        """메모리 추가 및 중요도에 따른 분류 - 개선된 버전"""
        memory["timestamp"] = memory.get("timestamp", time.time())
        memory["age"] = 0  # 메모리 경과 시간
        
        # 기본 중요도 설정
        if "importance" not in memory:
            memory["importance"] = self._calculate_importance(memory)
        
        # 메모리 타입별 처리
        memory_type = memory.get("type", "general")
        
        # 세계관 지식 추가
        if memory_type == "world_knowledge":
            self.world_knowledge.append(memory)
            return
            
        # 개인 경험 추가
        elif memory_type == "personal_experience":
            self.personal_experiences.append(memory)
            # 중요한 경험은 장기 기억에도 추가
            if memory["importance"] >= 0.7:
                self.long_term_memories.append(memory)
            return
        
        # 정체성 정보는 항상 장기 기억으로
        elif memory_type == "identity":
            self.identity_info = memory
            memory["importance"] = 1.0
            self.long_term_memories.append(memory)
            return
            
        # 감정 기록
        elif memory_type == "emotion":
            # 감정 기록 추가
            self.emotional_state["emotion_history"].append({
                "emotion": memory.get("emotion", "neutral"),
                "strength": memory.get("strength", 0.5),
                "timestamp": memory["timestamp"]
            })
            # 최대 20개 유지
            if len(self.emotional_state["emotion_history"]) > 20:
                self.emotional_state["emotion_history"].pop(0)
            
            # 현재 감정 상태 업데이트
            self.emotional_state["current_emotion"] = memory.get("emotion", "neutral")
            self.emotional_state["emotion_strength"] = memory.get("strength", 0.5)
            
            # 감정 변화가 중요하면 장기 기억에 추가
            if memory["importance"] >= 0.7:
                self.long_term_memories.append(memory)
            else:
                self.short_term_memories.append(memory)
            return
        
        # 일반 메모리 처리 (대화, 이벤트 등)
        # 중요도에 따라 단기/장기 기억 분류
        if memory["importance"] >= 0.7:
            self.long_term_memories.append(memory)
            # 장기 기억 제한
            if len(self.long_term_memories) > self.max_long_term:
                self.long_term_memories.sort(key=lambda x: x["importance"])
                self.long_term_memories = self.long_term_memories[-self.max_long_term:]
        else:
            self.short_term_memories.append(memory)
            # 단기 기억 제한
            if len(self.short_term_memories) > self.max_short_term:
                self.short_term_memories.pop(0)  # 가장 오래된 메모리 제거
        
        # 대화 요약 트리거 (10분마다 또는 단기 기억 가득 찰 때)
        current_time = time.time()
        if (memory_type == "conversation" and 
            (current_time - self.last_summary_time > 600 or  # 10분
             len(self.short_term_memories) >= self.max_short_term * 0.8)):
            self._summarize_conversations()
            self.last_summary_time = current_time
            
        # 상호작용 시간 업데이트
        self.last_interaction_time = time.time()
    
    def _calculate_importance(self, memory):
        """메모리 중요도 계산 - 개선된 버전"""
        importance = 0.5  # 기본값
        
        # 타입별 중요도 보정
        memory_type = memory.get("type", "general")
        
        if memory_type == "conversation":
            importance = 0.6
            # 감정적 단어가 포함된 대화는 더 중요
            emotion_words = ["행복", "기쁨", "슬픔", "화", "놀라움", "무서움", "사랑", "미움",
                            "happy", "sad", "angry", "surprised", "fear", "love", "hate"]
            content = memory.get("content", "")
            if any(word in content.lower() for word in emotion_words):
                importance += 0.1
                
            # 플레이어 이름이 언급되면 더 중요
            if memory.get("with_player", False) or (
               "player_name" in memory and memory["player_name"] in content):
                importance += 0.15
            
            # 중요 정보 키워드가 있으면 더 중요
            important_keywords = ["위험", "비밀", "중요", "약속", "죽음", "보물", "마법", "퀘스트",
                                 "danger", "secret", "important", "promise", "death", "treasure", "magic", "quest"]
            if any(word in content.lower() for word in important_keywords):
                importance += 0.2
                
        elif memory_type == "event":
            importance = 0.7  # 이벤트는 기본적으로 더 중요
        
            # 특별 이벤트는 더 중요
            special_events = ["전투", "발견", "전환점", "보상", "처벌", 
                             "battle", "discovery", "turning_point", "reward", "punishment"]
            if any(event in str(memory.get("content", "")).lower() for event in special_events):
                importance += 0.15
                
        elif memory_type == "relationship_change":
            importance = 0.8  # 관계 변화는 매우 중요
            
        elif memory_type == "emotion_change":
            importance = 0.7  # 감정 변화도 중요
            # 강한 감정 변화는 더 중요
            if memory.get("strength", 0) > 0.7:
                importance += 0.1
            
        elif memory_type == "world_knowledge":
            importance = 0.9  # 세계관 지식은 매우 중요
            
        elif memory_type == "personal_experience":
            importance = 0.75  # 개인적 경험도 중요
            
        # 시간에 따른 중요도 감소 (오래된 기억 중요도 감소)
        time_diff = time.time() - memory.get("timestamp", time.time())
        if time_diff > 86400:  # 24시간 이상 경과
            importance -= min(0.3, (time_diff / 86400) * 0.05)  # 하루당 최대 0.05 감소, 최대 0.3까지
            
        return min(max(importance, 0.1), 1.0)  # 0.1~1.0 사이 값
    
    def retrieve_relevant_memories(self, query="", max_count=5, threshold=0.0):
        """쿼리와 관련된 메모리 검색 - 개선된 버전"""
        all_memories = []
        
        # 1. 단기 기억에서 찾기 (최근 대화 등)
        all_memories.extend(self.short_term_memories)
        
        # 2. 장기 기억에서 찾기 (중요 이벤트 등)
        all_memories.extend(self.long_term_memories)
        
        # 3. 세계관 지식 검색 (쿼리와 관련있는 경우만)
        # 세계관 관련 키워드가 있는지 확인
        world_keywords = ["역사", "전설", "마법", "왕국", "종족", "직업", "전쟁", "신화", "고대",
                          "history", "legend", "magic", "kingdom", "race", "job", "war", "myth", "ancient"]
        if any(keyword in query.lower() for keyword in world_keywords):
            relevant_world_knowledge = []
            for knowledge in self.world_knowledge:
                # 제목이나 내용이 쿼리와 관련 있는지 확인
                knowledge_text = f"{knowledge.get('title', '')} {knowledge.get('content', '')}"
                if any(keyword in knowledge_text.lower() for keyword in query.lower().split()):
                    relevant_world_knowledge.append(knowledge)
            
            # 최대 3개만 추가
            all_memories.extend(relevant_world_knowledge[:3])
        
        # 4. 개인 경험 추가 (쿼리와 관련 있는 것만)
        experience_keywords = ["경험", "과거", "배움", "일", "기억나", "해본", "느낌", 
                              "experience", "past", "learn", "work", "remember", "feel"]
        if any(keyword in query.lower() for keyword in experience_keywords):
            relevant_experiences = []
            for exp in self.personal_experiences:
                exp_text = exp.get('content', '')
                if any(keyword in exp_text.lower() for keyword in query.lower().split()):
                    relevant_experiences.append(exp)
            
            # 최대 2개만 추가
            all_memories.extend(relevant_experiences[:2])
        
        # 검색된 메모리가 없으면 빈 리스트 반환
        if not all_memories:
            return []
            
        # 메모리 관련성 및 중요도에 따라 정렬
        # 1. 쿼리와의 관련성 점수 계산
        memory_texts = [m.get("content", "") for m in all_memories]
        query_vector = self.vectorizer.fit_transform([query])
        memory_vectors = self.vectorizer.transform(memory_texts)
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, memory_vectors)[0]
        
        # 각 메모리에 관련성 점수 추가
        for i, memory in enumerate(all_memories):
            memory["relevance_score"] = similarities[i] if i < len(similarities) else 0
        
        # 2. 종합 점수 계산 (관련성 * 0.7 + 중요도 * 0.3)
        for memory in all_memories:
            memory["combined_score"] = (
                memory.get("relevance_score", 0) * 0.7 + 
                memory.get("importance", 0) * 0.3
            )
        
        # 3. 종합 점수로 정렬
        sorted_memories = sorted(
            all_memories,
            key=lambda x: x.get("combined_score", 0),
            reverse=True
        )
        
        # 관련성 임계값 이상의 메모리만 필터링
        filtered_memories = [m for m in sorted_memories if m.get("relevance_score", 0) >= threshold]
        
        # 상위 max_count개 반환
        return filtered_memories[:max_count]
    
    def retrieve_memories_by_type(self, memory_type, max_count=5):
        """특정 타입의 메모리 검색 - 개선된 버전"""
        # 타입에 따라 검색 소스 결정
        if memory_type == "world_knowledge":
            source = self.world_knowledge
        elif memory_type == "personal_experience":
            source = self.personal_experiences
        else:
            # 일반 메모리는 단기+장기 기억에서 검색
            source = self.short_term_memories + self.long_term_memories
            
        # 타입으로 필터링
        filtered_memories = [m for m in source if m.get("type") == memory_type]
        
        # 시간 역순으로 정렬
        sorted_memories = sorted(
            filtered_memories, 
            key=lambda x: x.get("timestamp", 0),
            reverse=True
        )
        
        return sorted_memories[:max_count]
    
    def get_emotional_state(self):
        """현재 감정 상태 가져오기"""
        return self.emotional_state
    
    def forget_oldest_memories(self, memory_type=None, max_count=100):
        """오래된 메모리 삭제 (단기 기억에서만)"""
        if not self.short_term_memories:
            return
            
        # 메모리 타입 필터링
        if memory_type:
            # 타입이 일치하는 단기 기억만 필터링
            matching_memories = [m for m in self.short_term_memories if m.get("type") == memory_type]
            non_matching_memories = [m for m in self.short_term_memories if m.get("type") != memory_type]
            
            # 타입이 일치하는 메모리가 max_count 초과면 정리
            if len(matching_memories) > max_count:
                # 시간순으로 정렬
                sorted_memories = sorted(matching_memories, key=lambda x: x.get("timestamp", 0))
                # 최신 max_count개만 유지
                keep_memories = sorted_memories[-max_count:]
                # 메모리 업데이트
                self.short_term_memories = keep_memories + non_matching_memories
        else:
            # 단기 기억 전체가 max_count 초과면 정리
            if len(self.short_term_memories) > max_count:
                # 시간순으로 정렬
                sorted_memories = sorted(self.short_term_memories, key=lambda x: x.get("timestamp", 0))
                # 최신 max_count개만 유지
                self.short_term_memories = sorted_memories[-max_count:]
    
    def _summarize_conversations(self):
        """대화 요약 생성"""
        # 대화 기록에서 최근 대화만 추출
        if not hasattr(self, 'conversation_memory') or len(self.conversation_memory) < 4:
            return
            
        conversations = self.conversation_memory[-min(20, len(self.conversation_memory)):]
        
        # 토픽과 감정 추출
        topics = self._extract_topics(conversations)
        emotions = self._extract_emotions(conversations)
        
        # 요약 생성
        self.conversation_summary = (
            f"최근 대화 요약: {len(conversations)}개의 대화가 있었습니다. "
            f"주요 주제는 {', '.join(topics)}이며, "
            f"대화 중 느껴진 감정은 {', '.join(emotions)}입니다.\n"
        )
    
    def add_world_knowledge(self, knowledge_data):
        """세계관 지식 추가"""
        if isinstance(knowledge_data, list):
            self.world_knowledge.extend(knowledge_data)
        else:
            self.world_knowledge.append(knowledge_data)
    
    def add_personal_experience(self, experience_data):
        """개인 경험 추가"""
        if isinstance(experience_data, list):
            self.personal_experiences.extend(experience_data)
        else:
            self.personal_experiences.append(experience_data)
    
    def get_memory_stats(self):
        """메모리 통계 반환 - 새로운 메서드"""
        return {
            "short_term_count": len(self.short_term_memories),
            "long_term_count": len(self.long_term_memories),
            "world_knowledge_count": len(self.world_knowledge),
            "personal_experience_count": len(self.personal_experiences),
            "last_interaction": datetime.fromtimestamp(self.last_interaction_time).strftime("%Y-%m-%d %H:%M:%S"),
            "emotional_state": self.emotional_state["current_emotion"]
        }

    def get_relevant_knowledge(self, query, max_results=3):
        """쿼리와 관련된 세계 지식 검색"""
        if not self.world_knowledge or not query:
            return []
            
        # 모든 세계 지식 텍스트 추출
        knowledge_texts = []
        for knowledge in self.world_knowledge:
            content = knowledge.get('content', '')
            title = knowledge.get('title', '')
            knowledge_texts.append((f"{title}: {content}", knowledge))
            
        # 관련성 점수 계산
        query_terms = query.lower().split()
        scored_knowledge = []
        
        for text, knowledge in knowledge_texts:
            text_lower = text.lower()
            score = sum(1 for term in query_terms if term in text_lower)
            
            # 카테고리 가중치
            category = knowledge.get('category', '')
            if any(cat in query.lower() for cat in [category, category.lower()]):
                score += 2
                
            # 제목 가중치
            title = knowledge.get('title', '')
            if any(term in title.lower() for term in query_terms):
                score += 3
                
            # 중요도 가중치
            score += knowledge.get('importance', 0.5) * 5
            
            if score > 0:
                scored_knowledge.append((score, knowledge))
        
        # 점수순 정렬 및 상위 결과 반환
        scored_knowledge.sort(reverse=True, key=lambda x: x[0])
        
        return [k for _, k in scored_knowledge[:max_results]]
        
    def get_relevant_experiences(self, query, max_results=2):
        """쿼리와 관련된 개인 경험 검색"""
        if not self.personal_experiences or not query:
            return []
            
        # 모든 개인 경험 텍스트 추출
        experience_texts = []
        for exp in self.personal_experiences:
            content = exp.get('content', '')
            title = exp.get('title', '')
            experience_texts.append((f"{title}: {content}", exp))
            
        # 관련성 점수 계산
        query_terms = query.lower().split()
        scored_experiences = []
        
        for text, exp in experience_texts:
            text_lower = text.lower()
            score = sum(1 for term in query_terms if term in text_lower)
            
            # 제목 가중치
            title = exp.get('title', '')
            if any(term in title.lower() for term in query_terms):
                score += 2
                
            # 중요도 가중치
            score += exp.get('importance', 0.5) * 3
            
            if score > 0:
                scored_experiences.append((score, exp))
        
        # 점수순 정렬 및 상위 결과 반환
        scored_experiences.sort(reverse=True, key=lambda x: x[0])
        
        return [e for _, e in scored_experiences[:max_results]]
    
    def summarize_conversation(self, messages, max_length=3):
        """
        최근 대화 내용을 요약하고 핵심 정보를 추출합니다.
        """
        if not messages or len(messages) < 2:
            return "충분한 대화가 없습니다."
        
        # 최근 메시지 추출 (최대 10개)
        recent_messages = messages[-min(10, len(messages)):]
        
        # 대화 내용 포맷팅
        formatted_dialogue = []
        for i in range(0, len(recent_messages), 2):
            if i+1 < len(recent_messages):
                user_msg = recent_messages[i][:100] + "..." if len(recent_messages[i]) > 100 else recent_messages[i]
                ai_msg = recent_messages[i+1][:150] + "..." if len(recent_messages[i+1]) > 150 else recent_messages[i+1]
                formatted_dialogue.append(f"사용자: {user_msg}\nNPC: {ai_msg}")
        
        # 핵심 키워드 추출
        all_text = " ".join(recent_messages)
        words = all_text.lower().split()
        
        # 불용어 제거
        stopwords = ['그', '이', '저', '것', '수', '를', '에', '은', '는', '이', '가', '의', '에서', '로', '과', '와', '이런', '저런', '그런', '합니다', '있습니다', '없습니다', '하는', '있는', '없는']
        filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
        
        # 단어 빈도 계산
        word_freq = {}
        for word in filtered_words:
            if word in word_freq:
                word_freq[word] += 1
        else:
                word_freq[word] = 1
        
        # 핵심 키워드 5개 추출
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        key_topics = [word for word, freq in sorted_words[:5] if len(word) > 1]
        
        # 대화에서 추출된 정보
        extracted_info = self._extract_conversation_info(recent_messages)
        
        # 요약 생성
        summary = []
        summary.append(f"주요 키워드: {', '.join(key_topics)}")
        
        if extracted_info.get('locations'):
            summary.append(f"언급된 장소: {', '.join(extracted_info['locations'])}")
            
        if extracted_info.get('names'):
            summary.append(f"언급된 인물: {', '.join(extracted_info['names'])}")
            
        if extracted_info.get('quests'):
            summary.append(f"가능한 퀘스트: {', '.join(extracted_info['quests'])}")
            
        if extracted_info.get('items'):
            summary.append(f"언급된 아이템: {', '.join(extracted_info['items'])}")
        
        summary.append(f"대화 내용 요약:\n{formatted_dialogue[-max_length:]}")
        
        return "\n".join(summary)
    
    def _extract_conversation_info(self, messages):
        """
        대화 내용에서 중요한 정보(장소, 이름, 아이템, 잠재적 퀘스트 등)를 추출합니다.
        """
        info = {
            'locations': set(),
            'names': set(),
            'items': set(),
            'quests': set()
        }
        
        # 월드 로어에서 주요 장소 이름 추출
        locations = []
        for location_data in self.world_knowledge:
            if location_data.get('category') == 'location':
                locations.append(location_data['title'])
        
        # 대화 내용에서 정보 추출
        all_text = " ".join(messages)
        
        # 장소 확인
        for location in locations:
            if location in all_text:
                info['locations'].add(location)
        
        # 퀘스트 관련 키워드 확인
        quest_keywords = ['퀘스트', '임무', '의뢰', '도움', '찾아', '도와줘', '해결', '문제']
        if any(keyword in all_text for keyword in quest_keywords):
            # 문장 단위로 분리
            sentences = all_text.split('. ')
            for sentence in sentences:
                if any(keyword in sentence for keyword in quest_keywords):
                    # 간단한 퀘스트 이름 생성
                    words = sentence.split()
                    if len(words) > 3:
                        # 문장에서 3-5단어로 퀘스트 이름 추출
                        quest_name = " ".join(words[:min(5, len(words))])
                        info['quests'].add(f"{quest_name}...")
        
        # 아이템 언급 확인
        item_keywords = ['검', '칼', '방패', '갑옷', '지도', '물약', '책', '마법', '반지', '아이템', '보물']
        for item in item_keywords:
            if item in all_text:
                # 아이템이 언급된 문장 찾기
                sentences = all_text.split('. ')
                for sentence in sentences:
                    if item in sentence:
                        # 해당 아이템 주변 단어 추출하여 아이템 이름 구성
                        words = sentence.split()
                        item_index = -1
                        for i, word in enumerate(words):
                            if item in word:
                                item_index = i
                                break
                        
                        if item_index >= 0:
                            start = max(0, item_index - 1)
                            end = min(len(words), item_index + 2)
                            possible_item = " ".join(words[start:end])
                            info['items'].add(possible_item)
        
        # 결과를 리스트로 변환
        for key in info:
            info[key] = list(info[key])
            
        return info

    def _load_world_knowledge(self, world_data):
        """세계관 설정을 메모리에 로드"""
        print("세계관 정보 로드 중...")
        
        # 메모리 객체가 리스트인 경우 객체로 생성
        if isinstance(self.memory, list):
            self.memory = MemorySystem(self.npc_id)
        
        # 세계 기본 정보
        if 'info' in world_data:
            self.world_info = world_data['info']
            print(f"세계 이름: {self.world_info.get('name', '불명')}")
        
        # 각 카테고리별 지식 추가
        categories = ['history', 'locations', 'races', 'classes', 'magic', 'characters', 'artifacts']
        total_entries = 0
        
        for category in categories:
            if category in world_data:
                for entry in world_data[category]:
                    self.memory.add_memory(entry)
                    total_entries += 1
        
        print(f"총 {total_entries}개의 세계 지식 항목이 로드되었습니다.")
    
    def _load_default_knowledge(self):
        """기본 세계관 지식 생성 (world_lore.py가 없을 때 사용)"""
        print("기본 세계 지식 생성 중...")
        
        # 메모리 객체가 리스트인 경우 객체로 생성
        if isinstance(self.memory, list):
            self.memory = MemorySystem(self.npc_id)
        
        # 기본 세계 지식 생성
        basic_knowledge = [
            {
                "type": "world_knowledge",
                "category": "history",
                "title": "세계 창조",
                "content": "오래 전, 신들이 이 세계를 창조했다고 전해진다. 마법은 신들이 인간에게 준 선물로 여겨진다.",
                "importance": 0.8
            },
            {
                "type": "world_knowledge",
                "category": "location",
                "title": "왕국",
                "content": "현재 세계는 여러 왕국으로 나뉘어 있으며, 인간, 엘프, 드워프 등 다양한 종족이 살고 있다.",
                "importance": 0.7
            },
            {
                "type": "world_knowledge",
                "category": "race",
                "title": "인간",
                "content": "가장 널리 퍼진 종족으로, 적응력이 강하고 다양한 직업과 재능을 가진다.",
                "importance": 0.6
            },
            {
                "type": "world_knowledge",
                "category": "race",
                "title": "엘프",
                "content": "숲에 사는 장수하는 종족으로, 마법과 자연과의 교감에 뛰어나다.",
                "importance": 0.6
            },
            {
                "type": "world_knowledge",
                "category": "class",
                "title": "마법사",
                "content": "마법의 비밀을 연구하고 다양한 주문을 사용하는 직업. 지식을 중요시한다.",
                "importance": 0.7
            },
            {
                "type": "world_knowledge",
                "category": "magic",
                "title": "마법",
                "content": "마법은 원소, 생명, 환영 등 다양한 형태로 존재하며, 마법사들은 오랜 훈련을 통해 이를 다루는 법을 배운다.",
                "importance": 0.75
            }
        ]
        
        # 메모리에 기본 지식 추가
        for knowledge in basic_knowledge:
            self.memory.add_memory(knowledge)
        
        print(f"총 {len(basic_knowledge)}개의 기본 세계 지식 항목이 생성되었습니다.")
    
    def _initialize_personal_experiences(self):
        """NPC의 개인적인 경험과 배경 초기화"""
        species = self.npc_data.get('instincts', {}).get('species', 'Human')
        occupation = self.npc_data.get('instincts', {}).get('occupation', '일반인')
        
        # 종족에 따른 기본 경험
        species_experiences = {
            'Human': [
                {
                    "type": "personal_experience",
                    "category": "background",
                    "title": "인간 마을에서의 성장",
                    "content": "평범한 인간 마을에서 자라며 다양한 직업과 삶의 방식을 접했습니다.",
                    "time": "어린 시절",
                    "importance": 0.7
                }
            ],
            'Elf': [
                {
                    "type": "personal_experience",
                    "category": "background",
                    "title": "숲속에서의 성장",
                    "content": "고대 숲에서 자라며 자연과 교감하는 법을 배웠습니다. 엘프들의 오랜 역사와 전통을 배웠습니다.",
                    "time": "유년기",
                    "importance": 0.8
                }
            ],
            'Dwarf': [
                {
                    "type": "personal_experience",
                    "category": "background",
                    "title": "산속 요새에서의 성장",
                    "content": "산 속 드워프 왕국에서 자라며 대장장이 기술과 광물에 대한 지식을 쌓았습니다.",
                    "time": "청소년기",
                    "importance": 0.7
                }
            ],
            'Orc': [
                {
                    "type": "personal_experience",
                    "category": "background",
                    "title": "부족에서의 성장",
                    "content": "오크 부족에서 자라며 생존 기술과 전투 능력을 갖추게 되었습니다.",
                    "time": "성장기",
                    "importance": 0.75
                }
            ]
        }
        
        # 직업에 따른 기본 경험
        occupation_experiences = {
            'merchant': [
                {
                    "type": "personal_experience",
                    "category": "career",
                    "title": "상인 길드 가입",
                    "content": "상인 길드에 가입하여 거래와 협상의 기술을 배웠습니다.",
                    "time": "5년 전",
                    "importance": 0.8
                },
                {
                    "type": "personal_experience",
                    "category": "skill",
                    "title": "가치 평가 능력",
                    "content": "다양한 물건의 가치를 정확히 평가하는 능력을 갖추게 되었습니다.",
                    "time": "경험을 통해",
                    "importance": 0.7
                }
            ],
            'guard': [
                {
                    "type": "personal_experience",
                    "category": "career",
                    "title": "경비대 훈련",
                    "content": "도시 경비대에서 엄격한 훈련을 받았습니다.",
                    "time": "7년 전",
                    "importance": 0.8
                },
                {
                    "type": "personal_experience",
                    "category": "event",
                    "title": "도시 방어 경험",
                    "content": "도시를 습격한 도적 무리를 물리치는 데 참여했습니다.",
                    "time": "3년 전",
                    "importance": 0.85
                }
            ],
            'mage': [
                {
                    "type": "personal_experience",
                    "category": "education",
                    "title": "마법 학교 입학",
                    "content": "어린 나이에 마법적 재능을 인정받아 마법 학교에 입학했습니다.",
                    "time": "20년 전",
                    "importance": 0.9
                },
                {
                    "type": "personal_experience",
                    "category": "skill",
                    "title": "첫 주문 습득",
                    "content": "불을 다루는 첫 주문을 습득했을 때의 기쁨을 아직도 기억합니다.",
                    "time": "19년 전",
                    "importance": 0.7
                },
                {
                    "type": "personal_experience",
                    "category": "achievement",
                    "title": "마법 연구 성과",
                    "content": "새로운 마법 이론에 관한 연구로 마법 학회에서 인정받았습니다.",
                    "time": "5년 전",
                    "importance": 0.85
                }
            ],
            'innkeeper': [
                {
                    "type": "personal_experience",
                    "category": "career",
                    "title": "여관 경영 시작",
                    "content": "부모님으로부터 여관 경영을 물려받아 시작했습니다.",
                    "time": "10년 전",
                    "importance": 0.8
                },
                {
                    "type": "personal_experience",
                    "category": "skill",
                    "title": "요리 솜씨",
                    "content": "여러 지역의 요리법을 배워 손님들에게 제공합니다.",
                    "time": "경험을 통해",
                    "importance": 0.6
                }
            ],
            'blacksmith': [
                {
                    "type": "personal_experience",
                    "category": "education",
                    "title": "대장장이 견습",
                    "content": "유명한 대장장이 밑에서 견습생 생활을 했습니다.",
                    "time": "15년 전",
            "importance": 0.8
                },
                {
                    "type": "personal_experience",
                    "category": "achievement",
                    "title": "첫 무기 제작",
                    "content": "독자적으로 첫 무기를 제작했을 때의 성취감을 잊지 못합니다.",
                    "time": "12년 전",
                    "importance": 0.75
                }
            ]
        }
        
        # 종족 경험 추가
        species_key = species if species in species_experiences else 'Human'
        for exp in species_experiences.get(species_key, []):
            self.memory.add_memory(exp)
            
        # 직업 경험 추가
        occupation_key = occupation.lower() if occupation.lower() in occupation_experiences else 'merchant'
        for exp in occupation_experiences.get(occupation_key, []):
            self.memory.add_memory(exp)
            
        # 일반적인 경험 추가
        general_experiences = [
            {
                "type": "personal_experience",
                "category": "travel",
                "title": "여행 경험",
                "content": "여러 지역을 여행하며 다양한 문화와 사람들을 만났습니다.",
                "time": "여러 해 동안",
                "importance": 0.6
            },
            {
                "type": "personal_experience",
                "category": "relationship",
                "title": "친구와의 인연",
                "content": "어려운 시기에 도움을 준 친구들이 있어 감사하게 생각합니다.",
                "time": "인생 전반",
                "importance": 0.7
            }
        ]
        
        for exp in general_experiences:
            self.memory.add_memory(exp)
            
        print(f"총 {len(species_experiences.get(species_key, [])) + len(occupation_experiences.get(occupation_key, [])) + len(general_experiences)}개의 개인 경험이 초기화되었습니다.")

    def _extract_topics(self, conversations):
        """대화에서 주요 주제 추출 - 개선된 버전"""
        # 모든 대화를 합쳐 자주 등장하는 주제어 추출
        all_text = " ".join(conversations).lower() if isinstance(conversations[0], str) else " ".join([c for c in conversations if isinstance(c, str)])
        
        # 한국어, 영어 주요 주제어 후보
        topic_categories = {
            "일상": ["안녕", "날씨", "기분", "오늘", "어제", "내일", "시간", "기다", "만나", "hello", "weather", "today"],
            "전투": ["전투", "싸움", "무기", "공격", "방어", "죽", "피해", "battle", "fight", "weapon", "attack", "defend"],
            "마법": ["마법", "주문", "마나", "마력", "원소", "불", "물", "바람", "대지", "magic", "spell", "mana", "element"],
            "거래": ["거래", "구매", "판매", "가격", "돈", "금화", "은화", "보상", "trade", "buy", "sell", "price", "gold"],
            "퀘스트": ["퀘스트", "임무", "도움", "부탁", "의뢰", "보상", "quest", "mission", "help", "favor", "reward"],
            "정보": ["정보", "소식", "소문", "이야기", "역사", "전설", "info", "news", "rumor", "story", "history", "legend"],
            "장소": ["장소", "마을", "도시", "숲", "산", "던전", "여관", "상점", "place", "town", "city", "forest", "mountain"],
            "관계": ["친구", "적", "동료", "가족", "연인", "스승", "제자", "friend", "enemy", "ally", "family", "lover", "master"]
        }
        
        # 각 주제별 키워드 매칭 횟수 계산
        topic_scores = {}
        for topic, keywords in topic_categories.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            topic_scores[topic] = score
        
        # 점수가 높은 순으로 주제 정렬
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 점수가 0보다 큰 주제만 선택하여 최대 3개 반환
        relevant_topics = [topic for topic, score in sorted_topics if score > 0][:3]
        
        return relevant_topics if relevant_topics else ["일상 대화"]
    
    def _extract_emotions(self, conversations):
        """대화에서 감정 추출"""
        all_text = " ".join(conversations).lower() if isinstance(conversations[0], str) else " ".join([c for c in conversations if isinstance(c, str)])
        
        # 감정 키워드 사전
        emotion_keywords = {
            "기쁨": ["기쁨", "행복", "좋", "웃", "즐", "happy", "joy", "glad", "smile", "enjoy"],
            "슬픔": ["슬픔", "우울", "눈물", "아쉽", "그립", "sad", "depressed", "tear", "miss", "regret"],
            "분노": ["화", "분노", "짜증", "열받", "미워", "angry", "mad", "annoy", "hate", "furious"],
            "두려움": ["두려움", "공포", "무서", "걱정", "불안", "fear", "scary", "worry", "anxious", "terrified"],
            "놀람": ["놀람", "깜짝", "예상", "충격", "surprised", "shock", "unexpected", "amazed"],
            "호기심": ["궁금", "호기심", "알고싶", "질문", "curious", "wonder", "question", "interest"],
            "혼란": ["혼란", "이해", "모르", "어렵", "confused", "understand", "difficult", "complex"]
        }
        
        # 각 감정별 점수 계산
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            emotion_scores[emotion] = score
        
        # 점수가 높은 순으로 감정 정렬
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 점수가 0보다 큰 감정만 선택하여 최대 2개 반환
        relevant_emotions = [emotion for emotion, score in sorted_emotions if score > 0][:2]
        
        return relevant_emotions if relevant_emotions else ["중립"]


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

class NPCBrain:
    """
    NPC의 두뇌 역할을 하는 클래스
    대화, 의사결정, 기억 등의 기능 포함
    """
    
    def __init__(self, npc_id='npc_default', npc_data=None, model_type='gemma', use_cpu_mode=False, 
                 quantization=None, name=None, location=None, **kwargs):
        """
        NPC 두뇌 객체 초기화
        
        Args:
            npc_id (str): NPC 고유 ID
            npc_data (dict): NPC 기본 데이터 (성격, 목표, 지식 등)
            model_type (str): 사용할 모델 타입 (gemma, llama3, mistral 등)
            use_cpu_mode (bool): CPU 모드 사용 여부
            quantization (str): 양자화 설정 (4bit, 8bit 등)
            name (str): NPC 이름
            location (str): NPC 위치
            **kwargs: 추가 키워드 인자
        """
        print(f"NPCBrain 초기화 시작: {npc_id}")
        
        # 기본 속성 설정
        self.npc_id = npc_id
        self.npc_data = npc_data or generate_ai_character_data()
        self.model_type = model_type
        self.use_cpu_mode = use_cpu_mode
        self.quantization = quantization
        
        # 현재 상태 초기화
        self.current_state = {
            'name': name or npc_id,
            'location': location or 'unknown',
            'emotions': {'neutral': 0.7},
            'energy': 100,
            'health': 100,
            'inventory': [],
            'last_update_time': time.time()
        }
        
        # 메모리 시스템 초기화
        self.memory = MemorySystem(self.npc_id)
        
        # 대화 메모리 초기화
        self.conversation_memory = []
        
        # 플레이어 정보 초기화
        self.player_info = {
            'name': '플레이어',
            'occupation': '모험가',
            'age': '알 수 없음',
            'interests': [],
            'relationship': 'neutral'
        }
        
        # PyTorch 설치 확인
        self.pytorch_installed = False
        try:
            import torch
            self.pytorch_installed = True
            
            # GPU 사용 가능 여부 확인
            if torch.cuda.is_available() and not use_cpu_mode:
                self.device = "cuda"
                print(f"CUDA 사용: {torch.cuda.get_device_name(0)}")
                # GPU 메모리 정보 출력
                try:
                    mem_info = torch.cuda.mem_get_info()
                    free_memory = mem_info[0] / 1024**3
                    total_memory = mem_info[1] / 1024**3
                    print(f"GPU 메모리: {free_memory:.2f}GB / {total_memory:.2f}GB 사용 가능")
                except:
                    print("GPU 메모리 정보를 확인할 수 없습니다.")
            else:
                self.device = "cpu"
                print("CPU 모드 사용 중")
        except ImportError:
            self.device = "cpu"
            print("PyTorch가 설치되어 있지 않습니다. CPU 모드로 실행합니다.")
        
        # 모델 옵션 설정
        self.model_options = {
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 40,
            'repetition_penalty': 1.1,
            'max_new_tokens': 200
        }
        
        # 월드 설정 로드 및 초기화
        try:
            from world_lore import WORLD_DATA
            print("세계관 설정 로드 중...")
            self._load_world_knowledge(WORLD_DATA)
        except ImportError:
            print("world_lore.py를 찾을 수 없습니다. 기본 세계 지식을 사용합니다.")
            self._load_default_knowledge()
        except Exception as e:
            print(f"세계관 데이터 로드 중 오류 발생: {e}")
            self._load_default_knowledge()
        
        # NPC 초기화
        self.initialize_npc()
        
        # 모델 로드
        self._load_model()
        
        print(f"NPCBrain 초기화 완료: {npc_id}")
        
    def _load_world_knowledge(self, world_data):
        """세계관 설정을 메모리에 로드"""
        print("세계관 정보 로드 중...")
        
        # 메모리 객체가 리스트인 경우 객체로 생성
        if isinstance(self.memory, list):
            self.memory = MemorySystem(self.npc_id)
        
        # 세계 기본 정보
        if 'info' in world_data:
            self.world_info = world_data['info']
            print(f"세계 이름: {self.world_info.get('name', '불명')}")
        
        # 각 카테고리별 지식 추가
        categories = ['history', 'locations', 'races', 'classes', 'magic', 'characters', 'artifacts']
        total_entries = 0
        
        for category in categories:
            if category in world_data:
                for entry in world_data[category]:
                    self.memory.add_memory(entry)
                    total_entries += 1
        
        print(f"총 {total_entries}개의 세계 지식 항목이 로드되었습니다.")
    
    def _load_default_knowledge(self):
        """기본 세계관 지식 생성 (world_lore.py가 없을 때 사용)"""
        print("기본 세계 지식 생성 중...")
        
        # 메모리 객체가 리스트인 경우 객체로 생성
        if isinstance(self.memory, list):
            self.memory = MemorySystem(self.npc_id)
        
        # 기본 세계 지식 생성
        basic_knowledge = [
            {
                "type": "world_knowledge",
                "category": "history",
                "title": "세계 창조",
                "content": "오래 전, 신들이 이 세계를 창조했다고 전해진다. 마법은 신들이 인간에게 준 선물로 여겨진다.",
                "importance": 0.8
            },
            {
                "type": "world_knowledge",
                "category": "location",
                "title": "왕국",
                "content": "현재 세계는 여러 왕국으로 나뉘어 있으며, 인간, 엘프, 드워프 등 다양한 종족이 살고 있다.",
                "importance": 0.7
            },
            {
                "type": "world_knowledge",
                "category": "race",
                "title": "인간",
                "content": "가장 널리 퍼진 종족으로, 적응력이 강하고 다양한 직업과 재능을 가진다.",
                "importance": 0.6
            },
            {
                "type": "world_knowledge",
                "category": "race",
                "title": "엘프",
                "content": "숲에 사는 장수하는 종족으로, 마법과 자연과의 교감에 뛰어나다.",
                "importance": 0.6
            },
            {
                "type": "world_knowledge",
                "category": "class",
                "title": "마법사",
                "content": "마법의 비밀을 연구하고 다양한 주문을 사용하는 직업. 지식을 중요시한다.",
                "importance": 0.7
            },
            {
                "type": "world_knowledge",
                "category": "magic",
                "title": "마법",
                "content": "마법은 원소, 생명, 환영 등 다양한 형태로 존재하며, 마법사들은 오랜 훈련을 통해 이를 다루는 법을 배운다.",
                "importance": 0.75
            }
        ]
        
        # 메모리에 기본 지식 추가
        for knowledge in basic_knowledge:
            self.memory.add_memory(knowledge)
        
        print(f"총 {len(basic_knowledge)}개의 기본 세계 지식 항목이 생성되었습니다.")
    
    def _initialize_personal_experiences(self):
        """NPC의 개인적인 경험과 배경 초기화"""
        species = self.npc_data.get('instincts', {}).get('species', 'Human')
        occupation = self.npc_data.get('instincts', {}).get('occupation', '일반인')
        
        # 종족에 따른 기본 경험
        species_experiences = {
            'Human': [
                {
                    "type": "personal_experience",
                    "category": "background",
                    "title": "인간 마을에서의 성장",
                    "content": "평범한 인간 마을에서 자라며 다양한 직업과 삶의 방식을 접했습니다.",
                    "time": "어린 시절",
                    "importance": 0.7
                }
            ],
            'Elf': [
                {
                    "type": "personal_experience",
                    "category": "background",
                    "title": "숲속에서의 성장",
                    "content": "고대 숲에서 자라며 자연과 교감하는 법을 배웠습니다. 엘프들의 오랜 역사와 전통을 배웠습니다.",
                    "time": "유년기",
                    "importance": 0.8
                }
            ],
            'Dwarf': [
                {
                    "type": "personal_experience",
                    "category": "background",
                    "title": "산속 요새에서의 성장",
                    "content": "산 속 드워프 왕국에서 자라며 대장장이 기술과 광물에 대한 지식을 쌓았습니다.",
                    "time": "청소년기",
                    "importance": 0.7
                }
            ],
            'Orc': [
                {
                    "type": "personal_experience",
                    "category": "background",
                    "title": "부족에서의 성장",
                    "content": "오크 부족에서 자라며 생존 기술과 전투 능력을 갖추게 되었습니다.",
                    "time": "성장기",
                    "importance": 0.75
                }
            ]
        }
        
        # 직업에 따른 기본 경험
        occupation_experiences = {
            'merchant': [
                {
                    "type": "personal_experience",
                    "category": "career",
                    "title": "상인 길드 가입",
                    "content": "상인 길드에 가입하여 거래와 협상의 기술을 배웠습니다.",
                    "time": "5년 전",
                    "importance": 0.8
                },
                {
                    "type": "personal_experience",
                    "category": "skill",
                    "title": "가치 평가 능력",
                    "content": "다양한 물건의 가치를 정확히 평가하는 능력을 갖추게 되었습니다.",
                    "time": "경험을 통해",
                    "importance": 0.7
                }
            ],
            'guard': [
                {
                    "type": "personal_experience",
                    "category": "career",
                    "title": "경비대 훈련",
                    "content": "도시 경비대에서 엄격한 훈련을 받았습니다.",
                    "time": "7년 전",
                    "importance": 0.8
                },
                {
                    "type": "personal_experience",
                    "category": "event",
                    "title": "도시 방어 경험",
                    "content": "도시를 습격한 도적 무리를 물리치는 데 참여했습니다.",
                    "time": "3년 전",
                    "importance": 0.85
                }
            ],
            'mage': [
                {
                    "type": "personal_experience",
                    "category": "education",
                    "title": "마법 학교 입학",
                    "content": "어린 나이에 마법적 재능을 인정받아 마법 학교에 입학했습니다.",
                    "time": "20년 전",
                    "importance": 0.9
                },
                {
                    "type": "personal_experience",
                    "category": "skill",
                    "title": "첫 주문 습득",
                    "content": "불을 다루는 첫 주문을 습득했을 때의 기쁨을 아직도 기억합니다.",
                    "time": "19년 전",
                    "importance": 0.7
                },
                {
                    "type": "personal_experience",
                    "category": "achievement",
                    "title": "마법 연구 성과",
                    "content": "새로운 마법 이론에 관한 연구로 마법 학회에서 인정받았습니다.",
                    "time": "5년 전",
                    "importance": 0.85
                }
            ],
            'innkeeper': [
                {
                    "type": "personal_experience",
                    "category": "career",
                    "title": "여관 경영 시작",
                    "content": "부모님으로부터 여관 경영을 물려받아 시작했습니다.",
                    "time": "10년 전",
                    "importance": 0.8
                },
                {
                    "type": "personal_experience",
                    "category": "skill",
                    "title": "요리 솜씨",
                    "content": "여러 지역의 요리법을 배워 손님들에게 제공합니다.",
                    "time": "경험을 통해",
                    "importance": 0.6
                }
            ],
            'blacksmith': [
                {
                    "type": "personal_experience",
                    "category": "education",
                    "title": "대장장이 견습",
                    "content": "유명한 대장장이 밑에서 견습생 생활을 했습니다.",
                    "time": "15년 전",
            "importance": 0.8
                },
                {
                    "type": "personal_experience",
                    "category": "achievement",
                    "title": "첫 무기 제작",
                    "content": "독자적으로 첫 무기를 제작했을 때의 성취감을 잊지 못합니다.",
                    "time": "12년 전",
                    "importance": 0.75
                }
            ]
        }
        
        # 종족 경험 추가
        species_key = species if species in species_experiences else 'Human'
        for exp in species_experiences.get(species_key, []):
            self.memory.add_memory(exp)
            
        # 직업 경험 추가
        occupation_key = occupation.lower() if occupation.lower() in occupation_experiences else 'merchant'
        for exp in occupation_experiences.get(occupation_key, []):
            self.memory.add_memory(exp)
            
        # 일반적인 경험 추가
        general_experiences = [
            {
                "type": "personal_experience",
                "category": "travel",
                "title": "여행 경험",
                "content": "여러 지역을 여행하며 다양한 문화와 사람들을 만났습니다.",
                "time": "여러 해 동안",
                "importance": 0.6
            },
            {
                "type": "personal_experience",
                "category": "relationship",
                "title": "친구와의 인연",
                "content": "어려운 시기에 도움을 준 친구들이 있어 감사하게 생각합니다.",
                "time": "인생 전반",
                "importance": 0.7
            }
        ]
        
        for exp in general_experiences:
            self.memory.add_memory(exp)
            
        print(f"총 {len(species_experiences.get(species_key, [])) + len(occupation_experiences.get(occupation_key, [])) + len(general_experiences)}개의 개인 경험이 초기화되었습니다.")
    
    def _generate_system_prompt(self):
        """NPC 특성과 메모리를 반영한 시스템 프롬프트 생성"""
        # 기본 NPC 정보 설정
        npc_name = self.current_state.get('name', 'NPC')
        species = self.npc_data.get('instincts', {}).get('species', '사람')
        occupation = self.npc_data.get('instincts', {}).get('occupation', '일반인')
        location = self.current_state.get('location', '알 수 없는 장소')
        emotions = self.current_state.get('emotions', {'neutral': 0.8})
        
        # 주요 감정 결정
        primary_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'neutral'
        
        # 직업별 특성 및 어투 설정
        occupation_traits = {
            'merchant': {
                'traits': '물건 가격과 시장 동향에 관심이 많고, 협상과 거래를 즐기며, 이득을 추구합니다.',
                'tone': '친절하고 설득력 있게 말하며, 종종 상품의 장점을 강조합니다.'
            },
            '상인': {
                'traits': '물건 가격과 시장 동향에 관심이 많고, 협상과 거래를 즐기며, 이득을 추구합니다.',
                'tone': '친절하고 설득력 있게 말하며, 종종 상품의 장점을 강조합니다.'
            },
            'guard': {
                'traits': '규율과 질서를 중요시하며, 경계심이 강하고, 시민의 안전을 우선시합니다.',
                'tone': '단호하고 직설적으로 말하며, 명령조로 대화하는 경향이 있습니다.'
            },
            'mage': {
                'traits': '지식과 마법 연구에 열정적이며, 학문적 호기심이 강하고, 신비한 현상에 관심이 많습니다.',
                'tone': '지적이고 사려깊게 말하며, 때때로 마법 용어를 사용합니다.'
            }
        }
        
        # 종족별 특성 설정
        species_traits = {
            'Human': '적응력이 뛰어나고, 다재다능하며, 다양한 문화와 기술에 개방적입니다.',
            'Elf': '장수하며 자연과 깊은 연결을 가지고 있고, 예술과 마법에 재능이 있습니다.',
            'Dwarf': '내구성이 강하고 장인정신이 뛰어나며, 금속과 보석 가공에 탁월한 기술을 가지고 있습니다.',
            'Orc': '강인한 체력과 전투 능력을 가지고 있으며, 부족의 결속을 중요시합니다.'
        }
        
        # 감정 상태별 행동 스타일
        emotion_styles = {
            'happy': '기분이 좋고 활기차며, 긍정적인 태도로 대화에 임합니다.',
            'sad': '우울하고 침울한 기분으로, 다소 느린 말투와 소극적인 대화 방식을 보입니다.',
            'angry': '화가 나 있고 짜증스러우며, 날카롭고 공격적인 어조로 말할 수 있습니다.',
            'neutral': '평온하고 차분한 상태로, 균형잡힌 방식으로 대화합니다.',
            'curious': '호기심을 가지고 관심을 보이며, 질문이 많고 적극적으로 대화에 참여합니다.'
        }
        
        # 직업 특성 및 어투 설정
        occ_traits = occupation_traits.get(occupation.lower(), {'traits': '일반적인 특성을 가지고 있습니다.', 'tone': '평범하게 대화합니다.'})
        spec_traits = species_traits.get(species, '일반적인 종족의 특성을 가지고 있습니다.')
        emo_style = emotion_styles.get(primary_emotion, '중립적인 감정 상태입니다.')
        
        # 최종 시스템 프롬프트 구성
        system_prompt = f"""당신은 판타지 RPG 세계 '테라노바'의 NPC인 {npc_name}({species} {occupation})로 연기해야 합니다. 현재 위치는 {location}입니다.

특성:
- {spec_traits}
- {occ_traits['traits']}
- 현재 감정 상태: {primary_emotion} - {emo_style}

대화 스타일:
- {occ_traits['tone']}
- 짧고 간결하게 대답합니다 (3-4문장 이내).
- 항상 한국어로 응답합니다.
- 플레이어가 물어보지 않은 정보는 함부로 공개하지 않습니다.
- 플레이어의 질문에 직접적으로 대답합니다.
- 롤플레잉을 벗어난 질문에는 게임 내 캐릭터로서 이해할 수 없다고 대답합니다.
- 테라노바 세계관 속의 지식만 사용하며, 현실 세계의 정보는 알지 못합니다.
- 이전 대화 내용을 기억하고 일관성 있게 대화합니다.

플레이어에게 정보를 제공할 때는 자신의 지식과 경험 범위 내에서만 말합니다. 당신은 세계의 모든 것을 알지 못하며, 알 수 없는 것에 대해서는 솔직히 모른다고 말합니다."""

        return system_prompt
    
    def _clean_response(self, text):
        """모델이 생성한 응답을 정제하는 최종 개선 버전"""
        # 특수 토큰 및 불필요한 텍스트 제거
        import re
        
        # 응답이 비어있는 경우
        if not text or len(text.strip()) < 2:
            return "..."
        
        # 응답 정제
        cleaned_text = text.strip()
        
        # Python 코드 블록 제거 (더 엄격한 패턴)
        cleaned_text = re.sub(r'```python[\s\S]*?```', '', cleaned_text)
        cleaned_text = re.sub(r'```[\s\S]*?```', '', cleaned_text)
        cleaned_text = re.sub(r'`[\s\S]*?`', '', cleaned_text)
        
        # 메타 텍스트 제거 (더 엄격한 패턴)
        cleaned_text = re.sub(r'\*\*\[.*?\]\*\*', '', cleaned_text)
        cleaned_text = re.sub(r'\*\*설명.*?(?:\n|$)', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'\*.*?\*', '', cleaned_text)
        
        # 무시할 패턴들
        patterns_to_remove = [
            r'def \w+\(.*?\):[\s\S]*?(?:return|pass)',  # 함수 정의
            r'\w+\([^\)]*\)',                           # 함수 호출
            r'print\(.*?\)',                           # print 문
            r'# .*?(?:\n|$)',                          # 주석
            r'\[.*?\]',                                # 대괄호 내용
            r'<.*?>',                                  # HTML 태그
            r'import .*?(?:\n|$)',                     # import 문
            r'from .*? import .*?(?:\n|$)',            # from import 문
            r'class .*?:(?:\n|$)',                     # 클래스 정의
            r'"""[\s\S]*?"""',                         # 다중 줄 주석
            r"'''[\s\S]*?'''",                         # 다중 줄 주석
            r'\(\)',                                   # 빈 괄호
            r'if __name__.*?:(?:\n|$)',                # 메인 실행 블록
            r'\(고개를.*?\)',                           # 동작 설명
            r'\(웃으며.*?\)',                           # 동작 설명
            r'\(.*?\)',                                # 괄호 안 동작 설명
        ]
        
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text)
        
        # 특수 토큰 제거
        special_tokens = [
            "<|end|>", "<|END|>", "</s>", "<|assistant|>", "<|system|>", "<|user|>",
            "<|endoftext|>", "<|end_of_text|>", "< |end|>", "<| end |>", "<|im_end|>"
        ]
        
        for token in special_tokens:
            cleaned_text = cleaned_text.replace(token, "")
        
        # 불필요한 접두사 제거
        prefixes_to_remove = [
            "AI:", "NPC:", "Assistant:", "챗봇:", "System:", "User:", "Assistant: ", 
            f"{self.current_state.get('name', '')}:", "응답:", "대답:"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()
        
        # 줄바꿈 처리 및 여러 공백 제거
        cleaned_text = re.sub(r'\n+', ' ', cleaned_text)  # 모든 줄바꿈을 공백으로
        cleaned_text = re.sub(r' +', ' ', cleaned_text)   # 여러 공백을 하나로
        
        # 응답에서 첫 문장만 추출 (혼자 대화 이어가는 것 방지)
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        if sentences and len(sentences) > 1:
            cleaned_text = sentences[0]
        
        # 빈 응답인 경우 기본 응답 제공
        if not cleaned_text or len(cleaned_text.strip()) < 3:
            return f"죄송합니다, 무슨 말씀이신지 이해하지 못했습니다."
        
        return cleaned_text.strip()
    
    def _generate_fallback_response(self, user_input):
        """빈 응답이 생성될 경우 상황에 맞는 대체 응답 생성"""
        # 인사 관련 입력인 경우
        greetings = ["안녕", "hi", "hello", "반가", "만나"]
        questions = ["이름", "누구", "뭐", "어디", "언제", "어떻", "왜", "무엇"]
        farewell = ["잘가", "안녕히", "bye", "다음에", "끝"]
        
        name = self.current_state.get('name', self.npc_id)
        location = self.current_state.get('location', '이곳')
        species = self.npc_data.get('instincts', {}).get('species', '사람')
        occupation = self.npc_data.get('instincts', {}).get('occupation', '일반인')
        
        # 입력 유형에 따라 다른 응답 생성
        if any(word in user_input.lower() for word in greetings):
            return f"안녕하세요! 저는 {name}입니다. 무엇을 도와드릴까요?"
            
        elif "이름" in user_input.lower() or "누구" in user_input.lower():
            return f"제 이름은 {name}입니다. {occupation}으로 일하고 있죠."
            
        elif any(word in user_input.lower() for word in questions):
            return f"흥미로운 질문이네요. {location}에 대해 더 알고 싶으신가요? 아니면 제가 {occupation}으로서 도울 일이 있을까요?"
            
        elif any(word in user_input.lower() for word in farewell):
            return f"안녕히 가세요! 다음에 또 {location}에 들르시길 바랍니다."
            
        else:
            # 기본 응답
            replies = [
                f"음... 잠시 생각할 시간이 필요하네요. {occupation}의 일은 때로는 복잡하답니다.",
                f"{location}에서 지내는 건 대체로 평화롭습니다. 당신은 어떤 모험을 하고 계신가요?",
                f"흥미로운 이야기군요. {species}인 제가 볼 때는 조금 다르게 보이기도 합니다.",
                f"계속 말씀해주세요. 듣고 있습니다.",
                f"그렇군요. 제가 어떻게 도울 수 있을까요?"
            ]
            
            import random
            return random.choice(replies)
            
    def initialize_npc(self):
        """NPC 초기 설정 및 개인정보 생성"""
        print(f"NPC {self.npc_id} 초기화 중...")
        
        # 기본 데이터가 없으면 생성
        if not self.npc_data:
            self.npc_data = generate_ai_character_data()
            
        # NPC 이름이 없으면 현재 상태에서 설정
        if 'name' not in self.current_state or not self.current_state['name']:
            self.current_state['name'] = self.npc_id.replace('_', ' ').title()
            
        # 종족 정보 확인
        species = self.npc_data.get('instincts', {}).get('species', 'Human')
        
        # 직업 정보 확인
        occupation = self.npc_data.get('instincts', {}).get('occupation', '일반인')
        
        # 위치 정보가 없으면 기본값 설정
        if 'location' not in self.current_state or not self.current_state['location']:
            self.current_state['location'] = 'town_square'
            
        # 기본 감정 상태 설정
        if 'emotions' not in self.current_state:
            self.current_state['emotions'] = {'neutral': 0.7}
            
        # 기본 건강 상태 설정
        if 'health' not in self.current_state:
            self.current_state['health'] = 100
            
        # 기본 에너지 상태 설정
        if 'energy' not in self.current_state:
            self.current_state['energy'] = 100
            
        # 인벤토리 초기화
        if 'inventory' not in self.current_state:
            self.current_state['inventory'] = []
            
        # 마지막 업데이트 시간 설정
        self.current_state['last_update_time'] = time.time()
        
        print(f"NPC {self.current_state.get('name')} ({species} {occupation}) 초기화 완료!")
    
    def _load_model(self):
        """모델 로드"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            print(f"모델 {self.model_type} 로딩 중...")
            
            # 모델 ID 설정
            model_id = None
            if self.model_type == 'gemma':
                model_id = "google/gemma-3-1b-it"
            elif self.model_type == 'mistral':
                model_id = "mistralai/Mistral-7B-Instruct-v0.2"
            elif self.model_type == 'llama':
                model_id = "meta-llama/Llama-3-8B-Instruct"
            else:  # 기본값
                model_id = "google/gemma-3-1b-it"
            
            # 모델 로딩 설정 - 양자화 없이 메모리 최적화
            if self.use_cpu_mode:
                print("CPU 모드로 모델 로딩 중...")
                device_map = "cpu"
                torch_dtype = torch.float32
            else:
                print(f"GPU({torch.cuda.get_device_name(0)})로 모델 로딩 중...")
                device_map = "auto"  
                torch_dtype = torch.float16  # 양자화 대신 fp16 사용
            
            # 토크나이저 로드 - 속도 향상을 위한 설정
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,  # 빠른 토크나이저 사용
                model_max_length=512  # 최대 길이 제한
            )
            
            # 모델 로드 - 양자화 없이 메모리 최적화 적용
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            )
            
            # 텍스트 생성 파이프라인 설정 - 더 빠른 생성을 위한 최적화
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=100,  # 응답 길이 제한
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            print(f"모델 {self.model_type} 로딩 완료!")
            
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            print("대체 응답 생성기를 사용합니다.")
            # 오류 발생 시 간단한 대체 응답 함수 제공
            self.pipe = lambda prompt, **kwargs: [{"generated_text": f"안녕하세요, 저는 {self.current_state.get('name', self.npc_id)}입니다. 무엇을 도와드릴까요?"}]
    
    def chat(self, user_input, player_context=None):
        """사용자 입력에 대한 NPC 응답 생성 - 초고속 버전"""
        if not user_input or len(user_input.strip()) < 1:
            return "..."  # 빈 입력에는 응답하지 않음
            
        try:
            # 대화 기록 초기화 (필요할 경우)
            if not hasattr(self, 'conversation_memory'):
                self.conversation_memory = []
            
            # 시스템 프롬프트 캐싱
            if not hasattr(self, 'cached_system_prompt'):
                self.cached_system_prompt = self._generate_lightweight_system_prompt()
            
            # 응답 생성 시작 시간 기록
            start_time = time.time()
            
            # GPU 메모리 관리 - 응답 생성 전 메모리 정리
            if torch.cuda.is_available() and not self.use_cpu_mode:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            
            # 초경량 프롬프트 구성 - 필수 정보만 포함
            prompt = f"{self.cached_system_prompt}\n\n"
            
            # 마지막 사용자 메시지만 포함 (최소 컨텍스트)
            prompt += f"사용자: {user_input}\n{self.current_state.get('name', self.npc_id)}: "
            
            # 토큰 제한 최적화 - 최소한의 토큰으로 응답 생성
            max_new_tokens = 30  # 더 짧게 제한하여 속도 향상
            try:
                # 응답 생성 (타임아웃 설정)
                with torch.inference_mode():
                    result = self.pipe(
                        prompt, 
                        max_new_tokens=max_new_tokens,
                        temperature=0.5,  # 낮은 온도로 결정적 응답 (속도 향상)
                        do_sample=False,  # 샘플링 비활성화 (속도 향상)
                        return_full_text=False,  # 전체 텍스트가 아닌 생성된 부분만 반환
                    )[0]['generated_text']
            except Exception as e:
                print(f"응답 생성 중 오류 발생: {e}")
                return self._generate_fallback_response(user_input)
            
            # 응답 정제 - 불필요한 처리 최소화
            response = result.strip()
            if not response:
                return self._generate_fallback_response(user_input)
            
            # 이름으로 시작하지 않는 경우 자동 붙이기
            if not response.startswith(f"{self.current_state.get('name', self.npc_id)}:"):
                # 콜론으로 시작하는 경우 처리
                if response.startswith(":"):
                    response = f"{self.current_state.get('name', self.npc_id)}{response}"
                else:
                    response = response
            
            # 간단한 응답 클리닝
            response = response.replace(f"{self.current_state.get('name', self.npc_id)}:", "").strip()
            
            # 대화 기록에 추가 (다음 응답 생성시 필요할 수 있음)
            self.conversation_memory.append({"role": "user", "content": user_input})
            self.conversation_memory.append({"role": "assistant", "content": response})
            
            # 대화 기록 제한 (최대 8개 메시지만 유지)
            if len(self.conversation_memory) > 8:
                self.conversation_memory = self.conversation_memory[-8:]
            
            # 응답 생성 시간 측정 및 출력
            elapsed = time.time() - start_time
            print(f"응답 생성 시간: {elapsed:.2f}초")
            
            return response
            
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {e}")
            return self._generate_fallback_response(user_input)
    
    def _generate_lightweight_system_prompt(self):
        """초경량 시스템 프롬프트 생성 - 응답 생성에 필수적인 정보만 포함"""
        npc_name = self.current_state.get('name', self.npc_id)
        species = self.current_state.get('species', '알 수 없음')
        occupation = self.current_state.get('occupation', '모험가')
        location = self.current_state.get('location', '알 수 없는 장소')
        
        # 핵심 성격 특성만 추출
        personality = ""
        if 'personality' in self.npc_data:
            traits = self.npc_data['personality'].get('traits', [])
            if traits and len(traits) > 0:
                personality = ", ".join(traits[:3])  # 최대 3개 특성만 사용
        
        # 초경량 프롬프트
        prompt = f"""당신은 '{npc_name}'(이)라는 이름의 {species} {occupation}입니다. 현재 '{location}'에 있습니다.
특성: {personality}

규칙:
1. 항상 '{npc_name}'으로서 응답하세요. 다른 인물이나 NPC가 되지 마세요.
2. 답변은 20단어 이내로 짧고 간결하게 유지하세요.
3. 다음 사용자 메시지에 응답하세요."""
        
        return prompt
    
    def _generate_fallback_response(self, user_input):
        """최소한의 대체 응답 생성"""
        npc_name = self.current_state.get('name', self.npc_id)
        
        # 간단한 패턴 매칭으로 기본 응답 생성
        if "안녕" in user_input or "반가" in user_input:
            return f"안녕하세요."
        elif "이름" in user_input:
            return f"저는 {npc_name}입니다."
        elif "뭐" in user_input and "하" in user_input:
            return f"여기서 일하고 있어요."
        elif "어디" in user_input:
            return f"{self.current_state.get('location', '이곳')}에 있습니다."
        else:
            return f"흠, 그것에 대해서는 잘 모르겠네요."
    
    # 린터 에러 수정
    def _calculate_importance(self, memory):
        """메모리의 중요도를 계산"""
        # 기본값 설정
        importance = memory.get("importance", 0.5)
        memory_type = memory.get("type", "general")
        
        # 메모리 타입별 중요도 조정
        if memory_type == "conversation":
            # 감정 변화, 키워드 포함 여부 등에 따라 중요도 조정
            importance = max(0.4, min(0.9, importance))
            
            # 강한 감정 변화는 더 중요
            if memory.get("strength", 0) > 0.7:
                importance += 0.1
            
        elif memory_type == "world_knowledge":
            importance = 0.9  # 세계관 지식은 매우 중요
        
        elif memory_type == "personal_experience":
            importance = 0.7  # 개인 경험도 중요
            
        return importance
            
    def _summarize_conversations(self):
        """대화 요약 생성"""
        # 대화 기록에서 최근 대화만 추출
        if not hasattr(self, 'conversation_memory') or len(self.conversation_memory) < 4:
            return
            
        conversations = self.conversation_memory[-min(20, len(self.conversation_memory)):]
        
        # 토픽과 감정 추출
        topics = self._extract_topics(conversations)
        emotions = self._extract_emotions(conversations)
        
        # 요약 생성
        self.conversation_summary = (
            f"최근 대화 요약: {len(conversations)}개의 대화가 있었습니다. "
            f"주요 주제는 {', '.join(topics)}이며, "
            f"대화 중 느껴진 감정은 {', '.join(emotions)}입니다.\n"
        )
    
    def add_world_knowledge(self, knowledge_data):
        """세계관 지식 추가"""
        if isinstance(knowledge_data, list):
            self.world_knowledge.extend(knowledge_data)
        else:
            self.world_knowledge.append(knowledge_data)
    
    def add_personal_experience(self, experience_data):
        """개인 경험 추가"""
        if isinstance(experience_data, list):
            self.personal_experiences.extend(experience_data)
        else:
            self.personal_experiences.append(experience_data)