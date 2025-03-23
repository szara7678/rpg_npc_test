"""
테라노바(Terra Nova) 세계의 역사와 설정 데이터 
"""

# 세계 기본 설정
WORLD_INFO = {
    "name": "테라노바(Terra Nova)",
    "age": 4829,  # 현재 연도
    "calendar": "태양력",
    "moons": 2,   # '아르고'와 '루나' 두 개의 달
    "magic_types": ["원소 마법", "생명 마법", "환영 마법", "소환 마법", "금지된 마법"],
    "main_continents": ["아르가스", "엘도란", "미스티아", "노르헤임"],
    "major_oceans": ["무한의 바다", "폭풍의 바다", "고요한 바다"]
}

# 주요 연대기 및 역사적 사건
WORLD_HISTORY = [
    {
        "type": "world_knowledge",
        "category": "history",
        "title": "세계 창조",
        "year": "태초",
        "content": "태초에 다섯 위대한 신들(크로노스, 가이아, 아쿠아, 이그니스, 아에테르)이 테라노바 세계를 창조했다. 이들은 각각 시간, 대지, 물, 불, 공기의 원소를 다스리는 신들이었다.",
        "importance": 0.9
    },
    {
        "type": "world_knowledge",
        "category": "history",
        "title": "첫 문명의 등장",
        "year": "0~500년",
        "content": "첫 번째 문명인 '아르카디아'가 아르가스 대륙 중앙부에서 번성했다. 이들은 신들로부터 마법의 비밀을 직접 전수받은 최초의 인류였다.",
        "importance": 0.85
    },
    {
        "type": "world_knowledge",
        "category": "history",
        "title": "마법 대전쟁",
        "year": "1203~1215년",
        "content": "아르카디아 마법사들의 오만함이 극에 달해 금지된 생명 창조 마법을 시도했고, 이로 인해 마법 대전쟁이 발발했다. 이 전쟁으로 아르카디아 제국은 멸망하고 세계는 혼돈에 빠졌다.",
        "importance": 0.9
    },
    {
        "type": "world_knowledge",
        "category": "history",
        "title": "암흑의 시대",
        "year": "1215~1500년",
        "content": "마법 대전쟁 이후 마법의 사용이 금지되었고, 300년에 걸친 '암흑의 시대'가 시작되었다. 이 시기에 드워프와 엘프 종족이 인간과 결별하고 각자의 영역으로 떠났다.",
        "importance": 0.8
    },
    {
        "type": "world_knowledge",
        "category": "history",
        "title": "세븐 킹덤의 등장",
        "year": "1500~2000년",
        "content": "암흑의 시대가 끝나고 아르가스 대륙에 일곱 개의 강력한 왕국(세븐 킹덤)이 등장했다. 이들은 서로 동맹과 전쟁을 반복하며 500년간 대륙의 주요 세력으로 자리잡았다.",
        "importance": 0.85
    },
    {
        "type": "world_knowledge",
        "category": "history",
        "title": "마법 르네상스",
        "year": "2100~2400년",
        "content": "마법에 대한 금지가 점차 완화되며 '마법 르네상스' 시대가 시작되었다. 이 시기에 유명한 마법 학교인 '아스트랄 아카데미'가 설립되었고, 마법과 과학의 융합이 시작되었다.",
        "importance": 0.8
    },
    {
        "type": "world_knowledge",
        "category": "history",
        "title": "대항해 시대",
        "year": "2500~2800년",
        "content": "세계 탐험과 새로운 대륙 발견이 활발해진 '대항해 시대'가 시작됐다. 이 시기에 미스티아와 노르헤임 대륙이 발견되었고, 다양한 종족과의 교류가 확대되었다.",
        "importance": 0.75
    },
    {
        "type": "world_knowledge",
        "category": "history",
        "title": "어둠의 침략",
        "year": "3200~3215년",
        "content": "미지의 차원에서 온 '그림자군단'이 세계를 침략한 시기. 모든 종족이 연합하여 15년 간의 전쟁 끝에 그림자군단을 물리쳤으나, 세계는 크게 황폐화되었다.",
        "importance": 0.9
    },
    {
        "type": "world_knowledge",
        "category": "history",
        "title": "황금 시대",
        "year": "3500~4000년",
        "content": "그림자군단 격퇴 후 세계가 재건되며 예술, 문화, 마법, 과학이 크게 발전한 '황금 시대'가 시작되었다. 이 시기에 '크리스탈 기술'이라는 마법과 기술의 융합이 이루어졌다.",
        "importance": 0.8
    },
    {
        "type": "world_knowledge",
        "category": "history",
        "title": "현재: 혼돈의 전조",
        "year": "4800~4829년(현재)",
        "content": "최근 몇 십 년간 세계 곳곳에서 이상 현상이 발생하고 있다. 고대 유적에서 봉인이 풀리고, 잊혀진 마법이 되살아나며, 새로운 영웅들이 등장하는 시대가 시작되고 있다.",
        "importance": 0.95
    }
]

# 주요 국가 및 도시 정보
WORLD_LOCATIONS = [
    {
        "type": "world_knowledge",
        "category": "location",
        "title": "아스트랄 제국",
        "region": "아르가스 대륙 중앙부",
        "content": "테라노바 세계에서 가장 강력한 인간 제국. 마법과 과학이 고도로 발달했으며, 웅장한 수도 '크리스탈리아'는 수백 개의 마법탑으로 유명하다. 황제 '레오니다스 3세'가 통치하고 있다.",
        "importance": 0.9
    },
    {
        "type": "world_knowledge",
        "category": "location",
        "title": "실버우드",
        "region": "엘도란 대륙 북부 숲",
        "content": "고대 엘프들의 왕국. 수천 년 된 거대한 나무들이 울창한 숲을 이루고 있으며, 엘프들은 이 나무 위에 도시를 건설했다. 생명 마법과 자연과의 조화를 중시하는 문화를 가지고 있다.",
        "importance": 0.85
    },
    {
        "type": "world_knowledge",
        "category": "location",
        "title": "드라카니아",
        "region": "미스티아 대륙 화산지대",
        "content": "드래곤 종족과 드래곤본(dragon-born)들이 세운 왕국. 화산과 용암이 흐르는 험난한 지형에 위치하며, 드래곤의 비늘로 만든 건축물이 특징이다. 용의 언어와 화염 마법이 발달했다.",
        "importance": 0.8
    },
    {
        "type": "world_knowledge",
        "category": "location",
        "title": "아이언포지",
        "region": "노르헤임 대륙 산맥 지대",
        "content": "드워프들의 거대한 지하 왕국. 세계 최고의 대장간과 방대한 광산으로 유명하며, 드워프 장인들이 만든 무기와 갑옷은 최고의 품질을 자랑한다. 깊은 산 속에 숨겨진 이 왕국은 외부인의 출입이 제한적이다.",
        "importance": 0.85
    },
    {
        "type": "world_knowledge",
        "category": "location",
        "title": "헤이븐포트",
        "region": "아르가스 대륙 남부 해안",
        "content": "테라노바에서 가장 큰 무역 도시이자 항구. 다양한 종족과 문화가 공존하는 국제도시로, 모든 대륙을 오가는 무역선의 중심지다. 해적과 모험가들이 많이 모이는 곳으로도 유명하다.",
        "importance": 0.8
    },
    {
        "type": "world_knowledge",
        "category": "location",
        "title": "샤도우캐니언",
        "region": "미스티아 대륙 동부 협곡",
        "content": "세계에서 가장 위험한 지역 중 하나로, 그림자군단의 잔존 세력이 아직 남아있다고 알려져 있다. 이곳에서는 낮에도 햇빛이 거의 들지 않으며, 강력한 몬스터들이 서식한다.",
        "importance": 0.75
    },
    {
        "type": "world_knowledge",
        "category": "location",
        "title": "세레니티",
        "region": "아르가스 대륙 중부 평원",
        "content": "화려한 '크리스탈리아'와 달리 소박하고 평화로운 중소 도시. 농업이 발달했으며, 맛있는 음식과 연례 축제로 유명하다. 모험가들이 여정을 시작하기 전 마지막으로 들르는 안전한 도시로 여겨진다.",
        "importance": 0.7
    },
    {
        "type": "world_knowledge",
        "category": "location",
        "title": "미스트레이크",
        "region": "엘도란 대륙 중앙부",
        "content": "항상 안개가 자욱한 신비로운 호수와 그 주변 마을. 이 지역은 이계와의 경계가 얇아 종종 이상한 현상이 발생한다. 예언자와 점술사들이 많이 모여 사는 곳으로, 미래를 보는 의식이 유명하다.",
        "importance": 0.75
    }
]

# 종족 정보
WORLD_RACES = [
    {
        "type": "world_knowledge",
        "category": "race",
        "title": "인간",
        "content": "테라노바에서 가장 흔하고 적응력 높은 종족. 수명은 80~100년 정도이며, 다양한 기술과 문화를 발전시켰다. 모든 유형의 마법에 적성이 있으나 특별히 뛰어난 재능은 없다. 주로 아르가스 대륙에 분포해 있다.",
        "traits": ["적응력", "다재다능", "야망", "짧은 수명", "문화적 다양성"],
        "importance": 0.85
    },
    {
        "type": "world_knowledge",
        "category": "race",
        "title": "엘프",
        "content": "우아하고 장수하는 종족으로, 숲과 자연과의 조화를 중시한다. 수명은 800~1000년에 달하며, 날카로운 감각과 자연 마법에 뛰어난 재능이 있다. 엘도란 대륙의 거대 숲에 주로 거주한다. 인간들과는 복잡한 관계를 유지하고 있다.",
        "traits": ["장수", "자연 친화", "우아함", "예술적 재능", "활과 마법 전문가"],
        "importance": 0.85
    },
    {
        "type": "world_knowledge",
        "category": "race",
        "title": "드워프",
        "content": "땅속 깊은 곳에 거주하는 강인한 종족. 평균 수명은 350년 정도이며, 대장장이와 광부로서의 기술이 뛰어나다. 마법보다는 장인 정신과 기계 공학을 중시하며, 전투에서는 도끼와 망치를 주로 사용한다. 노르헤임 대륙의 산맥 지역에 왕국을 건설했다.",
        "traits": ["강인함", "장인정신", "고집", "충성심", "탁월한 기억력"],
        "importance": 0.8
    },
    {
        "type": "world_knowledge",
        "category": "race",
        "title": "오크",
        "content": "강력한 신체 능력을 가진 호전적 종족. 과거에는 야만적인 부족 생활을 했으나, 최근 세대는 점차 문명화되어 일부는 도시에 정착하기도 했다. 평균 수명은 120년이며, 전투와 생존 기술에 뛰어나다. 미스티아 대륙의 평원 지대에 주로 거주한다.",
        "traits": ["강한 체력", "호전성", "부족 문화", "명예 중시", "단순함"],
        "importance": 0.75
    },
    {
        "type": "world_knowledge",
        "category": "race",
        "title": "드래곤본",
        "content": "드래곤의 피를 이어받은 희귀한 종족. 인간과 비슷한 형태지만 피부는 비늘로 덮여 있으며, 일부는 약한 화염 브레스를 사용할 수 있다. 수명은 150~200년이며, 화염과 관련된 마법에 재능이 있다. 미스티아 대륙의 화산 지대에 주로 거주한다.",
        "traits": ["드래곤 혈통", "화염 저항", "마법 친화성", "자존심", "명예 중시"],
        "importance": 0.8
    },
    {
        "type": "world_knowledge",
        "category": "race",
        "title": "하프링",
        "content": "인간의 절반 정도 크기를 가진 소형 종족. 평화를 사랑하고 농사와 요리에 뛰어난 재능이 있으며, 손재주가 좋다. 평균 수명은 150년 정도이며, 주로 아르가스 대륙의 언덕과 평원 지역에 마을을 이루고 산다. 도둑 기술과 은신에도 타고난 재능이 있다.",
        "traits": ["소형", "영리함", "행운", "손재주", "평화주의"],
        "importance": 0.7
    },
    {
        "type": "world_knowledge",
        "category": "race",
        "title": "티플링",
        "content": "고대에 악마와의 계약으로 인해 악마의 피가 섞인 인간의 후손. 뿔과 꼬리가 있으며, 피부색이 붉거나 보라색인 경우가 많다. 암흑과 화염 마법에 재능이 있으며, 일반 인간과 비슷한 수명을 가진다. 사회적 편견으로 인해 독립적인 생활을 하는 경우가 많으며, 모험가나 용병으로 활동하는 경우가 많다.",
        "traits": ["악마의 혈통", "암흑 친화성", "매력", "독립성", "재치"],
        "importance": 0.75
    }
]

# 직업 및 클래스 정보
WORLD_CLASSES = [
    {
        "type": "world_knowledge",
        "category": "class",
        "title": "전사(Warrior)",
        "content": "전투의 최전선에서 활약하는 무력 전문가. 다양한 무기와 방어구 사용에 능숙하며, 특히 근접 전투에서 탁월한 능력을 발휘한다. 하위 직업으로는 기사(Knight), 바바리안(Barbarian), 팔라딘(Paladin) 등이 있다.",
        "skills": ["무기 숙련", "방어 전문화", "전투 외침", "생존력", "리더십"],
        "importance": 0.85
    },
    {
        "type": "world_knowledge",
        "category": "class",
        "title": "마법사(Mage)",
        "content": "다양한 원소와 현상을 조작하는 마법 전문가. 장기간의 학습과 연구를 통해 강력한 주문을 습득한다. 전투에서는 원거리 마법 공격이 주요 전략이다. 하위 직업으로는 원소술사(Elementalist), 환술사(Illusionist), 시공술사(Chronomancer) 등이 있다.",
        "skills": ["마법 주문", "마법서 연구", "원소 조작", "지식", "지능"],
        "importance": 0.9
    },
    {
        "type": "world_knowledge",
        "category": "class",
        "title": "도적(Rogue)",
        "content": "은밀함과 재빠름을 활용하는 전문가. 함정 탐지와 해제, 자물쇠 따기, 은신 등의 기술에 뛰어나다. 전투에서는 기습 공격과 약점 공략을 주로 사용한다. 하위 직업으로는 암살자(Assassin), 시프(Thief), 스카우트(Scout) 등이 있다.",
        "skills": ["은신", "함정 감지", "재빠른 움직임", "정밀 타격", "교활함"],
        "importance": 0.8
    },
    {
        "type": "world_knowledge",
        "category": "class",
        "title": "사제(Cleric)",
        "content": "신앙의 힘을 활용하는 종교적 전문가. 치유와 버프 마법에 특화되어 있으며, 언데드에 대항하는 신성한 힘을 사용할 수 있다. 하위 직업으로는 힐러(Healer), 주교(Bishop), 오라클(Oracle) 등이 있다.",
        "skills": ["치유 마법", "신성 마법", "축복", "언데드 퇴치", "신앙심"],
        "importance": 0.85
    },
    {
        "type": "world_knowledge",
        "category": "class",
        "title": "레인저(Ranger)",
        "content": "자연과 교감하며 야생에서의 생존에 능한 전문가. 활과 같은 원거리 무기 사용에 특화되어 있으며, 동물을 길들이고 함께 전투할 수 있다. 하위 직업으로는 수렵꾼(Hunter), 비스트마스터(Beast Master), 트래커(Tracker) 등이 있다.",
        "skills": ["야생 생존", "원거리 전투", "동물 교감", "추적", "자연 지식"],
        "importance": 0.75
    },
    {
        "type": "world_knowledge",
        "category": "class",
        "title": "바드(Bard)",
        "content": "음악과 이야기를 통해 마법적 효과를 일으키는 다재다능한 예술가. 동료를 격려하고 적을 혼란에 빠뜨리는 능력이 있으며, 다양한 지식과 사교 기술을 보유하고 있다. 하위 직업으로는 음유시인(Minstrel), 전쟁 음악가(War Chanter), 이야기꾼(Tale Weaver) 등이 있다.",
        "skills": ["음악 마법", "매혹", "다재다능", "지식", "사교술"],
        "importance": 0.7
    },
    {
        "type": "world_knowledge",
        "category": "class",
        "title": "드루이드(Druid)",
        "content": "자연의 힘을 다루는 자연 마법사. 동물로 변신하는 능력과 식물을 조종하는 능력이 있으며, 자연과의 깊은 유대감을 가지고 있다. 하위 직업으로는 쉐이프시프터(Shapeshifter), 스톰키퍼(Storm Keeper), 그린워든(Green Warden) 등이 있다.",
        "skills": ["변신", "자연 마법", "동식물 교감", "치유", "자연 균형 유지"],
        "importance": 0.8
    },
    {
        "type": "world_knowledge",
        "category": "class",
        "title": "연금술사(Alchemist)",
        "content": "물질의 변환과 마법적 약물 제조에 특화된 전문가. 다양한 폭발물, 치료제, 변형 물약 등을 제조할 수 있으며, 희귀한 물질과 재료에 대한 해박한 지식을 가지고 있다. 하위 직업으로는 폭탄 제조사(Bombmaker), 약제사(Apothecary), 변형술사(Transmuter) 등이 있다.",
        "skills": ["물약 제조", "폭탄 제작", "물질 변환", "재료 지식", "실험"],
        "importance": 0.75
    }
]

# 마법 시스템
WORLD_MAGIC = [
    {
        "type": "world_knowledge",
        "category": "magic",
        "title": "원소 마법",
        "content": "불, 물, 땅, 공기의 네 가지 기본 원소를 조작하는 마법. 가장 흔하고 기초적인 마법 유형으로, 대부분의 마법사들이 처음 배우는 마법이다. 전투에서 주로 사용되며, 원소 간의 조합을 통해 다양한 효과를 낼 수 있다.",
        "examples": ["파이어볼", "워터실드", "에어슬래시", "어스퀘이크"],
        "importance": 0.85
    },
    {
        "type": "world_knowledge",
        "category": "magic",
        "title": "생명 마법",
        "content": "생명력을 다루고 조작하는 마법. 치유와 회복에 주로 사용되며, 식물을 성장시키거나 죽은 것을 소생시키는 등의 효과가 있다. 잘못 사용할 경우 생명의 균형을 해칠 수 있어 주의가 필요하다.",
        "examples": ["힐", "리제너레이션", "네이처그로스", "리저렉션"],
        "importance": 0.8
    },
    {
        "type": "world_knowledge",
        "category": "magic",
        "title": "환영 마법",
        "content": "착각과 환상을 만들어내는 마법. 실제로 물리적인 효과는 없지만, 타인의 감각과 인식을 속이는 데 탁월하다. 스파이와 기만 전문가들이 선호하며, 정신에 직접 영향을 주는 고급 기술도 포함한다.",
        "examples": ["인비전", "미러이미지", "디스가이즈", "팬텀포스"],
        "importance": 0.75
    },
    {
        "type": "world_knowledge",
        "category": "magic",
        "title": "소환 마법",
        "content": "다른 차원이나 세계에서 생물이나 물체를 불러오는 마법. 정령, 악마, 천사 등의 존재를 일시적으로 소환하여 도움을 받을 수 있다. 고도의 집중력과 정신력이 필요하며, 소환된 존재와의 계약이 중요하다.",
        "examples": ["서먼 엘리멘탈", "콜 페밀리어", "디멘션 도어", "바인딩 컨트랙트"],
        "importance": 0.8
    },
    {
        "type": "world_knowledge",
        "category": "magic",
        "title": "시공간 마법",
        "content": "시간과 공간을 조작하는 고급 마법. 순간 이동, 시간 지연, 중력 조작 등의 효과가 있으며, 매우 복잡하고 위험해 숙련된 마법사만이 다룰 수 있다. 아스트랄 제국의 시공술사들이 이 분야의 전문가로 알려져 있다.",
        "examples": ["텔레포트", "타임 슬로우", "그래비티 필드", "디멘션 폴드"],
        "importance": 0.9
    },
    {
        "type": "world_knowledge",
        "category": "magic",
        "title": "금지된 마법",
        "content": "생명 창조, 영혼 조작, 사령술 등 윤리적, 자연적 법칙을 위반하는 마법. 대부분의 왕국에서 이러한 마법의 사용은 엄격히 금지되어 있으며, 실험조차 중범죄로 간주된다. 마법 대전쟁의 원인이 되었던 마법 유형이다.",
        "examples": ["네크로맨시", "소울 트랩", "블러드 매직", "언네츄럴 크리에이션"],
        "importance": 0.85
    }
]

# 판타지 세계의 유명 인물들
WORLD_CHARACTERS = [
    {
        "type": "world_knowledge",
        "category": "character",
        "title": "아르토리우스 대제",
        "content": "2500년 전 아르가스 대륙을 통일한 전설적인 황제. '빛의 검'이라 불리는 마법 검을 사용했으며, 그의 통치 시기는 '첫 번째 황금기'로 불린다. 그의 혈통은 현재 아스트랄 제국의 황실로 이어지고 있다.",
        "importance": 0.85
    },
    {
        "type": "world_knowledge",
        "category": "character",
        "title": "메이지 모르가나",
        "content": "마법 르네상스 시대의 가장 위대한 마법사. 아스트랄 아카데미의 설립자이며, 현대 마법 이론의 기초를 세웠다. 500년의 수명을 살았다고 전해지며, 그녀의 마법서는 지금도 귀중한 연구 자료로 사용된다.",
        "importance": 0.85
    },
    {
        "type": "world_knowledge",
        "category": "character",
        "title": "드라칸 제1의 왕",
        "content": "드래곤본 종족의 첫 번째 왕이자 드라카니아 왕국의 설립자. 전설에 따르면 진정한 드래곤의 피를 이어받아 변신 능력이 있었다고 한다. 그의 통치 하에 드라카니아는 강력한 마법 왕국으로 성장했다.",
        "importance": 0.8
    },
    {
        "type": "world_knowledge",
        "category": "character",
        "title": "영웅왕 아론",
        "content": "어둠의 침략 시기에 모든 종족을 통합해 그림자군단에 맞서 싸운 위대한 전사. '빛의 수호자'라는 별명으로도 알려져 있으며, 최후의 전투에서 그림자군단의 지도자와 함께 사라졌다고 전해진다.",
        "importance": 0.9
    },
    {
        "type": "world_knowledge",
        "category": "character",
        "title": "현자 엘드리안",
        "content": "800살이 넘은 현존하는 가장 오래된 엘프 현자. 수많은 역사적 사건을 직접 목격했으며, 실버우드의 고문관으로 활동하고 있다. 예언과 점성술의 대가이며, 미래에 대한 그의 예언은 대부분 적중했다고 알려져 있다.",
        "importance": 0.85
    }
]

# 고대 유물 및 마법 아이템
WORLD_ARTIFACTS = [
    {
        "type": "world_knowledge",
        "category": "artifact",
        "title": "빛의 검",
        "content": "아르토리우스 대제가 사용했던 전설적인 마법 검. 진정한 왕의 피를 이어받은 사람만이 검의 진정한 힘을 발휘할 수 있다고 전해진다. 현재는 아스트랄 제국의 왕실 보물로 보관되어 있다.",
        "importance": 0.8
    },
    {
        "type": "world_knowledge",
        "category": "artifact",
        "title": "세계수의 조각",
        "content": "세계 창조 당시 존재했던 거대한 세계수의 파편으로 만들어진 지팡이. 강력한 자연 마법과 생명력을 지니고 있으며, 현재는 실버우드의 엘프 여왕이 소유하고 있다.",
        "importance": 0.85
    },
    {
        "type": "world_knowledge",
        "category": "artifact",
        "title": "용의 심장",
        "content": "최초의 드래곤 중 하나의 심장을 결정화한 보석. 착용자에게 드래곤의 능력 일부를 부여한다고 알려져 있으며, 화염에 대한 완전한 면역과 비행 능력을 준다고 전해진다. 드라카니아 왕국의 왕관에 박혀 있다.",
        "importance": 0.85
    },
    {
        "type": "world_knowledge",
        "category": "artifact",
        "title": "시간의 모래시계",
        "content": "시간을 제한적으로 조작할 수 있는 고대 유물. 마법 대전쟁 이전 아르카디아 제국에서 만들어졌으며, 매우 위험해 사용이 금지되어 있다. 전설에 따르면 이 모래시계는 시간을 최대 하루까지 되돌릴 수 있다고 한다.",
        "importance": 0.9
    },
    {
        "type": "world_knowledge",
        "category": "artifact",
        "title": "그림자의 가면",
        "content": "착용자를 완전한 그림자로 변환시키는 저주받은 가면. 그림자군단의 지도자가 사용했던 것으로 알려져 있으며, 영웅왕 아론과의 최후 전투 이후 행방불명되었다. 일부 소문에 따르면 이 가면을 찾는 암흑 세력이 있다고 한다.",
        "importance": 0.85
    }
]

# 모든 세계 지식 통합
WORLD_KNOWLEDGE = (
    WORLD_HISTORY +
    WORLD_LOCATIONS +
    WORLD_RACES +
    WORLD_CLASSES +
    WORLD_MAGIC +
    WORLD_CHARACTERS +
    WORLD_ARTIFACTS
)

# NPCBrain 클래스에서 사용할 WORLD_DATA 정의
WORLD_DATA = {
    "info": WORLD_INFO,
    "history": WORLD_HISTORY,
    "locations": WORLD_LOCATIONS, 
    "races": WORLD_RACES,
    "classes": WORLD_CLASSES,
    "magic": WORLD_MAGIC,
    "characters": WORLD_CHARACTERS,
    "artifacts": WORLD_ARTIFACTS
} 