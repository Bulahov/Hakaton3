#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from sentence_transformers import CrossEncoder
# import torch

# # Укажите путь к вашей локальной папке с реранкером
# RERANKER_PATH = r"bge_model" 
# # Или название модели, если качаете из интернета
# # RERANKER_PATH = "BAAI/bge-reranker-v2-m3"

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Загружаю модель из: {RERANKER_PATH}")

# model = CrossEncoder(RERANKER_PATH, device=device)

# # Два примера: один явно подходит, второй - нет
# query = "Как создать виртуальную машину?"
# good_doc = "Для создания виртуальной машины перейдите в раздел Compute Cloud и нажмите Создать."
# bad_doc = "Рецепт борща: возьмите свеклу, капусту и картофель."

# scores = model.predict([
#     [query, good_doc],
#     [query, bad_doc]
# ])

# print(f"\nScore хороший документ: {scores[0]:.4f}")
# print(f"Score плохой документ: {scores[1]:.4f}")

# if scores[0] > scores[1] and scores[0] > 0:
#     print("\n✅ ВСЁ РАБОТАЕТ! Предупреждение можно игнорировать.")
# else:
#     print("\n❌ МОДЕЛЬ СЛОМАНА. Она не различает тексты. Нужно перекачать.")


# In[2]:


import os
import re
import json
import logging
import numpy as np
from typing import List, Dict
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from openai import OpenAI


# Создаём папку logs
os.makedirs("logs", exist_ok=True)

# Создаём логгер
bot_logger = logging.getLogger("rag_log")
bot_logger.setLevel(logging.INFO)

# Создаём файловый хендлер
file_handler = logging.FileHandler("logs/rag.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Формат логов
formatter = logging.Formatter(
    "%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)

# Привязываем хендлер
bot_logger.addHandler(file_handler)


# --- КОНФИГУРАЦИЯ ---
BASE_DIR = "NeuroEmotions_hackathon3_cloud_ru_data2"
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db", "chroma_db_e5_correct")
# Файл с полными текстами нужен для BM25
RAG_JSON_PATH = os.path.join(BASE_DIR,"rag_ready", "NeuroEmotions_all_chunks_for_rag_1.6.json")
FULL_TUTORIALS_PATH = os.path.join(BASE_DIR, "json", "NeuroEmotions_all_tutorials_1.6.json")
# Ключи и модели
load_dotenv()
# CLOUD_RU_API_KEY = os.getenv("CLOUD_RU_API_KEY")
CLOUD_RU_API_KEY = ""
CLOUD_RU_BASE_URL = "https://foundation-models.api.cloud.ru/v1"
CLOUD_RU_MODEL = "ai-sage/GigaChat3-10B-A1.8B"
#Ембеддер и реранкер
EMBEDDING_MODEL = "multilingual-e5-small"
#RERANKER_MODEL = "bge_model"
#RERANKER_MODEL = "reranker_model_mini"
RERANKER_MODEL = "reranker_model_tiny"

#Класс RAGа
class AdvancedRAG:
    def __init__(self):
        print("🚀 Инициализация Advanced RAG System...")

        # 1. Подключение к API LLM
        self.llm_client = OpenAI(api_key=CLOUD_RU_API_KEY, base_url=CLOUD_RU_BASE_URL)

        # 2. Векторный поиск (ChromaDB + E5)
        print("📦 Загрузка ChromaDB и E5...")
        self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.collection = self.chroma_client.get_collection("neuroemotions_e5_correct")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # 3. Поиск по ключевым словам (BM25)
        print("📖 Индексация BM25 (это займет пару секунд)...")
        self.documents_cache = [] # Храним тут тексты и метаданные
        self._init_bm25()

        # 4. Reranker
        print("⚖️ Загрузка Reranker (Cross-Encoder)...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # 5. Инициализация справочника кодов (Code Registry)
        print("🧩 Загрузка справочника кодов...")
        self._init_code_registry()

        print("✅ Система готова к работе!")

    def _init_bm25(self):
        """Загружаем тексты и строим индекс BM25"""
        with open(RAG_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.documents_cache = data # Сохраняем в памяти, чтобы доставать метаданные

        # токенизация: разбиваем по пробелам
        tokenized_corpus = [doc['text'].lower().split(" ") for doc in data]
        self.bm25 = BM25Okapi(tokenized_corpus)
    def _init_code_registry(self):
        """
        Создаем быстрый словарь: {URL -> [Список блоков кода]}
        Чтобы быстро менять [[CODE_BLOCK_N]] на реальный код.
        """
        self.code_registry = {}
        try:
            with open(FULL_TUTORIALS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # data['tutorials'] содержит список статей
                tutorials = data.get('tutorials', [])

            for tut in tutorials:
                url = tut['metadata']['url']
                # Сохраняем список блоков кода для этого URL
                self.code_registry[url] = tut.get('code_blocks', [])

            print(f"   Загружено кодов для {len(self.code_registry)} статей.")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки справочника кодов: {e}")
            print("   Функция восстановления кода работать не будет.")

    def _restore_code_blocks(self, text: str, url: str) -> str:
        """
        Ищет [[CODE_BLOCK_N]] и заменяет на реальный код.
        """
        if "[[CODE_BLOCK_" not in text:
            return text

        # Получаем список кодов для этой статьи
        code_blocks = self.code_registry.get(url, [])
        if not code_blocks:
            return text

        # Функция замены для регулярного выражения
        def replace_match(match):
            try:
                index = int(match.group(1)) # Получаем число N из CODE_BLOCK_N
                if 0 <= index < len(code_blocks):
                    block = code_blocks[index]
                    code_content = block['text']
                    lang = block.get('language', '')
                    # Форматируем красиво для LLM
                    return f"\n```{lang}\n{code_content}\n```\n"
                return match.group(0) # Если индекса нет, оставляем как есть
            except:
                return match.group(0)

        # Заменяем все вхождения [[CODE_BLOCK_(\d+)]]
        restored_text = re.sub(r'\[\[CODE_BLOCK_(\d+)\]\]', replace_match, text)
        return restored_text

    def hybrid_search(self, query: str, top_k_vector=10, top_k_keyword=10) -> List[Dict]:
        """
        Шаг 1: Гибридный поиск (Vector + Keyword)
        """
        # --- А. Векторный поиск ---
        query_vec = self.embedder.encode([f"query: {query}"]).tolist()
        vec_res = self.collection.query(query_embeddings=query_vec, n_results=top_k_vector)

        vector_candidates = []
        if vec_res['documents']:
            for i in range(len(vec_res['documents'][0])):
                vector_candidates.append({
                    'text': vec_res['documents'][0][i],
                    'metadata': vec_res['metadatas'][0][i],
                    'source': 'vector'
                })

        # --- Б. Поиск по ключевым словам (BM25) ---
        tokenized_query = query.lower().split(" ")
        # BM25 возвращает тексты, нам нужно найти их индексы или метаданные
        # rank_bm25 не возвращает индексы напрямую, поэтому:
        # Получаем топ N лучших индексов
        scores = self.bm25.get_scores(tokenized_query)
        top_n_indexes = np.argsort(scores)[::-1][:top_k_keyword]

        keyword_candidates = []
        for idx in top_n_indexes:
            doc = self.documents_cache[idx]
            keyword_candidates.append({
                'text': doc['text'],
                'metadata': doc['metadata'],
                'source': 'bm25'
            })

        # --- В. Объединение и дедупликация ---
        # Используем текст как ключ уникальности
        unique_docs = {}

        for doc in vector_candidates + keyword_candidates:
            # Убираем дубли (хэш от текста)
            doc_hash = hash(doc['text'])
            if doc_hash not in unique_docs:
                unique_docs[doc_hash] = doc

        print(f"   🔍 Hybrid Search: Найдено {len(unique_docs)} кандидатов (Vec={len(vector_candidates)}, BM25={len(keyword_candidates)})")
        return list(unique_docs.values())

    def rerank(self, query: str, candidates: List[Dict], top_k=5) -> List[Dict]:
        """
        Шаг 2: Переранжирование кандидатов
        """
        if not candidates:
            return []

        # Готовим пары [Query, Document Text]
        pairs = [[query, doc['text']] for doc in candidates]

        # Получаем оценки релевантности
        scores = self.reranker.predict(pairs)

        # Добавляем оценки к документам
        for i, doc in enumerate(candidates):
            doc['score'] = scores[i]

        # Сортируем по убыванию оценки
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)

        # Берем топ-K
        final_results = sorted_candidates[:top_k]

        print(f"   ⚖️ Reranker: Выбрано топ-{top_k} лучших. Лучший score: {final_results[0]['score']:.4f}")
        return final_results

    def generate_with_check(self, query: str, context: List[Dict]) -> str:
        """
        Шаг 3 и 4: Генерация с самопроверкой
        """
        # Сборка контекста
        context_str = ""
        for i, item in enumerate(context, 1):
            meta = item['metadata']
            raw_text = item['text']
            url = meta.get('source_url', '')

            # Восстанавливаем реальный код вместо [[CODE_BLOCK]]
            clean_text = self._restore_code_blocks(raw_text, url)

            # Формируем атрибуты для тега <doc>
            attrs = [f'id="{i}"', f'source="{meta.get("source_title", "Unknown")}"']

            if 'category' in meta:
                attrs.append(f'category="{meta["category"]}"')

            # Собираем тег
            attr_str = " ".join(attrs)
            context_str += f'<{attr_str}>\n{clean_text}\n</doc>\n'

        # --- ГЕНЕРАЦИЯ ОТВЕТА ---
        system_prompt = """
Ты — AI-репетитор по платформе Cloud.ru Evolution.
Твоя цель — помогать студентам осваивать облачные технологии, объяснять термины и давать примеры кода.

ПРАВИЛА ОТВЕТА:
1. Используй ТОЛЬКО предоставленный Контекст. Не придумывай функции, которых нет в Cloud.ru.
2. Если в контексте есть примеры кода, ОБЯЗАТЕЛЬНО включи их в ответ.
3. Отвечай структурированно, используй Markdown (жирный шрифт, списки, блоки кода).
4. Если информации нет в контексте, так и скажи: "В моих материалах нет ответа на этот вопрос".
5. Тон: Дружелюбный, профессиональный, педагогический.
6. Перед тем как дать финальный ответ или код, выпиши в начале точную цитату из контекста, на которой основан твой ответ. Начинай ответ со слов: 'Анализ контекста: ...
"""
        user_prompt = f"Вопрос: {query}\n\nКонтекст:\n{context_str}"

        print("   🤖 Генерирую черновик ответа...")
        draft_response = self._call_llm(system_prompt, user_prompt)

        # --- САМОПРОВЕРКА (SELF-CORRECTION) ---
        print("   🕵️ Самопроверка ответа...")
        verify_prompt = f"""
Твоя задача — проверить ответ на наличие галлюцинаций.
Контекст:
{context_str}

Вопрос пользователя: {query}
Черновик ответа: {draft_response}

Инструкция:
1. Проверь, все ли факты в "Черновике ответа" подтверждаются "Контекстом".
2. Если ответ содержит информацию, которой НЕТ в контексте (например, выдуманные команды, параметры) — ИСПРАВЬ его.
3. Если ответ верный и основан на контексте — верни его БЕЗ изменений.
4. Верни ТОЛЬКО финальный исправленный текст ответа.
"""
        final_response = self._call_llm("Ты — строгий редактор технической документации.", verify_prompt)

        return final_response

    def _call_llm(self, sys_prompt, user_prompt):
        try:
            response = self.llm_client.chat.completions.create(
                model=CLOUD_RU_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, # Низкая температура для точности
                max_tokens=5000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Ошибка LLM: {e}"

    def ask(self, query: str):
        print(f"\nВопрос: {query}")
        bot_logger.info(f"Вопрос: {query}")

        # 1. Гибридный поиск (Recall)
        # Берем по 15 кандидатов от каждого метода, чтобы было из чего выбирать
        candidates = self.hybrid_search(query, top_k_vector=15, top_k_keyword=15)
        bot_logger.info(f"Отобраны кандидаты")

        if not candidates:
            return "К сожалению, ничего не найдено."

        # 2. Reranking
        # Оставляем 5 самых лучших кусков
        best_docs = self.rerank(query, candidates, top_k=5)
        bot_logger.info(f"Отобраны лучшие куски")

        # Для отладки показываем, что выбрали
        print(f"   🏆 Топ документ: {best_docs[0]['metadata'].get('source_title')} (Score: {best_docs[0]['score']:.2f})")

        # 3. Сбор источников
        # Мы собираем уникальные ссылки из топ-5 документов
        sources = []
        seen_urls = set()
        for doc in best_docs:
            title = doc['metadata'].get('source_title', 'Документ')
            url = doc['metadata'].get('source_url', '#')
            category = doc['metadata'].get('category', '')

            if url not in seen_urls and url != '#':
                sources.append({
                    "title": title,
                    "url": url,
                    "category": category
                })
                seen_urls.add(url)
        bot_logger.info(f"Произведен сбор источников")

        # 3. Генерация + Проверка
        answer = self.generate_with_check(query, best_docs)
        bot_logger.info(f"Произведена генерация + проверка")

        return answer, sources

        
    def generate_questions(self, question: str, n: int = 3) -> List[str]:
        """
        Генерирует n вопросов для самопроверки на основе исходного вопроса.

        Args:
            question (str): Исходный вопрос пользователя.
            n (int): Количество генерируемых вопросов (по умолчанию 3).

        Returns:
            List[str]: Список из n вопросов для самопроверки.
        """
        print(f"\n❓ Генерация {n} вопросов для самопроверки по теме: {question}")

        # 1. Получаем релевантный контекст (аналогично ask)
        candidates = self.hybrid_search(question, top_k_vector=10, top_k_keyword=10)
        if not candidates:
            return [f"Не удалось сгенерировать вопросы по теме: {question}"]

        best_docs = self.rerank(question, candidates, top_k=3)

        # Собираем контекст (без восстановления кода, если не нужно — но лучше с ним)
        context_str = ""
        for i, item in enumerate(best_docs, 1):
            meta = item['metadata']
            raw_text = item['text']
            url = meta.get('source_url', '')
            clean_text = self._restore_code_blocks(raw_text, url)
            context_str += f"<doc id=\"{i}\" source=\"{meta.get('source_title', 'Unknown')}\">\n{clean_text}\n</doc>\n"

        # 2. Формируем промпт для генерации вопросов
        system_prompt = (
            "Ты — опытный преподаватель по облачным технологиям Cloud.ru. "
            "Твоя задача — сгенерировать вопросы для самопроверки знаний студента. "
            "Вопросы должны быть чёткими, охватывать ключевые понятия из контекста и помогать закрепить материал. "
            "Формулируй вопросы в стиле экзаменационных или учебных заданий."
        )

        user_prompt = (
            f"Исходный вопрос студента: \"{question}\"\n\n"
            f"Контекст:\n{context_str}\n\n"
            f"На основе этого контекста сгенерируй ровно {n} вопросов для самопроверки. "
            f"Каждый вопрос должен быть с новой строки и начинаться с «- » (минус и пробел). "
            f"Не добавляй пояснений, только список вопросов."
        )

        # 3. Вызываем LLM
        try:
            response = self.llm_client.chat.completions.create(
                model=CLOUD_RU_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,  # немного креативности
                max_tokens=500
            )
            raw_output = response.choices[0].message.content.strip()
        except Exception as e:
            return [f"Ошибка генерации вопросов: {e}"]

        # 4. Парсим ответ: ожидаем список в формате "- Вопрос 1\n- Вопрос 2..."
        questions = []
        for line in raw_output.split("\n"):
            if line.strip().startswith("- "):
                q = line.strip()[2:].strip()
                if q:
                    questions.append(q)
            # Также поддерживаем нумерованный список на случай отклонения от формата
            elif line.strip() and line.strip()[0].isdigit() and ". " in line:
                q = line.split(". ", 1)[1].strip()
                if q:
                    questions.append(q)

        # Если не удалось распарсить — вернём как есть, разбив по строкам
        if not questions:
            questions = [q.strip() for q in raw_output.split("\n") if q.strip()]

        # Ограничиваем до n вопросов
        return questions[:n]


    def recommend_materials(self, topic: str, n: int = 3) -> List[Dict[str, str]]:
        """
        Рекомендует n релевантных учебных материалов по заданной теме.

        Args:
            topic (str): Тема, по которой нужны материалы.
            n (int): Количество рекомендуемых источников (по умолчанию 3).

        Returns:
            List[Dict[str, str]]: Список словарей с ключами 'title', 'url', 'category'.
        """
        print(f"\n📚 Поиск рекомендуемых материалов по теме: {topic}")

        # 1. Гибридный поиск кандидатов
        candidates = self.hybrid_search(topic, top_k_vector=15, top_k_keyword=15)
        if not candidates:
            return []

        # 2. Переранжирование для повышения релевантности
        best_docs = self.rerank(topic, candidates, top_k=n)

        # 3. Формирование списка источников (аналогично ask)
        recommended = []
        seen_urls = set()

        for doc in best_docs:
            meta = doc['metadata']
            title = meta.get('source_title', 'Без названия')
            url = meta.get('source_url', '')
            category = meta.get('category', 'Общее')

            # Пропускаем дубли по URL
            if url in seen_urls or not url or url == '#':
                continue

            recommended.append({
                'title': title,
                'url': url,
                'category': category
            })
            seen_urls.add(url)

            if len(recommended) >= n:
                break

        # Если не набралось n уникальных — дополним из кандидатов без rerank (если нужно)
        if len(recommended) < n:
            for doc in candidates:
                if len(recommended) >= n:
                    break
                meta = doc['metadata']
                url = meta.get('source_url', '')
                if url in seen_urls or not url or url == '#':
                    continue
                recommended.append({
                    'title': meta.get('source_title', 'Без названия'),
                    'url': url,
                    'category': meta.get('category', 'Общее')
                })
                seen_urls.add(url)

        return recommended[:n]


# --- ЗАПУСК ---
if __name__ == "__main__":
    if "ВАШ_КЛЮЧ" in CLOUD_RU_API_KEY:
        print("⚠️ ОШИБКА: Вставьте API KEY!")
    else:
        bot = AdvancedRAG()

        # Сложный вопрос, где нужен и код, и термины
        q = "Как выполнить инференс модели на собственных изображениях?"

        ans = bot.ask(q)
        print("\n" + "="*50)
        print("🎓 ФИНАЛЬНЫЙ ОТВЕТ:")
        print(ans['answer'])
        print("\n📚 ИСТОЧНИКИ:")
        for src in ans['sources']:
            print(f"🔗 {src['title']} ({src['category']})")
            print(f"   {src['url']}")
        print("="*50)

        questions = bot.generate_questions("Как создать виртуальную машину в Cloud.ru?", n=5)
        for q in questions:
            print(f"- {q}")

        materials = bot.recommend_materials("Работа с объектным хранилищем в Cloud.ru", n=5)
        for m in materials:
            print(f"- {m['title']} ({m['category']})\n  {m['url']}\n")

