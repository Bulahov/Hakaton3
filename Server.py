#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import logging
from dotenv import load_dotenv
from RAG import AdvancedRAG
from flask import Flask, request, jsonify


# In[ ]:


# Создаём папку logs
os.makedirs("logs", exist_ok=True)

# Создаём логгер
bot_logger = logging.getLogger("server_log")
bot_logger.setLevel(logging.INFO)

# Создаём файловый хендлер
file_handler = logging.FileHandler("logs/server.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Формат логов
formatter = logging.Formatter(
    "%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)

# Привязываем хендлер
bot_logger.addHandler(file_handler)


# In[7]:


rag = AdvancedRAG()
app = Flask(__name__)
load_dotenv()
host = os.getenv("HOST")
port = os.getenv("PORT")

@app.route("/health", methods=["GET"])
def health():
    print("[INFO] /health вызван")
    bot_logger.info("/health вызван")
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    print(f"[INFO] /ask вызван с данными: {data}")
    bot_logger.info(f"/ask вызван с данными: {data}")

    question = data.get("question", "")
    if not question:
        print("[WARN] Параметр 'question' отсутствует")
        bot_logger.error("Параметр 'question' отсутствует")
        return jsonify({"error": "Параметр 'question' отсутствует"}), 400

    answer, sources = rag.ask(question)
    print(f"[INFO] Ответ для вопроса '{question}': {answer}, Источники: {sources}")
    bot_logger.info(f"Ответ для вопроса '{question}': {answer}, Источники: {sources}")

    return jsonify({
        "answer": answer,
        "sources": sources
    })

@app.route("/generate_questions", methods=["POST"])
def generate_questions():
    data = request.get_json()
    print(f"[INFO] /generate_questions вызван с данными: {data}")
    bot_logger.info(f"/generate_questions вызван с данными: {data}")

    topic = data.get("topic", "")
    questions = rag.generate_questions(topic)
    print(f"[INFO] Сгенерированные вопросы для темы '{topic}': {questions}")
    bot_logger.info(f"Сгенерированные вопросы для темы '{topic}': {questions}")

    return jsonify({"questions": questions})


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    print(f"[INFO] /recommend вызван с данными: {data}")
    bot_logger.info(f"/recommend вызван с данными: {data}")

    topic = data.get("topic", "")
    materials = rag.recommend_materials(topic)
    print(f"[INFO] Рекомендованные материалы для темы '{topic}': {materials}")
    bot_logger.info(f"Рекомендованные материалы для темы '{topic}': {materials}")

    return jsonify({"materials": materials})


@app.route("/upload", methods=["POST"])
def upload_file():
    print("[/upload] Получен запрос на загрузку файлов")
    bot_logger.info("[/upload] Получен запрос на загрузку файлов")

    # Получаем ВСЕ файлы
    files = request.files.getlist("file")

    if not files or len(files) == 0:
        bot_logger.error("[/upload] Файлы НЕ найдены")
        print("[/upload] Файлы НЕ найдены")
        return {"status": "error", "message": "Файлы не были переданы"}, 400

    print(f"[/upload] Найдено файлов: {len(files)}")

    saved_file_paths = []   # сюда соберём пути сохранённых файлов для RAG
    save_dir = "uploads"
    os.makedirs(save_dir, exist_ok=True)

    for file in files:
        print(f"\n[/upload] Обработка файла: {file.filename}")
        bot_logger.info(f"\n[/upload] Обработка файла: {file.filename}")
        print(f"[/upload] MIME: {file.mimetype}")

        # Проверка расширения (только .txt)
        if not file.filename.lower().endswith(".txt"):
            print(f"[/upload] Файл {file.filename} отклонён — неверный формат")
            bot_logger.error(f"[/upload] Файл {file.filename} отклонён — неверный формат")
            return {
                "status": "error",
                "message": f"Файл {file.filename} должен быть формата .txt"
            }, 400

        # Путь сохранения
        file_path = os.path.join(save_dir, file.filename)

        # Сохраняем файл
        file.save(file_path)
        print(f"[/upload] Файл сохранён: {file_path}")
        bot_logger.info(f"[/upload] Файл сохранён: {file_path}")

        saved_file_paths.append(file_path)

    print("\n[/upload] Все файлы сохранены.")
    bot_logger.info("\n[/upload] Все файлы сохранены.")
    print(f"[/upload] Массив файлов: {saved_file_paths}")


    rag.load_documents(saved_file_paths)
    
    # Примерный функционал RAG-модели
    # rag.clean_text(["raw_doc1", "raw_doc2"])
    # rag.chunk_documents(["cleaned_doc1", "cleaned_doc2"])
    # rag.create_embeddings(["chunk1", "chunk2", "chunk3"])
    # rag.load_embeddings(["embeddings/embeddings.npy"])

    return {
        "status": "ok",
        "files": saved_file_paths,
        "message": "Файлы успешно загружены и обработаны"
    }, 200

    

if __name__ == "__main__":
    bot_logger.info(f"=== Запуск Flask API на {host}:{port} ===")
    print(f"=== Запуск Flask API на {host}:{port} ===")
    app.run(host=host, port=port, debug=True)
    print("=== Flask API остановлен ===")
    bot_logger.info("=== Flask API остановлен ===")

