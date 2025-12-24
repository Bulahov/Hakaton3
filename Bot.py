#!/usr/bin/env python
# coding: utf-8

# ## Зависимости

# In[3]:


# !pip install python-telegram-bot --upgrade
# !pip install python-dotenv


# ## Библиотеки

# In[ ]:


import requests
import os
import re
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)





# После существующих импортов добавить:
from config.positions import get_position_config
from utils.auth import validate_email, parse_specialty_from_email, create_new_user
from utils.progress_manager import (
    load_user_data, save_user_data, get_skill_progress, 
    update_progress, format_progress_bar, add_personal_skill,
    get_all_user_skills
)
from utils.keyboards import (
    get_mode_selection_keyboard, 
    get_email_selection_keyboard,
    get_skill_selection_keyboard,
    get_skillup_active_keyboard,
    get_normal_mode_keyboard
)
from utils.recommendation_engine import get_recommendations







# Автоформатирование для markdown

def convert_headers_to_bold(text: str) -> str:
    """Заменяет # Заголовок → **Заголовок**, только вне блоков кода."""
    def process_part(part: str) -> str:
        if part.startswith("```") and part.endswith("```"):
            return part
        return re.sub(r'^#{1,6}\s+(.*)$', r'**\1**', part, flags=re.MULTILINE)
    
    parts = re.split(r'(```(?:[^`]|`[^`]|``[^`])*```)', text, flags=re.DOTALL)
    return "".join(process_part(p) for p in parts)

def fix_list_asterisks(text: str) -> str:
    """Заменяет одиночные * в начале строки (списки) на - или •, чтобы не ломать жирный шрифт."""
    def process_part(part: str) -> str:
        if part.startswith("```") and part.endswith("```"):
            return part
        # Заменяем * пункт → - пункт
        return re.sub(r'^\*\s+', '- ', part, flags=re.MULTILINE)
    
    parts = re.split(r'(```(?:[^`]|`[^`]|``[^`])*```)', text, flags=re.DOTALL)
    return "".join(process_part(p) for p in parts)

def prepare_markdown_text(text: str) -> str:
    """Полная предобработка текста для parse_mode='Markdown'."""
    text = convert_headers_to_bold(text)
    text = fix_list_asterisks(text)
    return text

async def send_markdown_chunks_safe(update: Update, text: str, max_length: int = 4000):
    """
    Отправляет текст с parse_mode='Markdown', разбивая на чанки,
    не разрезая блоки кода.
    """
    # 1. Предобработка
    text = prepare_markdown_text(text)

    # 2. Разбиваем на блоки: код и не-код
    blocks = re.split(r'(```(?:[^`]|`[^`]|``[^`])*```)', text, flags=re.DOTALL)
    
    chunks = []
    current = ""

    for block in blocks:
        if block.startswith("```") and block.endswith("```"):
            # Блок кода
            if len(current) + len(block) <= max_length:
                current += block
            else:
                if current.strip():
                    chunks.append(current)
                    current = ""
                # Даже если блок большой — добавляем целиком (Telegram примет)
                chunks.append(block)
        else:
            # Обычный текст — разбиваем по абзацам
            paragraphs = [p for p in block.split("\n\n") if p.strip()]
            for para in paragraphs:
                candidate = current + ("\n\n" if current else "") + para
                if len(candidate) > max_length:
                    if current.strip():
                        chunks.append(current)
                    current = para
                else:
                    current = candidate

    if current.strip():
        chunks.append(current)

    # 3. Отправка
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) > 4096:
            chunk = chunk[:4096]
        await update.message.reply_text(
            chunk,
            parse_mode="Markdown",
            disable_web_page_preview=True
        )
        await asyncio.sleep(0.2)


# ## Логи

# In[ ]:


# Создаём папку logs
os.makedirs("logs", exist_ok=True)

# Создаём логгер
bot_logger = logging.getLogger("bot_log")
bot_logger.setLevel(logging.INFO)

# Создаём файловый хендлер
file_handler = logging.FileHandler("logs/bot.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Формат логов
formatter = logging.Formatter(
    "%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)

# Привязываем хендлер
bot_logger.addHandler(file_handler)


# ## Загружаем конфиг

# In[ ]:


load_dotenv()  # Загружаем конфиг из .env

# Состояние пользователей
USER_STATE = {}  # {user_id: {"mode": "skillup|normal", "email": "..."}}

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL")


# ## Основная логика обработки текста

# In[ ]:


VALID_COMMANDS = ["/ask", "/generate_questions", "/recommend"]

def parse_user_message(text: str):
    text = text.strip()

    # Если начинается с команды
    if text.startswith("/"):
        parts = text.split(" ", 1)
        command = parts[0]

        if command in VALID_COMMANDS:
            payload = parts[1] if len(parts) > 1 else ""
            return command, payload

        # неизвестная команда
        return "error", f"Неизвестная команда: {command}"

    # Если команды нет — используем /ask
    return "/ask", text

# Основная логика обработки текста
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
  
  
  
  
  
    user_id = update.effective_user.id
    user_text = update.message.text
    
    # Инициализация состояния
    if user_id not in USER_STATE:
        USER_STATE[user_id] = {"mode": None, "email": None, "active_skill": None}
    
    state = USER_STATE[user_id]
    
    # === ВЫБОР РЕЖИМА ===
    if user_text == "💬 Обычный режим":
        state["mode"] = "normal"
        state["email"] = None
        state["active_skill"] = None
        keyboard = get_normal_mode_keyboard()
        
        help_text = "💬 Обычный режим\n\n"
        help_text += "Задавайте вопросы по документации Cloud.ru или введите /help\n\n"
        help_text += "Доступные команды:\n\n"
        help_text += "/health - Проверка работоспособности сервера\n"
        help_text += "/ask - Задайте вопрос. Возвращает ответ и источники\n"
        help_text += "/generate_questions - Генерация вопросов по теме\n"
        help_text += "/recommend - Рекомендации по теме. Возвращает список материалов\n\n"
        help_text += "Примеры:\n"
        help_text += "• Расскажи про облачные хранилища\n"
        help_text += "• /ask Что такое виртуализация?\n"
        help_text += "• /generate_questions Машинное обучение\n"
        help_text += "• /recommend Основы SQL"
        
        await update.message.reply_text(help_text, reply_markup=keyboard)
        return
    
    elif user_text == "🎓 SkillUp режим":
        state["mode"] = "skillup"
        state["active_skill"] = None
        await ask_for_email(update, context)
        return
    
    elif user_text == "🔄 Сменить режим":
        state["mode"] = None
        state["email"] = None
        state["active_skill"] = None
        keyboard = get_mode_selection_keyboard()
        await update.message.reply_text(
            "Выберите режим:",
            reply_markup=keyboard
        )
        return
    
    # === СМЕНА ПОЛЬЗОВАТЕЛЯ ===
    elif user_text == "👤 Сменить пользователя":
        state["email"] = None
        state["active_skill"] = None
        await ask_for_email(update, context)
        return
    
    # === СМЕНА НАВЫКА ===
    elif user_text == "🔄 Сменить навык":
        if state.get("email"):
            await show_skill_selection(update, context)
        return
    
    # === ОБРАБОТКА ВЫБОРА EMAIL ===
    if state.get("waiting_for_email"):
        await handle_email_selection(update, context)
        return
    
    # === ОБРАБОТКА ВЫБОРА НАВЫКА ===
    if state.get("waiting_for_skill"):
        await handle_skill_selection(update, context)
        return
    
    # === SKILLUP РЕЖИМ С АКТИВНЫМ НАВЫКОМ ===
    if state.get("mode") == "skillup" and state.get("email") and state.get("active_skill"):
        await handle_skillup_with_active_skill(update, context)
        return
    
    # === ОБЫЧНЫЙ РЕЖИМ (СУЩЕСТВУЮЩИЙ КОД) ===
    if state.get("mode") == "normal" or not state.get("mode"):
        # ОСТАВЛЯЕМ СУЩЕСТВУЮЩИЙ КОД БЕЗ ИЗМЕНЕНИЙ
        
        # ... весь остальной существующий код ...
        
        # [ЗДЕСЬ КОПИРУЕМ ВЕСЬ КОД ИЗ СУЩЕСТВУЮЩЕГО handle_message]
        # Начиная с:
        #   command, payload = parse_user_message(user_text)
        #   bot_logger.info(f"Команда после парсинга: {command}, payload: '{payload}'")
        # ...
        # До конца функции
        
        
        
        
        
        # === ОБРАБОТКА КНОПОК ОБЫЧНОГО РЕЖИМА ===
        if user_text == "Ask":
            state["waiting_for_normal_ask"] = True
            await update.message.reply_text("📝 Введите ваш вопрос:")
            return
        
        elif user_text == "Generate Questions":
            state["waiting_for_normal_generate"] = True
            await update.message.reply_text("📝 Введите тему для генерации вопросов:")
            return
        
        elif user_text == "Recommend":
            state["waiting_for_normal_recommend"] = True
            await update.message.reply_text("📝 Введите тему для рекомендаций:")
            return
        
        elif user_text == "Health":
            await health(update, context)
            return
        
        # === ОБРАБОТКА ВВОДА ПОСЛЕ КНОПОК ===
        if state.get("waiting_for_normal_ask"):
            state["waiting_for_normal_ask"] = False
            user_text = f"/ask {user_text}"
        
        elif state.get("waiting_for_normal_generate"):
            state["waiting_for_normal_generate"] = False
            user_text = f"/generate_questions {user_text}"
        
        elif state.get("waiting_for_normal_recommend"):
            state["waiting_for_normal_recommend"] = False
            user_text = f"/recommend {user_text}"
        
        
        
        
        
        bot_logger.info(f"Получено сообщение: {user_text}")
        print(f"[handle_message] Получен текст: {user_text}")
        
        command, payload = parse_user_message(user_text)
        bot_logger.info(f"Команда после парсинга: {command}, payload: '{payload}'")
        print(f"[handle_message] Команда: {command}, payload: '{payload}'")
    
        # Ошибка команды
        if command == "error":
            bot_logger.warning(f"Ошибка команды: {payload}")
            await update.message.reply_text(payload)
            return
    
        # Маршрутизация
        endpoint = {
            "/ask": "ask",
            "/generate_questions": "generate_questions",
            "/recommend": "recommend"
        }.get(command)
    
        bot_logger.info(f"Выбран endpoint: {endpoint}")
        
        try:
            bot_logger.info(f"Отправка запроса на сервер: {RAG_SERVER_URL}/{endpoint}")
            response = requests.post(
                f"{RAG_SERVER_URL}/{endpoint}",
                json={"question": payload}
            )
        except Exception as e:
            bot_logger.error(f"Ошибка при запросе к серверу: {e}")
            await update.message.reply_text(f"Не удалось связаться с сервером: {e}")
            return
    
        bot_logger.info(f"Статус-код ответа сервера: {response.status_code}")
        
        if response.status_code != 200:
            bot_logger.error(f"Ошибка сервера: {response.text}")
            await update.message.reply_text("Ошибка на сервере.")
            return
    
        data = response.json()
        bot_logger.info(f"Ответ сервера (json): {data}")
        print(f"[handle_message] Ответ сервера: {data}")
    
        # ---------------------------------------
        #   ОБРАБОТКА /ask
        # ---------------------------------------
        if command == "/ask":
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            bot_logger.info(f"Получен answer длины {len(answer)}, sources: {sources}")
        
            if not answer:
                bot_logger.warning("Пустой ответ при /ask")
                await update.message.reply_text("Не удалось получить ответ.")
                return
        
            text = f"*Ответ:*\n{answer}"
        
            if isinstance(sources, list) and sources:
                text += "\n\n*Источники:*"
                for src in sources:
                    if isinstance(src, dict):
                        src_line = ", ".join(f"{k}: {v}" for k, v in src.items())
                        text += f"\n- {src_line}"
                    else:
                        text += f"\n- {src}"
        
            bot_logger.info("Отправка ответа пользователю по /ask")
            
            await send_markdown_chunks_safe(update, text)
            return
    
        # ---------------------------------------
        #   ОБРАБОТКА /generate_questions
        # ---------------------------------------
        if command == "/generate_questions":
            questions = data.get("questions", [])
            bot_logger.info(f"Сгенерированные вопросы: {questions}")
        
            if not questions:
                bot_logger.warning("Список вопросов пуст")
                await update.message.reply_text("Не удалось сгенерировать вопросы.")
                return
        
            # Формируем текст: каждый вопрос с новой строки
            formatted_questions = "\n".join(f"- {q}" for q in questions)
            text = f"*Сгенерированные вопросы:*\n\n{formatted_questions}"
        
            bot_logger.info("Отправка сгенерированных вопросов пользователю")
            
            # Используем ту же функцию, что и для /ask — она уже поддерживает Markdown и длинные сообщения
            await send_markdown_chunks_safe(update, text)
            return
    
        # ---------------------------------------
        #   ОБРАБОТКА /recommend
        # ---------------------------------------
        if command == "/recommend":
            materials = data.get("materials", [])
            bot_logger.info(f"Рекомендованные материалы: {materials}")
        
            if not materials:
                bot_logger.warning("Список материалов пуст")
                await update.message.reply_text("Не удалось получить рекомендации.")
                return
        
            # Формируем текст
            text = "*Рекомендованные материалы:*\n"
            for m in materials:
                if isinstance(m, dict):
                    # Форматируем как "ключ: значение", экранирование не нужно — будет обработано в send_markdown_chunks_safe
                    line = ", ".join(f"{k}: {v}" for k, v in m.items())
                    text += f"\n- {line}"
                else:
                    text += f"\n- {m}"
        
            bot_logger.info("Отправка рекомендованных материалов пользователю")
        
            # Используем ту же надёжную функцию отправки
            await send_markdown_chunks_safe(update, text)
            return
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        return
    
    # Fallback
    await update.message.reply_text("Используйте /start для начала")
  
  
  
  
  
  
  
  
  
  
    


# ## Работа с файлами

# In[ ]:


# Загрузка файла
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Функция принимает txt-файлы и передает Flask серверу для дообучения модели"""
    user_id = update.effective_user.id
    bot_logger.info("Получено сообщение с файлами")

    print("\n====================")
    print(f"[handle_file] Получено сообщение с файлами от пользователя {user_id}")

    # Собираем документы
    documents = []

    if update.message.document:
        documents.append(update.message.document)
        bot_logger.info(f"Найден одиночный документ: {update.message.document.file_name}")

    # Если файлов нет
    if not documents:
        bot_logger.warning("Файлы отсутствуют в сообщении")
        print("[handle_file] Файлы отсутствуют в сообщении.")
        await update.message.reply_text("Не удалось определить файлы.") 
        return

    # Папка для загрузок
    save_dir = "uploads"
    os.makedirs(save_dir, exist_ok=True)

    # Обрабатываем каждый файл
    for doc in documents:
        file_name = doc.file_name
        file_name_lower = file_name.lower()

        bot_logger.info(f"Обработка файла: {file_name}")
        print(f"\n[handle_file] Найден файл: {file_name}")

        # Проверка расширения
        if not file_name_lower.endswith(".txt"):
            bot_logger.warning(f"[handle_file] Файл отклонён: {file_name}")
            print(f"[handle_file] Файл отклонён: {file_name}")
            await update.message.reply_text(
                f"Файл '{file_name}' отклонён. Разрешён только формат .txt"
            )
            continue

        # Скачивание файла
        file_obj = await context.bot.get_file(doc.file_id)
        file_path = os.path.join(save_dir, file_name)

        bot_logger.info(f"Скачивание файла: {file_path}")
        print(f"[handle_file] Скачивание файла: {file_path}")
        await file_obj.download_to_drive(file_path)
        bot_logger.info(f"Файл сохранён: {file_path}")
        print(f"[handle_file] Файл сохранён")

        await update.message.reply_text(f"Файл '{file_name}' успешно загружен.")

        # Отправка файла на сервер
        try:
            with open(file_path, "rb") as f:
                response = requests.post(
                    f"{RAG_SERVER_URL}/upload",
                    files={"file": (file_name, f, "text/plain")}
                )

            if response.status_code == 200:
                bot_logger.info(f"Файл '{file_name}' успешно обработан сервером")
                print(f"[handle_file] Сервер обработал файл")
                await update.message.reply_text(
                    f"Файл '{file_name}' успешно обработан сервером."
                )
                
            else:
                bot_logger.error(f"Ошибка сервера при обработке '{file_name}': код {response.status_code}")
                print(f"[handle_file] Сервер вернул ошибку: {response.status_code}")
                await update.message.reply_text(
                    f"Ошибка обработки файла '{file_name}' на сервере."
                )
                

        except Exception as e:
            bot_logger.error(f"Ошибка при отправке файла '{file_name}' на сервер: {e}")
            print(f"[handle_file] Ошибка отправки файла '{file_name}': {e}")
            await update.message.reply_text(
                f"Ошибка при отправке файла '{file_name}' на сервер."
            )


# ## Вспомогательные функции

# In[ ]:


# /health
async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("[health] Проверка статуса сервера...")
    bot_logger.info("Проверка статуса сервера...")

    try:
        response = requests.get(f"{RAG_SERVER_URL}/health")
    except Exception as e:
        bot_logger.error(f"[health] Ошибка запроса: {e}")
        print(f"[health] Ошибка запроса: {e}")
        await update.message.reply_text(f"Не удалось связаться с сервером: {e}")
        return

    print(f"[health] Сервер вернул статус: {response.status_code}")

    if response.status_code == 200:
        await update.message.reply_text(f"Сервер работает: {response.text}")
    else:
        await update.message.reply_text("Сервер ответил ошибкой.")


# /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"[INFO] Получена команда /start от пользователя {update.effective_user.id}")
    bot_logger.info("Получена команда /start")
    
    user_id = update.effective_user.id
    
    
    # СБРОС СОСТОЯНИЯ
    USER_STATE[user_id] = {"mode": None, "email": None, "active_skill": None}
    
    
    
    keyboard = get_mode_selection_keyboard()
    
    await update.message.reply_text(
        "👋 Добро пожаловать! Я AI-репетитор. Напишите /help или\n\n"
        "Выберите режим работы:",
        reply_markup=keyboard
    )


# /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"[INFO] Получена команда /help от пользователя {update.effective_user.id}")
    bot_logger.info("Получена команда /help")
    
    api_description = {
        "/health": "Проверка работоспособности сервера.",
        "/ask": "Задайте вопрос. Возвращает ответ и источники.",
        "/generate_questions": "Генерация вопросов по теме. Возвращает список вопросов.",
        "/recommend": "Рекомендации по теме. Возвращает список материалов."
    }

    help_text = "Доступные команды:\n\n"
    for route, desc in api_description.items():
        help_text += f"{route}: {desc}\n"

    help_text += """\nПримеры использования:
    Расскажи про облачные хранилища
    или
    /ask Что такое виртуализация?
    /generate_questions Машинное обучение
    /recommend Основы SQL\n"""
    
    help_text += "\nЗамечания:\n"
    help_text += "Если команду не указывать — используется /ask автоматически.\n"
    help_text += "\nТакже предусмотрена возможность загружать файлы в формате .txt для дообучения модели. "
    help_text += "При загрузке файлов не учитывается текст сообщения.\n"
    
    await update.message.reply_text(help_text)


# ## Применение

# In[ ]:


def main():
    print("=== Запуск Telegram бота... ===")
    bot_logger.info("Запуск Telegram бота")

    if not TELEGRAM_TOKEN:
        print("ОШИБКА: TELEGRAM_TOKEN не указан!")
        bot_logger.error("TELEGRAM_TOKEN не указан")
        return

    bot_logger.info("Создание приложения")
    print("[INFO] Создание приложения...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    bot_logger.info("Регистрация обработчиков")
    print("[INFO] Регистрация обработчиков...")
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))

    bot_logger.info("Бот запущен. Ожидание сообщений")
    print("=== Бот запущен. Ожидание сообщений... ===")

    try:
        app.run_polling()
    except Exception as e:
        bot_logger.error(f"Ошибка при запуске: {e}")
        print(f"Ошибка при запуске: {e}")
    finally:
        print("=== Бот остановлен ===")
        bot_logger.info("Бот остановлен")








# === НОВЫЕ ФУНКЦИИ ДЛЯ SKILLUP ===

async def ask_for_email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать кнопки с email для выбора"""
    user_id = update.effective_user.id
    USER_STATE[user_id]["waiting_for_email"] = True
    
    keyboard = get_email_selection_keyboard()
    
    await update.message.reply_text(
        "🎓 Вход в SkillUp\n\n"
        "Выберите ваш email:",
        reply_markup=keyboard
    )


async def handle_email_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора email из кнопок"""
    user_id = update.effective_user.id
    email = update.message.text.strip().lower()
    
    # Проверка на кнопку "Сменить режим"
    if email == "🔄 сменить режим":
        USER_STATE[user_id]["waiting_for_email"] = False
        keyboard = get_mode_selection_keyboard()
        await update.message.reply_text("Выберите режим:", reply_markup=keyboard)
        return
    
    # Валидация email
    valid, error = validate_email(email)
    if not valid:
        await update.message.reply_text(f"❌ {error}")
        return
    
    # Загрузка или создание пользователя
    user_data = load_user_data(email)
    if not user_data:
        user_data = create_new_user(email)
        save_user_data(email, user_data)
    
    USER_STATE[user_id]["email"] = email
    USER_STATE[user_id]["waiting_for_email"] = False
    
    await update.message.reply_text(
        f"✅ Вход выполнен!\n💼 {user_data['position_title']}"
    )
    
    await show_skill_selection(update, context)
    


async def handle_skill_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора навыка"""
    user_id = update.effective_user.id
    email = USER_STATE[user_id].get("email")
    selected_skill = update.message.text.strip()
    
    # Проверка на служебные кнопки
    if selected_skill == "🔄 Сменить режим":
        USER_STATE[user_id]["waiting_for_skill"] = False
        keyboard = get_mode_selection_keyboard()
        await update.message.reply_text("Выберите режим:", reply_markup=keyboard)
        return
    
    # Проверка, что навык существует
    all_skills = get_all_user_skills(email)
    if selected_skill not in all_skills:
        await update.message.reply_text("❌ Неизвестный навык. Выберите из списка.")
        return
    
    # Активируем навык
    USER_STATE[user_id]["active_skill"] = selected_skill
    USER_STATE[user_id]["waiting_for_skill"] = False
    
    # Показываем главное меню с активным навыком
    await show_skillup_menu_with_active_skill(update, context)
    
    
    
async def show_skillup_menu_with_active_skill(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Главное меню SkillUp с активным навыком"""
    user_id = update.effective_user.id
    email = USER_STATE[user_id].get("email")
    active_skill = USER_STATE[user_id].get("active_skill")
    
    if not email or not active_skill:
        await show_skill_selection(update, context)
        return
    
    user_data = load_user_data(email)
    progress = get_skill_progress(email, active_skill)
    
    # Текст меню
    text = f"🎓 {user_data['position_title']}\n\n"
    text += f"📚 Активный навык: {active_skill}\n"
    text += f"📊 Прогресс: {format_progress_bar(progress)} {progress}%\n\n"
    
    text += "💡 Выберите действие:"
    
    keyboard = get_skillup_active_keyboard(active_skill, progress)
    
    await update.message.reply_text(text, reply_markup=keyboard)
    
    
    
async def handle_skillup_with_active_skill(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка действий с активным навыком"""
    user_id = update.effective_user.id
    email = USER_STATE[user_id].get("email")
    active_skill = USER_STATE[user_id].get("active_skill")
    text = update.message.text
    
    # === КЛИК НА АКТИВНЫЙ НАВЫК (показываем детали) ===
    if text.startswith("📚 "):
        await show_active_skill_details(update, context)
        return
    
    # === ВОПРОС ===
    elif text == "❓ Вопрос":
        USER_STATE[user_id]["waiting_for_question"] = True
        await update.message.reply_text(
            f"❓ Задайте вопрос по навыку {active_skill}:\n\n"
            "Можете писать просто вопрос."
        )
        return
    
    # === ТЕСТ ===
    elif text == "✅ Тест":
        await generate_test_for_active_skill(update, context)
        return
    
    # === СОВЕТЫ ===
    elif text == "🎯 Советы":
        await show_recommendations(update, context, email)
        return
    
    # === ОБРАБОТКА ВОПРОСА ===
    elif USER_STATE[user_id].get("waiting_for_question"):
        await handle_question_for_active_skill(update, context)
        return
    
    # Fallback
    await show_skillup_menu_with_active_skill(update, context)        


async def generate_test_for_active_skill(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Генерация теста по активному навыку"""
    user_id = update.effective_user.id
    email = USER_STATE[user_id].get("email")
    active_skill = USER_STATE[user_id].get("active_skill")
    
    await update.message.reply_text("⏳ Генерирую вопросы...")
    
    try:
        response = requests.post(
            f"{RAG_SERVER_URL}/generate_questions",
            json={"topic": active_skill}
        )
        
        if response.status_code == 200:
            data = response.json()
            questions = data.get("questions", [])
            
            if not questions:
                await update.message.reply_text("😔 Не удалось сгенерировать вопросы")
                return
            
            text = f"✅ Тест: {active_skill}\n\n"
            text += "📝 Вопросы для самопроверки:\n\n"
            
            for i, q in enumerate(questions[:5], 1):
                text += f"{i}. {q}\n"
            
            # Обновляем прогресс
            update_progress(email, active_skill, increment=5)
            new_progress = get_skill_progress(email, active_skill)
            
            text += f"\n✅ Прогресс обновлён: {active_skill} → {new_progress}%"
            
            await send_markdown_chunks_safe(update, text)
        else:
            await update.message.reply_text("😔 Не удалось сгенерировать тест")
    
    except Exception as e:
        bot_logger.error(f"Error in generate_test: {e}")
        await update.message.reply_text(f"⚠️ Ошибка: {str(e)}")
    
    await show_skillup_menu_with_active_skill(update, context)




    
async def show_skill_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать список навыков для выбора активного"""
    user_id = update.effective_user.id
    email = USER_STATE[user_id].get("email")
    
    if not email:
        await ask_for_email(update, context)
        return
    
    user_data = load_user_data(email)
    all_skills = get_all_user_skills(email)
    
    # Формируем текст с прогрессом
    text = "📚 Выберите навык для работы:\n\n"
    
    text += "🔴 БАЗОВЫЕ:\n"
    for skill in user_data["base_skills"]:
        progress = get_skill_progress(email, skill)
        bar = format_progress_bar(progress, length=5)
        status = "✅" if progress == 100 else "🔄" if progress > 0 else "⚪"
        text += f"{status} {skill} {bar} {progress}%\n"
    
    if user_data["personal_skills"]:
        text += "\n🟢 ЛИЧНЫЕ:\n"
        for skill in user_data["personal_skills"]:
            progress = get_skill_progress(email, skill)
            bar = format_progress_bar(progress, length=5)
            status = "✅" if progress == 100 else "🔄" if progress > 0 else "⚪"
            text += f"{status} {skill} {bar} {progress}%\n"
    
    text += "\n👇 Нажмите на навык для активации"
    
    keyboard = get_skill_selection_keyboard(all_skills)
    
    USER_STATE[user_id]["waiting_for_skill"] = True
    
    await update.message.reply_text(text, reply_markup=keyboard)
        
    

async def handle_email_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка ввода email"""
    user_id = update.effective_user.id
    email = update.message.text.strip().lower()
    
    valid, error = validate_email(email)
    if not valid:
        await update.message.reply_text(f"❌ {error}")
        return
    
    # Загрузка или создание пользователя
    user_data = load_user_data(email)
    if not user_data:
        user_data = create_new_user(email)
        save_user_data(email, user_data)
    
    USER_STATE[user_id]["email"] = email
    USER_STATE[user_id]["waiting_for_email"] = False
    
    await update.message.reply_text(
        f"✅ Вход выполнен!\n💼 {user_data['position_title']}"
    )
    
    await show_skillup_menu(update, context)

async def show_skillup_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Главное меню SkillUp"""
    user_id = update.effective_user.id
    email = USER_STATE[user_id].get("email")
    user_data = load_user_data(email)
    
    text = f"👋 Привет!\n💼 {user_data['position_title']}\n\n"
    text += "📊 ПРОГРЕСС:\n"
    
    for skill in user_data["base_skills"]:
        progress = get_skill_progress(email, skill)
        bar = format_progress_bar(progress)
        status = "✅" if progress == 100 else "🔄" if progress > 0 else ""
        text += f"├ {skill}: {bar} {progress}% {status}\n"
    
    if user_data["personal_skills"]:
        text += f"\n💡 Личных: {len(user_data['personal_skills'])}\n"
    
    keyboard = get_skillup_main_keyboard()
    await update.message.reply_text(text, reply_markup=keyboard)

async def handle_skillup_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка сообщений в SkillUp режиме"""
    user_id = update.effective_user.id
    email = USER_STATE[user_id].get("email")
    text = update.message.text
    
    if text == "📚 Навыки":
        await show_skills_list(update, context, email)
    
    elif text == "❓ Вопрос":
        USER_STATE[user_id]["waiting_for_question"] = True
        await update.message.reply_text("Задайте вопрос по навыку:")
    
    elif text == "✅ Тест":
        await update.message.reply_text("Укажите навык для теста:")
        USER_STATE[user_id]["waiting_for_test_skill"] = True
    
    elif text == "🎯 Советы":
        await show_recommendations(update, context, email)
    
    elif USER_STATE[user_id].get("waiting_for_question"):
        await handle_skillup_question(update, context, email, text)
    
    elif USER_STATE[user_id].get("waiting_for_test_skill"):
        await handle_skillup_test(update, context, email, text)

async def show_skills_list(update: Update, context: ContextTypes.DEFAULT_TYPE, email: str):
    """Показать список навыков"""
    user_data = load_user_data(email)
    
    text = "📚 Навыки\n\n🔴 БАЗОВЫЕ:\n"
    for skill in user_data["base_skills"]:
        progress = get_skill_progress(email, skill)
        bar = format_progress_bar(progress)
        text += f"\n{skill}\n{bar} {progress}%\n"
    
    if user_data["personal_skills"]:
        text += "\n🟢 ЛИЧНЫЕ:\n"
        for skill in user_data["personal_skills"]:
            progress = get_skill_progress(email, skill)
            bar = format_progress_bar(progress)
            text += f"\n{skill}\n{bar} {progress}%\n"
    
    await update.message.reply_text(text)

async def handle_skillup_question(update: Update, context: ContextTypes.DEFAULT_TYPE, email: str, question: str):
    """Обработка вопроса в SkillUp режиме"""
    user_id = update.effective_user.id
    
    # Парсинг навыка из вопроса
    skill_context = None
    if ":" in question:
        parts = question.split(":", 1)
        skill_context = parts[0].strip()
        question = parts[1].strip()
    
    # Используем СУЩЕСТВУЮЩИЙ код для /ask
    query = f"{skill_context}: {question}" if skill_context else question
    
    try:
        response = requests.post(
            f"{RAG_SERVER_URL}/ask",
            json={"question": query}
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            
            text = f"🤖 {answer}\n\n"
            if sources:
                text += "📚 Источники:\n"
                for src in sources[:2]:
                    text += f"• {src.get('title', 'Документ')}\n"
            
            # Обновляем прогресс
            if skill_context:
                update_progress(email, skill_context, increment=5)
                new_progress = get_skill_progress(email, skill_context)
                text += f"\n✅ Прогресс: {skill_context} → {new_progress}%"
            
            await send_markdown_chunks_safe(update, text)
    
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")
    
    USER_STATE[user_id]["waiting_for_question"] = False
    await show_skillup_menu(update, context)

async def handle_skillup_test(update: Update, context: ContextTypes.DEFAULT_TYPE, email: str, skill: str):
    """Генерация теста по навыку"""
    user_id = update.effective_user.id
    
    try:
        response = requests.post(
            f"{RAG_SERVER_URL}/generate_questions",
            json={"question": skill}
        )
        
        if response.status_code == 200:
            data = response.json()
            questions = data.get("questions", [])
            
            text = f"✅ Тест: {skill}\n\n"
            for i, q in enumerate(questions[:10], 1):
                text += f"{i}. {q}\n"
            
            # Обновляем прогресс
            update_progress(email, skill, increment=5)
            
            await send_markdown_chunks_safe(update, text)
    
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {str(e)}")
    
    USER_STATE[user_id]["waiting_for_test_skill"] = False
    await show_skillup_menu(update, context)

async def show_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE, email: str):
    """Показать рекомендации материалов для активного навыка"""
    user_id = update.effective_user.id
    active_skill = USER_STATE[user_id].get("active_skill")
    
    if not active_skill:
        await update.message.reply_text("⚠️ Навык не выбран")
        return
    
    await update.message.reply_text("⏳ Ищу материалы...")
    
    try:
        response = requests.post(
            f"{RAG_SERVER_URL}/recommend",
            json={"topic": active_skill}
        )
        
        if response.status_code == 200:
            data = response.json()
            materials = data.get("materials", [])
            
            if not materials:
                await update.message.reply_text("😔 Материалы не найдены")
                return
            
            text = f"📚 Материалы по {active_skill}:\n\n"
            
            for i, m in enumerate(materials, 1):
                title = m.get("title", "Без названия")
                url = m.get("url", "")
                category = m.get("category", "")
                
                text += f"{i}. {title}\n"
                if category:
                    text += f"   Категория: {category}\n"
                if url:
                    text += f"   {url}\n"
                text += "\n"
            
            await send_markdown_chunks_safe(update, text)
        else:
            await update.message.reply_text("😔 Не удалось получить материалы")
    
    except Exception as e:
        bot_logger.error(f"Error in show_recommendations: {e}")
        await update.message.reply_text(f"⚠️ Ошибка: {str(e)}")



async def handle_question_for_active_skill(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка вопроса по активному навыку"""
    user_id = update.effective_user.id
    email = USER_STATE[user_id].get("email")
    active_skill = USER_STATE[user_id].get("active_skill")
    question = update.message.text
    
    # Формируем запрос с контекстом навыка
    query = f"{active_skill}: {question}"
    
    await update.message.reply_text("⏳ Ищу ответ...")
    
    try:
        response = requests.post(
            f"{RAG_SERVER_URL}/ask",
            json={"question": query}
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            
            text = f"🤖 Ответ:\n\n{answer}\n\n"
            
            if sources:
                text += "📚 Источники:\n"
                for i, src in enumerate(sources, 1):
                    title = src.get('title', 'Документ')
                    url = src.get('url', '')
                    text += f"{i}. {title}\n"
                    if url:
                        text += f"   {url}\n"
            
            # Обновляем прогресс
            update_progress(email, active_skill, increment=5)
            new_progress = get_skill_progress(email, active_skill)
            text += f"\n✅ Прогресс обновлён: {active_skill} → {new_progress}%"
            
            await send_markdown_chunks_safe(update, text)
        else:
            await update.message.reply_text("😔 Не удалось получить ответ")
    
    except Exception as e:
        bot_logger.error(f"Error in handle_question: {e}")
        await update.message.reply_text(f"⚠️ Ошибка: {str(e)}")
    
    USER_STATE[user_id]["waiting_for_question"] = False
    await show_skillup_menu_with_active_skill(update, context)









if __name__ == "__main__":
    main()

