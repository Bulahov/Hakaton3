"""Клавиатуры для Telegram бота"""

from telegram import ReplyKeyboardMarkup, KeyboardButton

def get_mode_selection_keyboard():
    """Выбор режима при /start"""
    keyboard = [
        [KeyboardButton("💬 Обычный режим")],
        [KeyboardButton("🎓 SkillUp режим")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_email_selection_keyboard():
    """Выбор email (кнопки вместо ввода текста)"""
    keyboard = [
        [KeyboardButton("devops@skillup.com")],
        [KeyboardButton("backend@skillup.com")],
        [KeyboardButton("ml@skillup.com")],
        [KeyboardButton("data@skillup.com")],
        [KeyboardButton("network@skillup.com")],
        [KeyboardButton("sysadmin@skillup.com")],
        [KeyboardButton("🔄 Сменить режим")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_skill_selection_keyboard(skills):
    """
    Клавиатура выбора навыка
    skills: list of skill names
    """
    keyboard = []
    for skill in skills:
        keyboard.append([KeyboardButton(skill)])
    keyboard.append([KeyboardButton("🔄 Сменить режим")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_skillup_active_keyboard(active_skill, progress):
    """
    Главное меню SkillUp с активным навыком
    active_skill: название активного навыка
    progress: прогресс в процентах
    """
    keyboard = [
        [KeyboardButton(f"📚 {active_skill} ({progress}%)")],  # Показываем активный навык
        [KeyboardButton("❓ Вопрос"), KeyboardButton("✅ Тест")],
        [KeyboardButton("🎯 Советы")],
        [KeyboardButton("🔄 Сменить навык"), KeyboardButton("👤 Сменить пользователя")],
        [KeyboardButton("🔄 Сменить режим")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_normal_mode_keyboard():
    """Клавиатура обычного режима"""
    keyboard = [
        [KeyboardButton("Ask"), KeyboardButton("Generate Questions")],
        [KeyboardButton("Recommend"), KeyboardButton("Health")],
        [KeyboardButton("🔄 Сменить режим")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)