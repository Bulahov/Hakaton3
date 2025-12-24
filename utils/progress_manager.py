"""Управление прогрессом пользователей"""

import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/user_progress.json")

def load_db():
    """Загрузить всю БД"""
    if not DB_PATH.exists():
        DB_PATH.parent.mkdir(exist_ok=True)
        with open(DB_PATH, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        return {}
    with open(DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_db(data):
    """Сохранить БД"""
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_user_data(email):
    """Загрузить данные пользователя"""
    db = load_db()
    return db.get(email)

def save_user_data(email, user_data):
    """Сохранить данные пользователя"""
    db = load_db()
    db[email] = user_data
    save_db(db)

def get_skill_progress(email, skill):
    """Получить прогресс по навыку (0-100)"""
    user_data = load_user_data(email)
    if not user_data:
        return 0
    return user_data.get("progress", {}).get(skill, {}).get("percent", 0)

def update_progress(email, skill, increment=5):
    """Обновить прогресс по навыку"""
    user_data = load_user_data(email)
    if not user_data:
        return
    
    if skill not in user_data["progress"]:
        user_data["progress"][skill] = {
            "percent": 0,
            "questions_asked": 0,
            "tests_passed": 0,
            "last_activity": datetime.now().isoformat()
        }
    
    skill_progress = user_data["progress"][skill]
    new_percent = min(100, skill_progress["percent"] + increment)
    skill_progress["percent"] = new_percent
    skill_progress["questions_asked"] += 1
    skill_progress["last_activity"] = datetime.now().isoformat()
    
    save_user_data(email, user_data)

def add_personal_skill(email, skill_name):
    """Добавить личный навык"""
    user_data = load_user_data(email)
    if not user_data:
        return False
    
    all_skills = user_data["base_skills"] + user_data["personal_skills"]
    if skill_name in all_skills:
        return False
    
    user_data["personal_skills"].append(skill_name)
    save_user_data(email, user_data)
    return True

def get_all_user_skills(email):
    """Получить все навыки пользователя (базовые + личные)"""
    user_data = load_user_data(email)
    if not user_data:
        return []
    return user_data["base_skills"] + user_data["personal_skills"]

def format_progress_bar(percent, length=10):
    """Форматировать прогресс-бар"""
    filled = int(percent / 100 * length)
    return "█" * filled + "░" * (length - filled)