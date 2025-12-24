"""Авторизация пользователей"""

import re
from datetime import datetime
from config.positions import get_position_config


def validate_email(email):
    """Валидация email - возвращает (valid, error_message)"""
    email = email.strip().lower()

    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return False, "Некорректный формат email"

    specialty = email.split('@')[0]
    if not get_position_config(specialty):
        return False, f"Неизвестная специальность: {specialty}"

    return True, ""


def parse_specialty_from_email(email):
    """Извлечь специальность из email"""
    return email.split('@')[0].lower()


def create_new_user(email):
    """Создать нового пользователя"""
    specialty = parse_specialty_from_email(email)
    position_config = get_position_config(specialty)

    if not position_config:
        raise ValueError(f"Unknown specialty: {specialty}")

    now = datetime.now().isoformat()

    return {
        "email": email,
        "specialty": specialty,
        "position_title": position_config["title"],
        "base_skills": position_config["base_skills"].copy(),
        "personal_skills": [],
        "progress": {},
        "total_questions": 0,
        "total_tests": 0,
        "created_at": now,
        "last_login": now
    }