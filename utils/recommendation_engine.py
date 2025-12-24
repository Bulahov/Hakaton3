"""Генерация рекомендаций навыков"""

from config.positions import get_position_config
from utils.progress_manager import load_user_data


def get_recommendations(email, max_count=6):
    """Получить рекомендации навыков"""
    user_data = load_user_data(email)
    if not user_data:
        return []

    specialty = user_data["specialty"]
    position_config = get_position_config(specialty)

    if not position_config:
        return []

    current_skills = set(user_data["base_skills"] + user_data["personal_skills"])
    recommendations = []

    # Рекомендованные для должности
    for skill in position_config.get("recommended_skills", []):
        if skill not in current_skills and len(recommendations) < max_count:
            recommendations.append({
                "name": skill,
                "reason": f"Рекомендовано для {position_config['title']}"
            })

    return recommendations[:max_count]