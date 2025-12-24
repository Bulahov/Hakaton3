"""Конфигурация должностей - ХАРДКОД для MVP"""

POSITIONS = {
    "devops": {
        "title": "DevOps Engineer",
        "base_skills": ["Kubernetes", "Docker", "CI/CD", "Terraform"],
        "recommended_skills": ["Prometheus", "Grafana", "ArgoCD"]
    },
    "backend": {
        "title": "Backend Developer",
        "base_skills": ["PostgreSQL", "Git", "REST API", "Redis"],
        "recommended_skills": ["GraphQL", "gRPC", "Microservices"]
    },
    "ml": {
        "title": "ML Engineer",
        "base_skills": ["PyTorch", "Huggingface", "Model Training"],
        "recommended_skills": ["MLOps", "ONNX", "TensorRT"]
    },
    "data": {
        "title": "Data Engineer",
        "base_skills": ["Spark", "Delta Lake", "Trino", "Airflow"],
        "recommended_skills": ["Kafka", "dbt", "Iceberg"]
    },
    "network": {
        "title": "Network Engineer",
        "base_skills": ["VPC", "VPN", "Routing", "Site-to-Site"],
        "recommended_skills": ["Load Balancing", "CDN", "DNS"]
    },
    "sysadmin": {
        "title": "System Administrator",
        "base_skills": ["Linux", "Bash", "Virtualization", "Docker"],
        "recommended_skills": ["Monitoring", "Ansible", "Chef"]
    }
}

def get_position_config(specialty: str):
    return POSITIONS.get(specialty)