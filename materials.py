# Default properties (could be extended later)
MATERIAL_PROPERTIES = {
    "Steel": {"k": 43, "rho": 7850, "c": 470},
    "Aluminum": {"k": 205, "rho": 2700, "c": 900},
    "Iron": {"k": 80, "rho": 7874, "c": 450},
}

# 1. Material to alpha mapping
material_alpha = {
    "aluminum": 9.7e-5,
    "steel": 1.2e-5,
    "iron": 2.3e-5,
}