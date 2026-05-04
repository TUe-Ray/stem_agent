from typing import Any, Dict

def run(input_data: Dict[str, Any]) -> Dict[str, Any]:
    output = str(input_data.get("output", "")).lower()
    requirements = input_data.get("requirements", [])
    missing = []
    for req in requirements:
        req_text = str(req).lower()
        important_words = [
            word.strip(".,:;!?()[]{}")
            for word in req_text.split()
            if len(word.strip(".,:;!?()[]{}")) >= 5
        ]
        if important_words and not any(word in output for word in important_words):
            missing.append(req)
    total = max(len(requirements), 1)
    score = 1.0 - (len(missing) / total)
    return {"passed": len(missing) == 0, "missing": missing, "score": score}
