def calculate_grounding_ratio(response: str, cannot_answer_phrase: str = "I cannot answer this") -> float:
    """Basic structural metric checking if the response appropriately abstained."""
    if cannot_answer_phrase.lower() in response.lower():
        return 0.0
    return 1.0