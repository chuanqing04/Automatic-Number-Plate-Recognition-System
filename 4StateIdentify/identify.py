import json

def identify_state(ocr_json: dict) -> dict:
    plate_text = ocr_json.get("plate_text", "").upper()

    if not plate_text:
        result = ocr_json.copy()
        result["state"] = "Unknown"
        return result

    mapping = {
        "A": "Perak",
        "B": "Selangor",
        "C": "Pahang",
        "D": "Kelantan",
        "E": "Sabah",        # old
        "F": "Putrajaya",
        "J": "Johor",
        "K": "Kedah",
        "L": "Labuan",
        "M": "Malacca",
        "N": "Negeri Sembilan",
        "P": "Penang",
        "Q": "Sarawak",
        "R": "Perlis",
        "S": "Sabah",
        "T": "Terengganu",
        "V": "Kuala Lumpur",
        "W": "Kuala Lumpur",
        "Z": "Military",
    }

    first_char = plate_text[0]
    state = mapping.get(first_char, "Unknown")

    # optional special case
    if plate_text.startswith("PUTRAJAYA"):
        state = "Putrajaya"

    result = ocr_json.copy()
    result["state"] = state
    return result


def write_state_json(ocr_json: dict, output_path: str) -> str:
    updated = identify_state(ocr_json)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    return output_path
