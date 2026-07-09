import json
from dataclasses import asdict, dataclass
from typing import List, Optional, TypedDict


@dataclass
class TaskGoal:
    task: str
    target_object: str
    target_bin: str
    vla_instruction: str
    success_condition: str
    raw_user_command: str


class _Entry(TypedDict):
    keywords: List[str]
    id: str
    name_en: str


# Ordered by keyword specificity (more specific phrases first) so that a
# generic keyword (e.g. "캔") doesn't shadow a more specific one.
OBJECT_TABLE: List[_Entry] = [
    {"keywords": ["플라스틱 컵"], "id": "plastic_cup", "name_en": "the plastic cup"},
    {"keywords": ["종이박스", "종이 박스"], "id": "paper_box", "name_en": "the paper box"},
    {"keywords": ["캔"], "id": "can", "name_en": "the can"},
]

BIN_TABLE: List[_Entry] = [
    {"keywords": ["플라스틱 수거함"], "id": "plastic_bin", "name_en": "the plastic recycling bin"},
    {"keywords": ["종이 수거함"], "id": "paper_bin", "name_en": "the paper recycling bin"},
    {"keywords": ["캔 수거함"], "id": "can_bin", "name_en": "the can recycling bin"},
]


class RuleBasedTaskParser:
    """Rule-based Korean command -> TaskGoal parser.

    No LLM involved. Kept intentionally simple (keyword matching only) so it
    can later be swapped for a Claude Agent-based parser behind the same
    `parse(user_command: str) -> TaskGoal` interface.
    """

    def parse(self, user_command: str) -> TaskGoal:
        object_entry = self._match(user_command, OBJECT_TABLE)
        bin_entry = self._match(user_command, BIN_TABLE)

        if object_entry is None or bin_entry is None:
            raise ValueError(f"Could not parse command: {user_command!r}")

        object_en = object_entry["name_en"]
        bin_en = bin_entry["name_en"]

        vla_instruction = f"Pick {object_en} and place it in {bin_en}."
        success_condition = f"{object_en[0].upper()}{object_en[1:]} is inside {bin_en}."

        return TaskGoal(
            task="pick_and_place",
            target_object=object_entry["id"],
            target_bin=bin_entry["id"],
            vla_instruction=vla_instruction,
            success_condition=success_condition,
            raw_user_command=user_command,
        )

    @staticmethod
    def _match(command: str, table: List[_Entry]) -> Optional[_Entry]:
        for entry in table:
            for keyword in entry["keywords"]:
                if keyword in command:
                    return entry
        return None


if __name__ == "__main__":
    parser = RuleBasedTaskParser()

    sample_commands = [
        "플라스틱 컵을 플라스틱 수거함에 넣어줘",
        "캔을 캔 수거함에 넣어줘",
        "종이박스를 종이 수거함에 넣어줘",
    ]

    for command in sample_commands:
        goal = parser.parse(command)
        print(json.dumps(asdict(goal), ensure_ascii=False, indent=2))
        print()
