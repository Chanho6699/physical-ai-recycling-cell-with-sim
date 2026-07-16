import json
from dataclasses import asdict, dataclass
from typing import List, Optional, TypedDict

from llm_agent.task_goal import TaskGoal as NLTaskGoal


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


# Separate object/bin tables for the Real2Sim natural-language demo. Kept
# apart from OBJECT_TABLE/BIN_TABLE above because the target ids here
# (plastic_bottle/plastic_cup) describe Real2Sim sim_object_type values,
# not the VLA-instruction-oriented ids used by RuleBasedTaskParser.
NL_OBJECT_TABLE: List[dict] = [
    # English keywords added so instructions can be compared cross-language
    # (e.g. benchmark/run_smolvla_language_comparison_eval.py) -- this
    # parser gates target_object/target_bin selection for the whole demo
    # (see RuleBasedTaskGoalParser.parse() below), so an English
    # instruction that only this table doesn't recognize would fail before
    # ever reaching the VLA server, making it look like a VLA-language
    # difference when it was actually just this table being Korean-only.
    {"keywords": ["플라스틱 병", "페트병", "병", "plastic bottle", "bottle"], "id": "plastic_bottle"},
    {"keywords": ["플라스틱 컵", "컵", "plastic cup", "cup"], "id": "plastic_cup"},
]

NL_BIN_TABLE: List[dict] = [
    {"keywords": ["플라스틱 수거함", "플라스틱 통", "plastic bin", "plastic recycling bin", "recycling bin", "bin"], "id": "plastic_bin"},
]


class RuleBasedTaskGoalParser:
    """Rule-based Korean command -> (new-style) TaskGoal parser.

    Separate from RuleBasedTaskParser above -- this one targets the
    Real2Sim natural-language demo's TaskGoal shape (llm_agent.task_goal).
    No LLM involved yet; kept simple so it can later be swapped for an
    LLM-based parser behind the same `parse(instruction) -> TaskGoal | None`
    interface. Returns None (rather than raising) for unsupported commands
    so callers can print a friendly message instead of a traceback.
    """

    def parse(self, instruction: str) -> Optional[NLTaskGoal]:
        object_id = self._match_id(instruction, NL_OBJECT_TABLE)
        bin_id = self._match_id(instruction, NL_BIN_TABLE)

        if object_id is None or bin_id is None:
            return None

        return NLTaskGoal(
            action="pick_and_place",
            target_object=object_id,
            target_bin=bin_id,
            instruction=instruction,
        )

    @staticmethod
    def _match_id(command: str, table: List[dict]) -> Optional[str]:
        # .lower() is a no-op on Korean text (no case) and makes the new
        # English keywords above tolerant of "Bottle"/"BIN"/etc. -- avoids
        # a purely-cosmetic capitalization difference silently failing
        # the whole demo the way an unrecognized instruction does.
        command_lower = command.lower()
        for entry in table:
            for keyword in entry["keywords"]:
                if keyword.lower() in command_lower:
                    return entry["id"]
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

    nl_parser = RuleBasedTaskGoalParser()
    nl_sample_commands = [
        "플라스틱 병을 플라스틱 수거함에 넣어줘",
        "페트병을 플라스틱 수거함에 넣어줘",
        "병을 플라스틱 수거함에 넣어줘",
        "컵을 플라스틱 수거함에 넣어줘",
        "플라스틱 컵을 플라스틱 수거함에 넣어줘",
        "유리병을 유리 수거함에 넣어줘",  # unsupported -> None
    ]

    for command in nl_sample_commands:
        nl_goal = nl_parser.parse(command)
        print(command, "->", nl_goal)
