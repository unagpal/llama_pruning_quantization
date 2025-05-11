import pandas as pd

# Construct multiple choice prompt from RACE-H reading comprehension question
def race_entry_to_prompt (row: pd.Series) -> str:
    if len(row["options"]) != 4:
        raise ValueError("Invalid race question format: 4 answer choices expected")

    return f"""Article: {row["article"]}
Question: {row.question}
Select the best choice from the following options:
A) {row["options"][0]}
B) {row["options"][1]}
C) {row["options"][2]}
D) {row["options"][3]}
"""

# Load RACE-H questions and associated prompts
def load_data () -> pd.DataFrame:
    df = pd.read_parquet("race_high_test")
    df["prompt"] = df.apply(race_entry_to_prompt, axis=1)
    return df
