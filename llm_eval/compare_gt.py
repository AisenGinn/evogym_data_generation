import json

ANSWER_DICT = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D',
    "A": 0, "B": 1, "C": 2, "D": 3
}

def load_json(file_path):
    """Helper function to load JSON data from a file."""
    with open(file_path, "r") as f:
        return json.load(f)

def map_question_to_answer_batches(questions, answers, batch_size=20):
    """
    Maps batch indices from questions to answers, handling skipped batches.

    Args:
        questions (str or list): File path to questions JSON or already loaded data.
        answers (str or list): File path to answers JSON or already loaded data.
        batch_size (int): The number of questions per batch.

    Returns:
        dict: A mapping of question batch_start -> answer batch_start.
        list: List of confirmed batch starts.
        list: List of skipped batch starts in questions.
    """

    # Load JSON data if file paths are given
    if isinstance(questions, str):
        questions = load_json(questions)
    if isinstance(answers, str):
        answers = load_json(answers)

    confirm_batch_start = []  # Stores batch starts that align in both files
    skipped_batches = []       # Tracks batch indices skipped in answers
    question_to_answer_mapping = {}  # Maps question batch_start -> answer batch_start

    question_idx = 0  # Tracks batch_start in questions
    answer_idx = 0    # Tracks batch_start in answers

    while question_idx < len(questions):
        confirm = True
        batch_questions = questions[question_idx: question_idx + batch_size]

        # Ensure we don't go out of bounds in answers
        if answer_idx >= len(answers):  
            #print(f"⚠️ Skipping batch_start {question_idx} - No corresponding batch in answers.")
            skipped_batches.append(question_idx)
            question_idx += batch_size  # Move to next batch in questions
            continue

        batch_answer = answers[answer_idx: answer_idx + batch_size]

        if len(batch_questions) != len(batch_answer):  
            print(f"⚠️ Skipping batch_start {question_idx} due to batch size mismatch.")
            skipped_batches.append(question_idx)
            question_idx += batch_size  # Move to next batch in questions
            continue

        # Validate answers match in both batches
        for i in range(len(batch_questions)):  
            question_answer = ANSWER_DICT[batch_questions[i]['correct_answer']]
            gt = batch_answer[i]["correct_answer"]

            if question_answer != gt:
                confirm = False
                break  # Skip this batch if any mismatch is found

        if confirm:
            confirm_batch_start.append(question_idx)
            question_to_answer_mapping[question_idx] = answer_idx  # Map question batch_start -> answer batch_start
            answer_idx += batch_size  # Increment answer index **only for valid batches**

        # Always increment question index to check the next batch in questions
        question_idx += batch_size

    return question_to_answer_mapping, confirm_batch_start, skipped_batches

# Example Usage
questions_path = "/media/hdd2/users/changhe/saved_questions/Balancer-v0/Balancer-v0_QA_better_easy_2_repeat.json"
answers_path = "/media/hdd2/users/changhe/saved_answers/gpto3mini/Balancer-v0/Balancer-v0_ANS_better_easy_2_repeat_3.json"

mapping, confirmed_batches, skipped_batches = map_question_to_answer_batches(questions_path, answers_path, batch_size=20)

#print("✅ Confirmed batch starts:", confirmed_batches)
#print("❌ Skipped batches (in answers):", skipped_batches)
print("🔗 Question to Answer Mapping:", mapping)
