"""
ai_assistant.py
---------------
Generates personalized study advice using a small, LOCAL Hugging Face
Transformer model (flan-t5-small by default). No cloud API calls, no API
key, no internet needed once the model has been downloaded once by
`transformers` (it is cached locally after the first run).

WHAT IS flan-t5:
A small "instruction-tuned" text-to-text model from Google. Instruction-
tuned means it was trained specifically to follow plain-English
instructions (like "summarize this" or "give me 3 tips"), which makes it
good for short, structured advice generation without any fine-tuning of
our own.

WHY flan-t5-small / flan-t5-base HERE:
- Runs on a normal laptop CPU in a few seconds (no GPU required).
- Small download (~80MB for -small, ~250MB for -base).
- Good enough for short, templated advice - we are not asking it to do
  open-ended reasoning, just to fill in a well-structured prompt.
"""

from functools import lru_cache

MODEL_NAME = "google/flan-t5-small"  # swap to "google/flan-t5-base" for slightly better quality


@lru_cache(maxsize=1)
def _get_pipeline():
    """
    Load the Hugging Face text2text-generation pipeline once and cache it.

    WHY CACHE: loading a Transformer model from disk into memory takes a
    few seconds. Without caching, every Streamlit interaction would reload
    it from scratch, making the app feel slow.
    """
    from transformers import pipeline
    return pipeline("text2text-generation", model=MODEL_NAME)


def _build_prompt(student_info: dict) -> str:
    """Builds the instruction prompt sent to the local model."""
    return (
        "You are a friendly, encouraging academic advisor for a student.\n\n"
        f"Student Information\n"
        f"Attendance: {student_info['attendance_percentage']}%\n"
        f"Study Hours per week: {student_info['weekly_study_hours']}\n"
        f"Previous Grade: {student_info['previous_grade']}\n"
        f"Failures: {student_info['failures']}\n"
        f"Predicted Grade: {student_info['predicted_grade']}\n"
        f"Confidence: {student_info['confidence']}%\n\n"
        "Write, in simple and encouraging language:\n"
        "1. A short explanation of why this prediction was likely made.\n"
        "2. Five practical study improvement tips.\n"
        "3. A simple one-week study schedule.\n"
        "4. Two healthy study habits.\n"
    )


def generate_study_advice(student_info: dict) -> str:
    """
    Generate personalized study advice for one student.

    Parameters
    ----------
    student_info : dict
        Must contain: attendance_percentage, weekly_study_hours,
        previous_grade, failures, predicted_grade, confidence.

    Returns
    -------
    str - the model's generated advice text.
    """
    try:
        generator = _get_pipeline()
        prompt = _build_prompt(student_info)
        # max_new_tokens keeps the response short and fast on CPU.
        # do_sample=False -> deterministic, on-topic output (no randomness),
        # which is more reliable for small models like flan-t5-small.
        output = generator(prompt, max_new_tokens=220, do_sample=False)
        return output[0]["generated_text"].strip()
    except Exception as exc:
        # Graceful fallback if the model hasn't been downloaded yet / no
        # internet on first run / any other runtime issue - the rest of
        # the app should still work even if this piece fails.
        return (
            "AI advice is unavailable right now "
            f"({exc.__class__.__name__}). "
            "Make sure `transformers` and `torch` are installed and that "
            "you have internet access the first time the model is loaded "
            "(it is cached locally afterward). "
            "In the meantime: focus on improving attendance, aim for at "
            "least 5-6 focused study hours a week, and review topics from "
            "past failures first."
        )


if __name__ == "__main__":
    sample = {
        "attendance_percentage": 72,
        "weekly_study_hours": 2,
        "previous_grade": 78,
        "failures": 1,
        "predicted_grade": "B",
        "confidence": 88,
    }
    print(generate_study_advice(sample))
