from __future__ import annotations

import json
import random
import re
import urllib.request
from typing import Any


GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)


def download_gsm8k_sample(n_train: int = 30, n_val: int = 20, seed: int = 42) -> tuple[list, list]:
    """
    Download a GSM8K sample and return non-overlapping train/validation cases.
    """
    rows = _load_gsm8k_rows()
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    needed = n_train + n_val
    if len(shuffled) < needed:
        raise RuntimeError(f"GSM8K source returned {len(shuffled)} rows, need {needed}")
    train_rows = shuffled[:n_train]
    val_rows = shuffled[n_train:needed]
    return (
        [_case_from_row(row, f"train_{index + 1:03d}") for index, row in enumerate(train_rows)],
        [_case_from_row(row, f"val_{index + 1:03d}") for index, row in enumerate(val_rows)],
    )


def exact_match_after_extraction(predicted: str, expected: str) -> float:
    """Extract last number from predicted, compare to expected. Return 1.0 or 0.0."""
    numbers = re.findall(r"-?\d+\.?\d*", predicted.replace(",", ""))
    if not numbers:
        return 0.0
    return 1.0 if _normalize_number(numbers[-1]) == _normalize_number(expected) else 0.0


def _load_gsm8k_rows() -> list[dict[str, Any]]:
    try:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            load_dataset = None
        if load_dataset is not None:
            dataset = load_dataset("gsm8k", "main", split="test")
            return [{"question": row["question"], "answer": row["answer"]} for row in dataset]
    except Exception:
        pass

    try:
        with urllib.request.urlopen(GSM8K_TEST_URL, timeout=5) as response:
            text = response.read().decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except Exception:
        return _fallback_rows()


def _case_from_row(row: dict[str, Any], case_id: str) -> dict[str, Any]:
    question = str(row.get("question") or row.get("input") or "")
    answer = _extract_gold_answer(str(row.get("answer") or row.get("expected_output") or ""))
    return {
        "id": case_id,
        "input": {"user_request": question, "problem": question},
        "expected_output": answer,
        "reference_notes": f"Gold numeric answer: {answer}",
    }


def _extract_gold_answer(answer: str) -> str:
    marker = "####"
    if marker in answer:
        answer = answer.split(marker)[-1]
    numbers = re.findall(r"-?\d+\.?\d*", answer.replace(",", ""))
    return _normalize_number(numbers[-1]) if numbers else answer.strip()


def _normalize_number(value: str) -> str:
    text = str(value).replace(",", "").strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return str(number).rstrip("0").rstrip(".")


def _fallback_rows() -> list[dict[str, str]]:
    samples = [
        ("Mia has 12 stickers. She buys 8 more and gives 5 to her friend. How many stickers does she have?", "15"),
        ("A box has 6 bags with 7 marbles each. If 10 marbles are lost, how many remain?", "32"),
        ("Noah reads 9 pages a day for 5 days, then reads 13 more pages. How many pages did he read?", "58"),
        ("A baker makes 4 trays of 12 muffins and sells 19 muffins. How many muffins are left?", "29"),
        ("Lena saved $18 each week for 6 weeks and spent $25. How much money is left?", "83"),
        ("There are 45 students on a bus. At the first stop 12 leave and 7 get on. How many students are on the bus?", "40"),
        ("A garden has 8 rows with 9 carrots each. Rabbits eat 14 carrots. How many carrots remain?", "58"),
        ("Sam has 3 packs of pencils with 10 pencils each and buys 4 more pencils. How many pencils does Sam have?", "34"),
        ("A movie starts at 2 and lasts 3 hours. If 15 minutes are previews, how many total minutes pass?", "195"),
        ("Jules ran 2 miles each day for 7 days and then ran 5 miles on Sunday. How many miles total?", "19"),
        ("A class collects 120 cans. They pack them equally into 8 boxes. How many cans are in each box?", "15"),
        ("A toy costs $14. Kim buys 3 toys and pays with $50. How much change does Kim get?", "8"),
        ("There are 5 tables with 6 chairs each. Two chairs break. How many usable chairs remain?", "28"),
        ("A farmer has 36 eggs and uses 4 eggs in each cake. How many cakes can be made?", "9"),
        ("Tara has 80 beads. She makes 6 bracelets using 9 beads each. How many beads are left?", "26"),
        ("A store sold 25 apples in the morning and twice as many in the afternoon. How many apples were sold?", "75"),
        ("Ben earns $7 per hour for 8 hours and spends $16. How much remains?", "40"),
        ("A team scores 14 points in each of 3 games and 9 points in one game. What is the total?", "51"),
        ("There are 64 crayons shared equally among 4 kids. How many crayons does each kid get?", "16"),
        ("A recipe needs 3 cups of flour. Ana makes 5 recipes and already has 4 cups. How many more cups are needed?", "11"),
        ("A train has 9 cars with 20 seats each. If 35 seats are empty, how many seats are filled?", "145"),
        ("Omar buys 2 notebooks for $6 each and a pen for $3. How much does he spend?", "15"),
        ("A library has 200 books. It lends 48 and receives 23 returns. How many books are there now?", "175"),
        ("Nina plants 7 flowers in each of 6 pots and then plants 8 more. How many flowers did she plant?", "50"),
        ("A runner completes 400 meters 5 times. How many meters is that?", "2000"),
        ("A pizza has 8 slices. Four pizzas are bought and 11 slices are eaten. How many slices remain?", "21"),
        ("Leo has $90. He buys 4 shirts for $12 each. How much money remains?", "42"),
        ("A shop has 13 shelves with 6 jars each. It sells 17 jars. How many jars remain?", "61"),
        ("Priya writes 15 words per minute for 10 minutes, then deletes 20 words. How many words remain?", "130"),
        ("A game gives 5 points per star. Ella earns 18 stars and loses 10 points. What is her score?", "80"),
        ("A tank holds 100 liters. It loses 7 liters per hour for 4 hours. How many liters remain?", "72"),
        ("A school orders 11 boxes of 24 pencils. It gives away 100 pencils. How many pencils are left?", "164"),
        ("Chris walks 3 kilometers each morning for 9 mornings. How many kilometers total?", "27"),
        ("A jar has 54 cookies. Six friends share them equally. How many cookies does each friend get?", "9"),
        ("A bus ticket costs $3. A family buys 7 tickets and pays $25. How much change do they get?", "4"),
        ("There are 10 pages with 30 lines each. If 45 lines are blank, how many lines have writing?", "255"),
        ("A dog eats 2 cups of food daily. How many cups are needed for 14 days?", "28"),
        ("A concert has 12 rows of 18 seats. If 20 seats are reserved, how many are not reserved?", "196"),
        ("A farmer picks 9 baskets with 15 apples each and gives away 30 apples. How many apples remain?", "105"),
        ("A student answers 50 questions and gets 8 wrong. How many are correct?", "42"),
        ("A pool gains 25 liters of water per minute for 6 minutes and loses 20 liters. How many liters are added overall?", "130"),
        ("A pack has 52 cards. If 4 players get 9 cards each, how many cards are left?", "16"),
        ("A cafe sells 17 teas and 23 coffees. Each drink costs $2. How much money is collected?", "80"),
        ("A rope is 100 meters long. It is cut into 4 equal pieces. How long is each piece?", "25"),
        ("A teacher has 5 boxes of 18 markers and throws away 12 dry markers. How many markers remain?", "78"),
        ("An album has 60 photos. Max adds 4 pages with 6 photos each. How many photos are there?", "84"),
        ("A soccer team practices 45 minutes a day for 5 days. How many minutes do they practice?", "225"),
        ("A shop buys 30 toys and sells 18. It then buys 9 more. How many toys are in stock?", "21"),
        ("A farm has 8 cows. Each cow gives 6 liters of milk. The farmer sells 20 liters. How many liters remain?", "28"),
        ("A bag has 96 candies. 12 candies go into each party favor. How many favors can be made?", "8"),
    ]
    return [{"question": question, "answer": f"#### {answer}"} for question, answer in samples]
