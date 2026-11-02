"""Search episodes: the answer's grade (normalised exact match, else the BrowseComp-Plus judge), the episode record
(search_outcome) and the group's advantages.

Judge: BrowseComp-Plus's grader prompt and reply parser (scripts_evaluation/evaluate_run.py; its call's top_p 0.8,
top_k 20, 4096 tokens, enable_thinking false) on an OpenAI-compatible chat endpoint, shared by training (grade) and
evaluation (bcp, dev_eval), as CLM runs it: judge model Qwen3.5-27B (BCP's: Qwen3-32B) at temperature 0 in both
(BCP's evaluation: 0.7; OCTOPUS_JUDGE_TEMPERATURE). grade's settings: OCTOPUS_JUDGE_URL (base URL ending
in /v1; unset = exact match only), OCTOPUS_JUDGE_MODEL, OCTOPUS_JUDGE_KEY (optional bearer), OCTOPUS_JUDGE_TEMPERATURE
(0), OCTOPUS_JUDGE_TIMEOUT (s, 600), OCTOPUS_JUDGE_CACHE (a JSONL of past verdicts, shared by concurrent episodes,
keyed on the model the endpoint serves, judge_identity: its SGLang /model_info model_path, not its served name;
unset = no cache: evaluation unsets it). A judge that fails raises: octopus/envs/search/agent.py records it as the
episode's error, so the episode is rejected, never trained on. A reply without a verdict is not a failure: the judge
is asked once more for the two lines alone (JUDGE_FOLLOW_UP); still none, grade grades the answer incorrect
("judge_unparsed", counted in the episode record).
"""

from __future__ import annotations

import functools
import json
import os
import re
import string
import urllib.request
from pathlib import Path

from octopus.rl.group import episode_record, group_step, search_gate
from octopus.runtime.core import post_chat, tree_branches

# BrowseComp-Plus's grader (scripts_evaluation/evaluate_run.py GRADER_TEMPLATE; not simple-evals' BrowseComp prompt)
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. 

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()
# a reply without 'correct: yes/no' (Qwen3.5-27B at temperature 0, about 1 in 1200: a long reasoning that ends on the
# confidence line alone) is asked for its verdict on the same conversation: the same prompt again gives the same reply
JUDGE_FOLLOW_UP = "Give your judgement as these two lines only:\ncorrect: yes or no\nconfidence: a number from 0 to 100"


def normalize(text: str) -> str:
    """Lower case, no punctuation, no articles, single spaces (SQuAD); \\boxed{...} and \\text{...} unwrapped (their
    braces and a "$" go with the punctuation), as 29% of OpenResearcher questions ask for a boxed answer."""
    text = re.sub(r"\\(?:boxed|text)\s*\{", "", str(text))
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(w for w in text.split() if w not in ("a", "an", "the"))


def numbers(text: str) -> list[str]:
    """The numbers in text as written, with their sign and inner punctuation ("-4", "3.5", "1:30")."""
    return re.findall(r"-?\d+(?:[.,:/]\d+)*", str(text))


def exact_match(response: str, gold: str) -> bool:
    """Normalised texts equal, and the same numbers as written: normalize's dropped punctuation would make "3.5" and
    "35", "-4" and "4", "1:30" and "130" match. A number written another way ("1,000" / "1000") goes to the judge."""
    return normalize(response) == normalize(gold) != "" and numbers(response) == numbers(gold)


def parse_judge(text: str) -> tuple[bool | None, float | None]:
    """(correct, confidence) as evaluate_run.parse_judge_response reads them: **x:**, then **x**:, then x:."""
    def field(name: str, value: str) -> str | None:
        for pattern in (rf"\*\*{name}:\*\*\s*{value}", rf"\*\*{name}\*\*:\s*{value}", rf"{name}:\s*{value}"):
            if match := re.search(pattern, text, re.I):
                return match.group(1)
        return None
    correct, confidence = field("correct", "(yes|no)"), field("confidence", r"(\d+(?:\.\d+)?)")
    return (correct.lower() == "yes" if correct else None), (min(float(confidence), 100) if confidence else None)


@functools.cache
def judge_identity(url: str, model: str, key: str | None = None) -> str:
    """The model behind the judge endpoint's served name `model` (e.g. "judge"): its SGLang server's /model_info
    model_path (its /v1/models root is the served name). OCTOPUS_JUDGE_CACHE keys on it, so a judge swapped under the
    same served name (Qwen3-32B -> Qwen3.5-27B) never reads the old one's verdicts. Raises when /model_info cannot be
    read or does not serve `model`."""
    info_url = url.rstrip("/").removesuffix("/v1") + "/model_info"
    headers = {"User-Agent": "octopus-judge/1.0", **({"Authorization": "Bearer " + key} if key else {})}
    try:
        with urllib.request.urlopen(urllib.request.Request(info_url, headers=headers), timeout=60) as r:
            info = json.load(r)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"judge endpoint {info_url} cannot be read: {e}") from e
    if info.get("served_model_name") != model or not info.get("model_path"):
        raise RuntimeError(f"judge endpoint {info_url} gives no model_path for the served model {model!r}")
    return info["model_path"]

def judge(question: str, response: str, gold: str, url: str, model: str, key: str | None = None,
          temperature: float = 0.0) -> tuple[bool | None, float | None]:
    """The judge's (correct, confidence) on the response, from OCTOPUS_JUDGE_CACHE when it has this verdict (per
    judge_identity and temperature); raises when the endpoint fails (after 3 tries). A reply without a verdict is
    followed by JUDGE_FOLLOW_UP on the same conversation; without one still, correct is None. Only verdicts are
    cached."""
    cache = os.environ.get("OCTOPUS_JUDGE_CACHE")
    cache_key = cache and ["bcp", judge_identity(url, model, key), temperature, question, response, gold]
    if cache and os.path.exists(cache):
        with open(cache) as handle:
            for line in handle:
                row = json.loads(line)
                if row["key"] == cache_key:
                    return row["correct"], row["confidence"]
    body = {"model": model, "temperature": temperature, "top_p": 0.8, "top_k": 20, "max_tokens": 4096,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": GRADER_TEMPLATE.format(
                question=question, response=response, correct_answer=gold)}]}
    headers = {"Content-Type": "application/json", "User-Agent": "octopus-judge/1.0"}
    if key:
        headers["Authorization"] = "Bearer " + key
    def ask() -> str:
        out = post_chat(url.rstrip("/") + "/chat/completions", body, headers,
                        timeout=int(os.environ.get("OCTOPUS_JUDGE_TIMEOUT", "600")), attempts=3)
        return out["choices"][0]["message"].get("content") or ""
    reply = ask()
    correct, confidence = parse_judge(reply)
    if correct is None:
        body["messages"] += [{"role": "assistant", "content": reply}, {"role": "user", "content": JUDGE_FOLLOW_UP}]
        correct, confidence = parse_judge(ask())
    if correct is None:
        return None, confidence
    if cache:
        with open(cache, "a") as handle:  # one short line per append: concurrent episodes do not interleave
            handle.write(json.dumps({"key": cache_key, "correct": correct, "confidence": confidence}) + "\n")
    return correct, confidence


def grade(question: str, response: str | None, gold: str) -> tuple[bool, str]:
    """(correct, how): "em", "judge", "judge_unparsed" (incorrect: the judge gave no verdict, asked twice) or "none"
    (no answer / no judge configured)."""
    if not response:
        return False, "none"
    if exact_match(response, gold):
        return True, "em"
    if not os.environ.get("OCTOPUS_JUDGE_URL"):
        return False, "none"
    correct, _ = judge(question, response, gold, os.environ["OCTOPUS_JUDGE_URL"],
                       os.environ.get("OCTOPUS_JUDGE_MODEL", "judge"), os.environ.get("OCTOPUS_JUDGE_KEY"),
                       float(os.environ.get("OCTOPUS_JUDGE_TEMPERATURE", "0")))
    return (False, "judge_unparsed") if correct is None else (correct, "judge")


def search_outcome(trial: Path, requests: list[dict]) -> dict:
    """Search's episode record (octopus/rl/group.py's contract): group.episode_record with the grades
    octopus/envs/search/agent.run_trial wrote to verifier/answers.json: y main's answer, the control's y, and the own
    grade of each reader and read control (their done's answer), the only leaves search grades; every other leaf
    None. Recorded only: rewinds, the tree's rewinds (OCTOPUS_REWIND_TOOL; main's and every branch's): calls, the
    steps they removed and the characters of the notes that replaced them; judge_unparsed, the graded answers the
    judge gave no verdict on (grade: incorrect)."""
    answers = json.loads((trial / "verifier/answers.json").read_text())
    tree = json.loads((trial / "agent/tree.json").read_text())
    grades = {k: a["correct"] for k, a in answers.items() if k not in ("main", "control")}
    control_y = answers["control"]["correct"] if "control" in answers else None
    record = episode_record(trial, tree, requests, answers["main"]["correct"], grades, control_y)
    # rl.branch_credit gate (group.search_branch_ok): each branch's done answer and the search/open calls of its
    # subtree (itself and its branches at any depth: a branch may answer from its branches' findings); a thread's own
    # are those after its fork's prefix_len (its messages start with the forking thread's history at the fork) and
    # those its rewinds removed
    nodes = list(tree_branches(tree.get("forks") or []))
    own = {name: sum((c.get("function") or {}).get("name") in ("search", "open")
                     for m in b["messages"][fork["prefix_len"]:] + rewound(b.get("rewinds", []))
                     if m.get("role") == "assistant" for c in m.get("tool_calls") or []) for name, fork, b in nodes}
    for branch, (name, _, b) in zip(record["branches"], nodes, strict=True):
        branch["summary"] = b["summary"]
        branch["searched"] = sum(n for k, n in own.items() if k == name or k.startswith(name + "/"))
    # (.get: a tree from before tree.json kept them, re-graded)
    rewinds = tree.get("rewinds", []) + [r for _, _, b in tree_branches(tree.get("forks") or [])
                                         for r in b.get("rewinds", [])]
    record["rewinds"] = {"calls": len(rewinds),
                         "steps": sum(m.get("role") == "assistant" for m in rewound(rewinds)),
                         "note_chars": sum(len(r.get("note", "")) for r in rewinds)}
    record["judge_unparsed"] = sum(a.get("how") == "judge_unparsed" for a in answers.values())
    return record


def rewound(rewinds: list[dict]) -> list[dict]:
    """The messages a thread's rewinds removed from its history (tree.json: main's rewinds, a branch's)."""
    return [m for r in rewinds for m in r.get("messages", [])]


def search_read_rewards(episode: dict, c: dict, b: dict) -> tuple[float, float]:
    """Search's read_rewards (group.read_values' hook): (r_reader, r_rc) for read control c of branch b, each leaf's own
    grade on the answer's scale 2y-1: its result (its done's answer, what main gets; none when it was stopped) graded
    like the answer (passed, search_outcome)."""
    return 2.0 * b["passed"] - 1.0, 2.0 * c["passed"] - 1.0


def search_group_step(episodes: list[dict], branch_credit: str = "outcome") -> list[dict]:
    """octopus.rl.group.group_step on a group of search episodes, read pairs graded on their own result
    (search_read_rewards); rl.branch_credit "outcome" or "gate" (search's rule: group.search_gate)."""
    return group_step(episodes, read_rewards=search_read_rewards, branch_credit=branch_credit, gate=search_gate)
