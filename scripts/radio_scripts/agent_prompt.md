# Claude 에이전트에게 주는 지시 틀

`brief.md` 와 함께 대본 한 편을 쓰는 에이전트(Claude Code `Agent`, general-purpose)에게 주는 지시. 10-19 set10 에 쓴 그대로이고,
`{곡 목록}` 에 맡길 곡 두세 개(키 — 곡 — 가수)를, `{저장 폴더}` 에 `var/radio_script_drafts/<묶음>` 을 넣는다. 에이전트마다 곡을
나눠 맡겨 곡이 겹치지 않게 한다. 곡이 소재로 정해진 편은 곡을 하나만 주고 "The subject is the song itself" 를 더하며, 이전에
확인한 사실이 있으면 해석·결론·대사 번호를 뺀 사실 목록만 넘긴다(넘긴 조사의 각도로 이야기가 끌려갈 수 있다).

모델은 Opus 5.5, 추론 medium 으로 한 편에 1분 반~4분 40초(`var/radio_script_drafts/timings.md`). 이 Pi 에서 다섯씩 돌려도
메모리 문제는 없었다(에이전트는 무거운 프로세스를 띄우지 않는다).

```
You are writing one Korean radio script for a small desk robot named Ray. Your full instructions are in
`/home/delight/Ray-heon/scripts/radio_scripts/brief.md` — read it first and follow it exactly (it is in Korean). Pay
particular attention to the sections "말맛" and "이야기의 틀": the script is read aloud and should sound alive to the ear,
not like a report of facts.

Songs you may use — pick exactly one (key — title — artist):
{곡 목록}

The subject may be the song itself or something else (a work, a word, an object, a scene from history…). If it is not
the song, decide the subject and the conclusion first, then pick the song from your list that connects to the
conclusion. Other writers have other songs; you choose your own subject.

Ground rules:
- Read only the brief and web pages you search. Do NOT read `assets/radio_scripts/`, `docs/`, `var/radio_script_drafts/`,
  or any other project file. The one exception is the format-check command below, which you just run.
- Verify every fact with web search (including anything about the chosen song: lyrics, who sings this version, its
  background); leave out what you cannot verify, phrasing weakly sourced anecdotes as hearsay as the brief says. If
  dropping a fact breaks the story, take the story another way instead of compressing the gap.
- Write only two files: `{저장 폴더}/<key>.md` and `{저장 폴더}/<key>.sources.md` (`<key>`: short lowercase English
  snake_case for your subject; if the name exists, choose another).
- Put a space after every sentence-ending `.`, `?`, `!` inside a line — the checker splits sentences on punctuation
  followed by whitespace.

When the script is written, run this format check (replace KEY) and fix every problem until it prints OK:

    cd /home/delight/Ray-heon && uv run python - <<'PY'
    from pathlib import Path
    from voice_pipeline.engines.gpt_live.radio import check_script, _parse_frontmatter
    p = Path("{저장 폴더}/KEY.md")
    fields, body = _parse_frontmatter(p.read_text(encoding="utf-8"))
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    print(p.name, fields, len(lines), "lines", check_script(lines) or "OK")
    PY

The checker does not cover every rule in the brief (the opening, the conclusion rule, 말맛, no words that assume an
earlier story, no time-of-day or season assertions, first line not starting or ending with "~해 볼게요", 14–20 lines, no
"~대요"/"~래요", the passage into the song, last line introducing the song with singer and title, no wording carried over
from the brief). Check those yourself, and before finishing read the script once as if hearing it aloud and fix
anything flat or list-like.

Final report (short, in Korean): the two file paths, the checker output, the subject, the chosen song and why, the first
line, the conclusion sentence, and any fact you dropped for lack of verification. Do not paste the whole script.
```
