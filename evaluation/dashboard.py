"""Generate an HTML dashboard from scored eval results.

Usage:
    uv run python -m evaluation.dashboard data/eval/results/scored.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif; background: #f5f5f7; color: #1d1d1f; font-size: 14px; line-height: 1.5; }

/* Header */
.header { background: #1d1d1f; color: #f5f5f7; padding: 12px 24px; display: flex; align-items: center; gap: 16px; }
.header h1 { font-size: 16px; font-weight: 600; white-space: nowrap; }
.header .meta { font-size: 12px; color: #86868b; }

/* Tabs */
.tabs { background: #fff; border-bottom: 1px solid #d2d2d7; padding: 0 24px; display: flex; gap: 0; position: sticky; top: 0; z-index: 10; }
.tab { padding: 10px 20px; font-size: 13px; font-weight: 500; color: #86868b; cursor: pointer; border-bottom: 2px solid transparent; transition: color 0.15s, border-color 0.15s; white-space: nowrap; }
.tab:hover { color: #1d1d1f; }
.tab.active { color: #1d1d1f; border-bottom-color: #0071e3; }

/* Content */
.content { padding: 16px 24px; }
.tab-pane { display: none; }
.tab-pane.active { display: block; }

/* Cards row */
.cards { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
.card { background: #fff; border-radius: 10px; padding: 12px 16px; min-width: 140px; max-width: 200px; }
.card-label { font-size: 11px; color: #86868b; font-weight: 500; letter-spacing: 0.02em; }
.card-value { font-size: 28px; font-weight: 700; line-height: 1.2; margin-top: 2px; }
.card-sub { font-size: 11px; color: #86868b; margin-top: 4px; }
.card-value .unit { font-size: 14px; font-weight: 500; }

/* Score colors */
.good { color: #248a3d; }
.warn { color: #b25000; }
.bad { color: #d70015; }

/* Tags */
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
.tag-ok { background: #d1fae5; color: #065f46; }
.tag-fail { background: #fee2e2; color: #991b1b; }
.tag-warn { background: #fef3c7; color: #92400e; }
.tag-mute { background: #f3f4f6; color: #6b7280; }

/* Sections */
.section { margin-bottom: 20px; }
.section-title { font-size: 13px; font-weight: 600; color: #1d1d1f; margin-bottom: 8px; }
.cat-header { font-size: 16px; font-weight: 700; color: #1d1d1f; margin: 24px 0 8px; padding-bottom: 6px; border-bottom: 2px solid #1d1d1f; }

/* Panel */
.panel { background: #fff; border-radius: 10px; padding: 14px 18px; margin-bottom: 12px; }

/* Compact table */
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align: left; padding: 6px 10px; color: #86868b; font-weight: 500; font-size: 11px; border-bottom: 1px solid #e5e7eb; }
td { padding: 6px 10px; border-bottom: 1px solid #f3f4f6; }
tr:last-child td { border-bottom: none; }

/* Key-value pairs */
.kv-group { margin-bottom: 10px; }
.kv-group-title { font-size: 11px; font-weight: 600; color: #86868b; margin-bottom: 4px; letter-spacing: 0.03em; }
.kv-grid { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; font-size: 13px; }
.kv-grid.lg { font-size: 14px; gap: 6px 16px; }
.kv-key { color: #86868b; white-space: nowrap; }
.kv-val { color: #1d1d1f; font-weight: 500; }
.kv-row { display: flex; gap: 8px; padding: 3px 0; font-size: 13px; }

/* Overview two-panel layout */
.overview-panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
.overview-panels .panel { margin-bottom: 0; }
.overview-panels .kv-group-title { font-size: 12px; margin-bottom: 8px; }

/* Item list (dense) */
.item { background: #fff; border-radius: 8px; padding: 10px 14px; margin-bottom: 6px; }
.item-header { display: flex; align-items: center; gap: 8px; font-size: 13px; }
.item-id { font-weight: 600; color: #1d1d1f; min-width: 70px; }
.item-suite { font-size: 11px; color: #86868b; }
.item-body { margin-top: 4px; font-size: 13px; color: #424245; }
.item-row { display: flex; gap: 8px; margin-top: 2px; }
.item-label { font-size: 11px; color: #86868b; min-width: 40px; flex-shrink: 0; }
.item-text { font-size: 13px; }
.item-scores { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
.score-chip { font-size: 11px; padding: 1px 6px; border-radius: 3px; background: #f3f4f6; }
.score-chip.high { background: #d1fae5; color: #065f46; }
.score-chip.mid { background: #fef3c7; color: #92400e; }
.score-chip.low { background: #fee2e2; color: #991b1b; }
.reasoning { font-size: 12px; color: #86868b; margin-top: 4px; font-style: italic; }

/* Suite group */
.suite-group { margin-bottom: 16px; }
.suite-header { font-size: 12px; font-weight: 600; color: #424245; padding: 6px 0; margin-bottom: 4px; border-bottom: 1px solid #e5e7eb; display: flex; align-items: center; gap: 8px; }

/* Diff highlight */
.diff-mark { background: #fef3c7; padding: 1px 4px; border-radius: 2px; }

/* Latency bar */
.lat-bar { display: inline-block; height: 5px; border-radius: 2px; vertical-align: middle; }
.lat-bar-bg { display: inline-block; width: 100px; height: 5px; background: #e5e7eb; border-radius: 2px; vertical-align: middle; }

/* Memory subsection */
.subsection { margin-top: 16px; padding-top: 12px; border-top: 1px solid #e5e7eb; }
.subsection-title { font-size: 12px; font-weight: 600; color: #424245; margin-bottom: 8px; }

/* Mono */
.mono { font-family: 'SF Mono', 'Consolas', 'Monaco', monospace; font-size: 12px; }
.mute { color: #86868b; }

/* Collapsible */
.collapsible-toggle { cursor: pointer; user-select: none; }
.collapsible-toggle::before { content: '▸ '; font-size: 10px; }
.collapsible-toggle.open::before { content: '▾ '; }
.collapsible-body { display: none; }
.collapsible-body.open { display: block; }

/* Question stage detail (per-question drill-down) */
.qd-toggle { display: inline-block; margin: 4px 0 2px; font-size: 11px; color: #0071e3; font-weight: 500; }
.qd-detail { margin: 4px 0 10px; padding: 10px 12px; background: #f9f9fb; border: 1px solid #e5e7eb; border-radius: 6px; }
.qd-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 8px 18px; }
.qd-sec-title { font-size: 11px; font-weight: 700; color: #1d1d1f; margin-bottom: 3px; }
.qd-line { font-size: 12px; color: #424245; padding: 1px 0; line-height: 1.45; }
.qd-k { color: #86868b; display: inline-block; min-width: 84px; }
.qd-mono { font-family: 'SF Mono', 'Consolas', 'Monaco', monospace; font-size: 11px; word-break: break-all; }

/* Sub-tabs */
.sub-tab.active { color: #1d1d1f !important; border-bottom-color: #0071e3 !important; }
.sub-pane { display: none; }
.sub-pane.active { display: block; }

/* Empty state */
.empty { color: #86868b; font-size: 13px; padding: 20px; text-align: center; }
"""

# ---------------------------------------------------------------------------
# JS
# ---------------------------------------------------------------------------

_JS = """\
document.addEventListener('DOMContentLoaded', function() {
    const tabs = document.querySelectorAll('.tab');
    const panes = document.querySelectorAll('.tab-pane');
    tabs.forEach(function(tab) {
        tab.addEventListener('click', function() {
            tabs.forEach(function(t) { t.classList.remove('active'); });
            panes.forEach(function(p) { p.classList.remove('active'); });
            tab.classList.add('active');
            document.getElementById(tab.dataset.target).classList.add('active');
        });
    });

    document.querySelectorAll('.sub-tab').forEach(function(tab) {
        tab.addEventListener('click', function() {
            var parent = tab.parentElement.parentElement;
            parent.querySelectorAll('.sub-tab').forEach(function(t) { t.classList.remove('active'); });
            parent.querySelectorAll('.sub-pane').forEach(function(p) { p.classList.remove('active'); p.style.display = 'none'; });
            tab.classList.add('active');
            var target = parent.querySelector('#' + tab.dataset.subtarget);
            if (target) { target.classList.add('active'); target.style.display = 'block'; }
        });
    });

    document.querySelectorAll('.collapsible-toggle').forEach(function(toggle) {
        toggle.addEventListener('click', function() {
            toggle.classList.toggle('open');
            var body = toggle.nextElementSibling;
            if (body && body.classList.contains('collapsible-body')) {
                body.classList.toggle('open');
            }
        });
    });
});
"""

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

_CRITERION_LABELS = {
    "relevance": "관련성",
    "voice_appropriateness": "음성 적합성",
    "naturalness": "자연스러움",
    "correctness": "정확성",
    "helpfulness": "유용성",
    "engagement": "대화 참여",
    "empathy": "공감",
    "format_adaptation": "포맷 적응",
    "context_coherence": "맥락 유지",
    "correction_quality": "교정 품질",
    "boundary_communication": "한계 전달",
}

_MEMORY_CRITERION_LABELS = {
    "response_relevance": "응답 관련성",
    "memory_appropriateness": "메모리 적절성",
    "factual_accuracy": "사실 정확성",
    "naturalness": "자연스러움",
}

_LATENCY_LABELS = {
    "turn_detection_delay": "턴 감지",
    "turn_shift_to_playback": "응답 생성",
    "llm_ttft": "LLM 첫 토큰",
    "tts_ttfc": "TTS 첫 청크",
    "bridge": "Bridge 전송",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_cls(value: float, good: float, bad: float) -> str:
    if value >= good:
        return "good"
    if value <= bad:
        return "bad"
    return "warn"


def _score_cls_lower_better(value: float, good: float, bad: float) -> str:
    if value <= good:
        return "good"
    if value >= bad:
        return "bad"
    return "warn"


def _score_chip(value: float, max_val: float = 5.0) -> str:
    if value >= max_val * 0.8:
        cls = "high"
    elif value >= max_val * 0.6:
        cls = "mid"
    else:
        cls = "low"
    return f'<span class="score-chip {cls}">{value:.1f}</span>'


def _wer_tag(wer: float) -> str:
    if wer == 0:
        return '<span class="tag tag-ok">정확</span>'
    if wer <= 0.15:
        return f'<span class="tag tag-warn">{wer:.0%}</span>'
    return f'<span class="tag tag-fail">{wer:.0%}</span>'


def _outcome_tag(outcome: str | None) -> str:
    tags = {
        "truncated": ("tag-ok", "중단됨"),
        "completed": ("tag-fail", "미감지"),
        "cancelled": ("tag-warn", "취소됨"),
    }
    if outcome in tags:
        cls, label = tags[outcome]
        return f'<span class="tag {cls}">{label}</span>'
    return f'<span class="tag tag-mute">{outcome or "—"}</span>'


def _error_tag(error: str | None) -> str:
    if not error:
        return ""
    labels = {
        "no_response": "무응답",
        "no_recognition": "인식 실패",
        "no_turn_shift": "턴 감지 실패",
        "early_turn_shift": "조기 턴 전환",
        "late_turn_shift": "지연 턴 전환",
        "premature_turn_shift": "섣부른 턴 전환 (대기 필요)",
        "generation_failed": "생성 실패",
        "incomplete": "미완료",
    }
    label = labels.get(error, error)
    return f'<span class="tag tag-fail">{label}</span>'


def _call_issues_tag(issues: dict) -> str:
    parts = []
    retry = issues.get("retry_count", 0)
    error = issues.get("error_count", 0)
    if retry:
        parts.append(f"재시도 {retry}")
    if error:
        parts.append(f"API 오류 {error}")
    label = " · ".join(parts)
    cls = "tag-fail" if error else "tag-warn"
    return f'<span class="tag {cls}">{label}</span>'


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# Per-question stage detail (inline drill-down)
# ---------------------------------------------------------------------------

_QD_LATENCY_ORDER = (
    "memory_ms",
    "context_ms",
    "llm_ttft_ms",
    "llm_ms",
    "tts_ttfc_ms",
    "tts_ms",
    "speculative_ms",
    "prepare_to_streaming_ms",
    "turn_shift_to_playback_ms",
    "bridge_ms",
    "interrupt_latency_ms",
)


def _qd_section(title: str, lines: list[str]) -> str:
    if not lines:
        return ""
    body = "".join(f'<div class="qd-line">{x}</div>' for x in lines)
    return f'<div><div class="qd-sec-title">{title}</div>{body}</div>'


def _build_question_detail(turn: dict) -> str:
    """Flat 7-section per-question stage view for the inline drill-down.

    Lays every pipeline stage of one question/turn side by side so cross-stage
    behavior is visible in one place. Renders only sections that have data
    (ASR-only / text-mode turns omit turn-detection / generation / TTS), and
    returns "" when there is nothing worth a toggle.
    """
    sc = turn.get("stage_calls", {})
    secs: list[str] = []

    # ① ASR
    lines = []
    if turn.get("input_text"):
        lines.append(f'<span class="qd-k">원본</span>{_esc(turn["input_text"])}')
    if turn.get("asr_text"):
        lines.append(f'<span class="qd-k">인식</span>{_esc(turn["asr_text"])}')
    score = turn.get("asr_score")
    if score:
        lines.append(
            f'<span class="qd-k">WER</span>{score["wer"]:.3f} '
            f'<span class="mute">(S{score["substitutions"]} D{score["deletions"]} '
            f"I{score['insertions']} / {score['ref_words']}w)</span>"
        )
    secs.append(_qd_section("① ASR", lines))

    # ② Turn detection
    lines = []
    delay = turn.get("turn_detection_delay_ms")
    if delay is not None:
        lines.append(f'<span class="qd-k">감지 지연</span>{delay:.0f}ms')
    if turn.get("turn_shift_reason"):
        lines.append(f'<span class="qd-k">turn reason</span>{_esc(str(turn["turn_shift_reason"]))}')
    if turn.get("cancelled_turn_shifts"):
        lines.append(f'<span class="qd-k">전환 취소</span>{turn["cancelled_turn_shifts"]}회')
    vap = sc.get("vap")
    if vap:
        extra = f" · overrun {vap['overrun']}" if vap.get("overrun") else ""
        lines.append(
            f'<span class="qd-k">vap</span>{vap["count"]} frames · avg {vap["avg_ms"]}ms · max {vap["max_ms"]}ms{extra}'
        )
    tg = sc.get("turngpt")
    if tg:
        extra = f" · slow {tg['slow']}" if tg.get("slow") else ""
        lines.append(f'<span class="qd-k">turngpt</span>{tg["count"]} calls · avg {tg["avg_ms"]}ms{extra}')
    secs.append(_qd_section("② 턴 감지", lines))

    # ③ Prepare gate — ASR updates are ~all pure appends (new = prev + suffix),
    # so show just the added tail; fall back to full prev → new on divergence.
    events = turn.get("similarity_events", [])
    if events:
        lines = []
        for e in events:
            prev = e.get("prev_text") or ""
            new = e.get("new_text") or ""
            if new.startswith(prev):
                added = new[len(prev) :].lstrip()
                delta = f'<span class="mute">+</span> {_esc(added)}' if added else ""
            else:
                delta = f'<span class="mute">{_esc(prev)} → {_esc(new)}</span>'
            tail = f" {delta}" if delta else ""
            lines.append(
                f'<span class="qd-mono">{e.get("similarity", 0):.3f} vs {e.get("threshold", 0):.2f} → '
                f"<b>{_esc(str(e.get('decision', '?')))}</b></span>{tail}"
            )
        secs.append(_qd_section(f"③ Prepare gate ({len(events)})", lines))

    # ④ Generation
    lines = []
    if turn.get("outcome"):
        spec = turn.get("speculative_attempts")
        spec_str = f" · speculative {spec}" if spec else ""
        lines.append(f'<span class="qd-k">outcome</span>{_esc(str(turn["outcome"]))}{spec_str}')
    lat = turn.get("latency", {})
    for col in _QD_LATENCY_ORDER:
        if lat.get(col):
            lines.append(f'<span class="qd-k">{col[:-3]}</span>{lat[col]:.0f}ms')
    secs.append(_qd_section("④ 생성", lines))

    # ⑤ TTS — the stream op carries the real timing (ttfc = perceived latency);
    # synthesize is ~0ms for lazy-generator vendors (ElevenLabs), so skip those.
    lines = []
    for c in sc.get("tts", []):
        op = str(c.get("operation", ""))
        if op == "synthesize" and c["elapsed_ms"] < 1:
            continue
        st = "" if c.get("status") == "ok" else f' <span class="bad">{_esc(str(c.get("status", "")))}</span>'
        if op == "stream":
            ttfc = c.get("ttfc_ms")
            ttfc_str = f"ttfc {ttfc:.0f}ms · " if ttfc else ""
            audio = c.get("audio_sec")
            audio_str = f" · {audio}s audio" if audio else ""
            lines.append(f'<span class="qd-mono">stream</span> {ttfc_str}{c["elapsed_ms"]:.0f}ms total{audio_str}{st}')
        else:
            lines.append(f'<span class="qd-mono">{_esc(op)}</span> {c["elapsed_ms"]:.0f}ms{st}')
    secs.append(_qd_section("⑤ TTS", lines))

    # ⑥ Response & quality
    lines = []
    if turn.get("response_text"):
        lines.append(f'<span class="qd-k">응답</span>{_esc(turn["response_text"][:300])}')
    qs = turn.get("quality_scores")
    if qs:
        chips = " · ".join(f"{_CRITERION_LABELS.get(k, k)} {v}" for k, v in qs.items())
        lines.append(f'<span class="qd-k">품질</span>{chips}')
    secs.append(_qd_section("⑥ 응답·품질", lines))

    # ⑦ call_records timeline + API issues — one module.operation per line
    lines = []
    counts = sc.get("counts")
    if counts:
        lines += [f'<span class="qd-mono">{_esc(k)}</span> <span class="mute">×{v}</span>' for k, v in counts.items()]
    ci = turn.get("call_issues")
    if ci and (ci.get("retry_count") or ci.get("error_count")):
        bits = []
        if ci.get("retry_count"):
            bits.append(f"재시도 {ci['retry_count']}")
        if ci.get("error_count"):
            bits.append(f"API 오류 {ci['error_count']}")
        lines.append(f'<span class="qd-k">issues</span><span class="bad">{" · ".join(bits)}</span>')
    if lines:
        secs.append(_qd_section("⑦ call_records", lines))

    secs = [s for s in secs if s]
    if not secs:
        return ""
    return '<div class="qd-grid">' + "".join(secs) + "</div>"


def _detail_block(turn: dict, label: str = "단계 상세") -> str:
    """Collapsible toggle + detail body for one question (reuses .collapsible JS)."""
    detail = _build_question_detail(turn)
    if not detail:
        return ""
    return (
        f'<div class="collapsible-toggle qd-toggle">{_esc(label)}</div>'
        f'<div class="collapsible-body qd-detail">{detail}</div>'
    )


# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------


def _build_overview(scored: dict) -> str:
    scores = scored.get("scores", {})
    asr = scores.get("asr", {})
    latency = scores.get("latency", {})
    interruption = scores.get("interruption", {})
    quality = scores.get("quality", {})
    memory = scores.get("memory", {})

    p = []

    # --- Metadata ---
    started = scored.get("started_at", "—")
    finished = scored.get("finished_at", "—")
    total = scored.get("total", 0)
    config = scored.get("config", {})
    pipeline = config.get("pipeline", {})
    runner = config.get("runner", {})

    # Compute duration
    duration_str = ""
    try:
        from datetime import datetime as _dt

        fmt = "%Y-%m-%d %H:%M:%S"
        dt_start = _dt.strptime(started, fmt)
        dt_end = _dt.strptime(finished, fmt)
        delta = dt_end - dt_start
        total_sec = int(delta.total_seconds())
        if total_sec >= 60:
            mins, secs = divmod(total_sec, 60)
            duration_str = f" ({mins}분 {secs}초)"
        else:
            duration_str = f" ({total_sec}초)"
    except Exception:
        pass

    p.append('<div class="overview-panels">')

    # Run info panel
    p.append('<div class="panel">')
    p.append('<div class="kv-group-title">실행</div>')
    p.append('<div class="kv-grid lg">')
    p.append(
        f'<span class="kv-key">시각</span><span class="kv-val">{_esc(started)} — {_esc(finished)}{duration_str}</span>'
    )
    p.append(f'<span class="kv-key">결과</span><span class="kv-val">{total}턴 실행</span>')

    suites = runner.get("suites", [])
    if suites:
        cat_counts: dict[str, int] = {}
        turns = scored.get("turns", [])
        for s_name in suites:
            s_turns = [t for t in turns if t.get("suite_name") == s_name]
            cat = s_name.split("_")[0]
            cat_map = {"asr": "ASR", "tt": "Turn-taking", "int": "Interruption", "lq": "Quality", "mem": "Memory"}
            cat_label = cat_map.get(cat, cat)
            cat_counts[cat_label] = cat_counts.get(cat_label, 0) + len(s_turns)

        cat_parts = []
        for cat_label, count in cat_counts.items():
            cat_parts.append(f"{cat_label} {count}")
        p.append(f'<span class="kv-key">범위</span><span class="kv-val">{" · ".join(cat_parts)}</span>')

    if runner.get("quick"):
        p.append('<span class="kv-key">모드</span><span class="kv-val">Quick (샘플링)</span>')

    p.append("</div></div>")

    # Pipeline config panel
    p.append('<div class="panel">')
    p.append('<div class="kv-group-title">Pipeline</div>')
    p.append('<div class="kv-grid lg">')
    if pipeline:
        if pipeline.get("llm_model"):
            temp = pipeline.get("llm_temperature", "")
            temp_str = f", temp={temp}" if temp != "" else ""
            p.append(
                f'<span class="kv-key">LLM</span><span class="kv-val">{_esc(pipeline["llm_model"])}{temp_str}</span>'
            )
        if pipeline.get("writer_llm_model"):
            p.append(
                f'<span class="kv-key">Writer LLM</span><span class="kv-val">{_esc(pipeline["writer_llm_model"])}</span>'
            )
        if pipeline.get("tts_model"):
            voice = pipeline.get("tts_voice", "")
            voice_str = f", {voice}" if voice else ""
            p.append(
                f'<span class="kv-key">TTS</span><span class="kv-val">{_esc(pipeline["tts_model"])}{voice_str}</span>'
            )
        if pipeline.get("asr_model"):
            lang = pipeline.get("asr_language", "")
            lang_str = f" ({lang})" if lang else ""
            p.append(
                f'<span class="kv-key">ASR</span><span class="kv-val">{_esc(pipeline["asr_model"])}{lang_str}</span>'
            )
        if pipeline.get("vap_model"):
            p.append(f'<span class="kv-key">VAP</span><span class="kv-val">{_esc(pipeline["vap_model"])}</span>')
        if pipeline.get("turngpt_model"):
            p.append(
                f'<span class="kv-key">TurnGPT</span><span class="kv-val">{_esc(pipeline["turngpt_model"])}</span>'
            )
        if pipeline.get("vad_model"):
            p.append(f'<span class="kv-key">VAD</span><span class="kv-val">{_esc(pipeline["vad_model"])}</span>')
    p.append("</div></div>")

    p.append("</div>")

    # --- Summary cards ---
    p.append('<div class="cards">')

    if asr:
        wer = asr.get("mean_wer", 0)
        cls = _score_cls_lower_better(wer, 0.05, 0.2)
        p.append(
            f'<div class="card"><div class="card-label">ASR 정확도</div>'
            f'<div class="card-value {cls}">{(1 - wer):.0%}</div>'
            f'<div class="card-sub">{asr.get("perfect_count", 0)}/{asr.get("total_scored", 0)} 완벽 인식</div></div>'
        )

    ts_pb = latency.get("turn_shift_to_playback_ms", {})
    if ts_pb:
        med = ts_pb.get("median_ms", 0)
        cls = _score_cls_lower_better(med, 1500, 3000)
        p.append(
            f'<div class="card"><div class="card-label">응답 속도</div>'
            f'<div class="card-value {cls}">{med:.0f}<span class="unit">ms</span></div>'
            f'<div class="card-sub">턴 감지 → 재생 (중위값)</div></div>'
        )

    td = latency.get("turn_detection_delay_ms", {})
    if td:
        med = td.get("median_ms", 0)
        cls = _score_cls_lower_better(med, 800, 2000)
        p.append(
            f'<div class="card"><div class="card-label">턴 감지</div>'
            f'<div class="card-value {cls}">{med:.0f}<span class="unit">ms</span></div>'
            f'<div class="card-sub">침묵 → 감지 (중위값)</div></div>'
        )

    if interruption and interruption.get("testable"):
        dr = interruption.get("detection_rate", 0)
        cls = _score_cls(dr, 0.7, 0.5)
        p.append(
            f'<div class="card"><div class="card-label">인터럽션 감지</div>'
            f'<div class="card-value {cls}">{dr:.0%}</div>'
            f'<div class="card-sub">{interruption.get("detected", 0)}/{interruption.get("testable", 0)} 감지</div></div>'
        )

    if quality:
        mean = quality.get("mean_score", 0)
        cls = _score_cls(mean, 4.0, 3.0)
        p.append(
            f'<div class="card"><div class="card-label">응답 품질</div>'
            f'<div class="card-value {cls}">{mean:.1f}<span class="unit">/5</span></div>'
            f'<div class="card-sub">LLM Judge 평균</div></div>'
        )

    mem_q = memory.get("quality", {})
    if mem_q and mem_q.get("mean_score"):
        mq = mem_q["mean_score"]
        cls = _score_cls(mq, 4.0, 3.0)
        p.append(
            f'<div class="card"><div class="card-label">메모리 품질</div>'
            f'<div class="card-value {cls}">{mq:.1f}<span class="unit">/5</span></div>'
            f'<div class="card-sub">기억 활용 품질</div></div>'
        )

    p.append("</div>")

    return "\n".join(p)


_CATEGORY_LABELS = {
    "asr": "ASR",
    "tt": "Turn-taking",
    "int": "Interruption",
    "lq": "Quality",
    "mem": "Memory",
}


def _turn_category(suite_name: str) -> str:
    return suite_name.split("_")[0]


def _order_keys(scored: dict) -> tuple[dict[str, int], dict[str, int]]:
    """Suite / question ordering ranks derived from questions.json.

    `suite_descriptions` and `question_texts` in the run config are built by
    iterating questions.json in order, so their key order is the canonical
    suite / question sequence. Every dashboard listing and grouping sorts by
    these ranks so the displayed order matches questions.json — independent of
    playback order, which the acoustic-bed scheduler may reorder into SNR
    blocks. Unknown ids sort last.
    """
    cfg = scored.get("config", {})
    suite_rank = {name: i for i, name in enumerate(cfg.get("suite_descriptions", {}))}
    q_rank = {qid: i for i, qid in enumerate(cfg.get("question_texts", {}))}
    return suite_rank, q_rank


def _get_suite_desc(scored: dict, suite_name: str) -> str:
    return scored.get("config", {}).get("suite_descriptions", {}).get(suite_name, "")


def _build_asr(scored: dict) -> str:
    turns = scored.get("turns", [])
    asr_turns = [t for t in turns if "asr_score" in t]

    if not asr_turns:
        return '<div class="empty">ASR 데이터 없음</div>'

    suite_rank, q_rank = _order_keys(scored)

    from itertools import groupby

    p = []

    # --- Description ---
    p.append(
        '<div class="panel mute" style="font-size:12px">'
        "원본 텍스트와 음성 인식 결과의 단어 오류율(WER)을 측정. "
        "WER 0%는 완벽 인식, 낮을수록 좋음."
        "</div>"
    )

    # --- Summary ---
    asr_cat_turns = [t for t in asr_turns if _turn_category(t["suite_name"]) == "asr"]
    other_turns = [t for t in asr_turns if _turn_category(t["suite_name"]) != "asr"]

    total_perfect = sum(1 for t in asr_turns if t["asr_score"]["wer"] == 0)
    total_count = len(asr_turns)
    accuracy = total_perfect / total_count if total_count else 0
    cls = _score_cls(accuracy, 0.9, 0.7)

    total_wer = sum(t["asr_score"]["wer"] for t in asr_turns) / total_count if total_count else 0
    wer_cls = _score_cls_lower_better(total_wer, 0.05, 0.2)

    p.append('<div class="cards">')
    p.append(
        f'<div class="card"><div class="card-label">완벽 인식</div>'
        f'<div class="card-value {cls}">{total_perfect}<span class="unit">/{total_count}</span></div></div>'
    )
    p.append(
        f'<div class="card"><div class="card-label">평균 WER</div>'
        f'<div class="card-value {wer_cls}">{total_wer:.1%}</div></div>'
    )
    p.append("</div>")

    p.append('<div class="panel">')
    p.append('<div class="section-title">Suite별 요약</div>')

    # Header row
    p.append(
        '<div style="display:flex;align-items:center;gap:16px;padding:3px 0;font-size:11px;color:#86868b;border-bottom:1px solid #e5e7eb;margin-bottom:4px">'
        '<span style="min-width:180px">Suite</span>'
        '<span style="min-width:120px">설명</span>'
        "<span>완벽 인식</span>"
        "</div>"
    )

    # ASR suites: per-suite with description
    if asr_cat_turns:
        sorted_asr = sorted(asr_cat_turns, key=lambda t: suite_rank.get(t["suite_name"], 10**6))
        for suite_name, group in groupby(sorted_asr, key=lambda t: t["suite_name"]):
            suite_turns = list(group)
            perfect = sum(1 for t in suite_turns if t["asr_score"]["wer"] == 0)
            desc = _get_suite_desc(scored, suite_name)
            all_ok = perfect == len(suite_turns)
            tag_cls = "tag-ok" if all_ok else "tag-warn"
            p.append(
                f'<div style="display:flex;align-items:center;gap:16px;padding:3px 0;font-size:14px">'
                f'<span style="font-weight:500;min-width:180px">{suite_name}</span>'
                f'<span class="mute" style="min-width:120px">{_esc(desc)}</span>'
                f'<span class="tag {tag_cls}">{perfect}/{len(suite_turns)}</span>'
                f"</div>"
            )

    # Other categories: grouped, compact line
    if other_turns:
        p.append(
            '<div style="margin-top:10px;padding-top:10px;border-top:1px solid #e5e7eb;font-size:14px;display:flex;gap:20px;flex-wrap:wrap">'
        )
        sorted_other = sorted(other_turns, key=lambda t: suite_rank.get(t["suite_name"], 10**6))
        for cat, group in groupby(sorted_other, key=lambda t: _turn_category(t["suite_name"])):
            cat_turns = list(group)
            cat_label = _CATEGORY_LABELS.get(cat, cat)
            perfect = sum(1 for t in cat_turns if t["asr_score"]["wer"] == 0)
            all_ok = perfect == len(cat_turns)
            tag_cls = "tag-ok" if all_ok else "tag-warn"
            p.append(f'<span>{cat_label} <span class="tag {tag_cls}">{perfect}/{len(cat_turns)}</span></span>')
        p.append("</div>")

    p.append("</div>")

    # --- Voice summary ---
    by_voice = scored.get("scores", {}).get("asr", {}).get("by_voice", {})
    if by_voice:
        p.append('<div class="panel">')
        p.append('<div class="section-title">Voice별 요약</div>')
        p.append(
            '<div style="display:flex;align-items:center;gap:16px;padding:3px 0;font-size:11px;color:#86868b;border-bottom:1px solid #e5e7eb;margin-bottom:4px">'
            '<span style="min-width:100px">Voice</span>'
            '<span style="min-width:80px">평균 WER</span>'
            "<span>완벽 인식</span>"
            "</div>"
        )
        for voice, stats in by_voice.items():
            v_wer = stats.get("mean_wer", 0)
            v_cls = _score_cls_lower_better(v_wer, 0.05, 0.2)
            v_perfect = stats.get("perfect_count", 0)
            v_total = stats.get("total_scored", 0)
            all_ok = v_perfect == v_total
            tag_cls = "tag-ok" if all_ok else "tag-warn"
            p.append(
                f'<div style="display:flex;align-items:center;gap:16px;padding:3px 0;font-size:14px">'
                f'<span style="font-weight:500;min-width:100px">{_esc(voice)}</span>'
                f'<span class="{v_cls}" style="min-width:80px">{v_wer:.1%}</span>'
                f'<span class="tag {tag_cls}">{v_perfect}/{v_total}</span>'
                f"</div>"
            )
        p.append("</div>")

    # --- Per-question results: questions.json order (category > suite > question) ---
    all_sorted = sorted(asr_turns, key=lambda t: q_rank.get(t["question_id"], 10**6))

    current_cat = None
    current_suite = None
    for t in all_sorted:
        cat = _turn_category(t["suite_name"])
        suite = t["suite_name"]

        # Category header
        if cat != current_cat:
            if current_cat is not None:
                p.append("</div>")
            current_cat = cat
            current_suite = None
            cat_label = _CATEGORY_LABELS.get(cat, cat)
            p.append(f'<div class="section"><div class="cat-header">{cat_label}</div>')

        # Suite sub-header
        if suite != current_suite:
            current_suite = suite
            desc = _get_suite_desc(scored, suite)
            desc_str = f' <span class="mute">— {_esc(desc)}</span>' if desc else ""
            p.append(f'<div class="suite-header">{suite}{desc_str}</div>')

        wer = t["asr_score"]["wer"]
        asr_text = t.get("asr_text") or ""
        sys_text = t.get("system_text") or ""
        has_diff = sys_text and sys_text != asr_text

        voice = t.get("voice", "")
        voice_tag = f'<span class="tag" style="font-size:10px;margin-left:4px">{_esc(voice)}</span>' if voice else ""

        p.append('<div class="item">')
        p.append(
            f'<div class="item-header"><span class="item-id">{t["question_id"]}</span>{voice_tag}{_wer_tag(wer)}</div>'
        )
        p.append('<div class="item-body">')
        p.append(
            f'<div class="item-row"><span class="item-label mute">원본</span><span class="item-text">{_esc(t["input_text"])}</span></div>'
        )
        p.append(
            f'<div class="item-row"><span class="item-label mute">인식</span><span class="item-text">{_esc(asr_text)}</span></div>'
        )
        if has_diff:
            p.append(
                f'<div class="item-row"><span class="item-label mute">시스템</span>'
                f'<span class="item-text diff-mark">{_esc(sys_text)}</span></div>'
            )
        p.append("</div>")
        p.append(_detail_block(t))
        p.append("</div>")

    if current_cat is not None:
        p.append("</div>")

    return "\n".join(p)


def _build_histogram(values: list[float], stats: dict) -> str:
    """Build an SVG histogram for latency values."""
    if not values:
        return ""
    max_val = max(values)
    if max_val == 0:
        return ""

    bin_size = 100

    bin_start = int(min(values) // bin_size) * bin_size
    bin_end = int(max_val // bin_size + 1) * bin_size + bin_size
    bins: list[tuple[int, int]] = []
    counts: list[int] = []
    for lo in range(bin_start, bin_end, bin_size):
        hi = lo + bin_size
        c = sum(1 for v in values if lo <= v < hi)
        bins.append((lo, hi))
        counts.append(c)

    # Trim trailing empty bins
    while counts and counts[-1] == 0:
        counts.pop()
        bins.pop()
    if not counts:
        return ""

    max_count = max(counts)
    n_bins = len(bins)
    median = stats.get("median_ms", 0)
    p95 = stats.get("p95_ms", 0)

    margin_l, margin_r, margin_t, margin_b = 10, 10, 16, 22
    w = 700
    chart_h = 60
    chart_w = w - margin_l - margin_r
    h = margin_t + chart_h + margin_b
    gap = max(1, min(2, chart_w / n_bins * 0.06))
    bar_w = (chart_w - gap * (n_bins - 1)) / n_bins

    def x_pos(bin_idx: int) -> float:
        return margin_l + bin_idx * (bar_w + gap)

    def x_val(v: float) -> float:
        frac = (v - bins[0][0]) / (bins[-1][1] - bins[0][0])
        return margin_l + frac * chart_w

    svg = [f'<svg width="100%" viewBox="0 0 {w} {h}" style="max-width:{w}px;display:block">']

    # Bars
    for i, (_lo, _hi) in enumerate(bins):
        c = counts[i]
        if c == 0:
            continue
        bar_h = c / max_count * chart_h if max_count else 0
        bx = x_pos(i)
        by = margin_t + chart_h - bar_h
        svg.append(f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bar_h}" fill="#0071e3" opacity="0.6" rx="2"/>')
        if c > 0:
            svg.append(
                f'<text x="{bx + bar_w / 2}" y="{by - 2}" font-size="9" fill="#1d1d1f" text-anchor="middle">{c}</text>'
            )

    # X-axis labels
    label_interval = max(1, n_bins // 8)
    for i, (lo, _hi) in enumerate(bins):
        if i % label_interval == 0 or i == n_bins - 1:
            tx = x_pos(i) + bar_w / 2
            svg.append(f'<text x="{tx}" y="{h - 2}" font-size="8" fill="#9ca3af" text-anchor="middle">{lo}</text>')

    # Median marker
    mx = x_val(median)
    svg.append(
        f'<line x1="{mx}" y1="{margin_t - 2}" x2="{mx}" y2="{margin_t + chart_h}" stroke="#0071e3" stroke-width="1.5" stroke-dasharray="3,2"/>'
    )

    # P95 marker
    if p95 > median:
        px = x_val(p95)
        svg.append(
            f'<line x1="{px}" y1="{margin_t - 2}" x2="{px}" y2="{margin_t + chart_h}" stroke="#d97706" stroke-width="1.5" stroke-dasharray="2,2"/>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def _build_turn_taking(scored: dict) -> str:
    scores = scored.get("scores", {})
    latency = scores.get("latency", {})
    turns = scored.get("turns", [])

    tt_turns = [
        t
        for t in turns
        if t.get("latency", {}).get("turn_shift_to_playback_ms") or t.get("turn_detection_delay_ms") is not None
    ]
    failed_turns = [
        t
        for t in turns
        if t.get("error")
        in (
            "no_response",
            "no_recognition",
            "no_turn_shift",
            "early_turn_shift",
            "late_turn_shift",
            "premature_turn_shift",
            "generation_failed",
        )
    ]
    tt_ids = {id(t) for t in tt_turns}
    failed_only = [t for t in failed_turns if id(t) not in tt_ids]
    all_tt_turns = tt_turns + failed_only

    if not all_tt_turns and not latency:
        return '<div class="empty">턴테이킹 데이터 없음 (text mode에서는 수집되지 않음)</div>'

    suite_rank, q_rank = _order_keys(scored)

    from itertools import groupby

    p = []

    # --- Sub-tabs ---
    sub_tab_style = (
        "padding:8px 16px;font-size:12px;font-weight:500;color:#86868b;"
        "cursor:pointer;border-bottom:2px solid transparent"
    )
    p.append(
        '<div class="tabs" style="position:static;background:transparent;'
        'border-bottom:1px solid #d2d2d7;padding:0;margin-bottom:16px">'
    )
    p.append(
        f'<div class="sub-tab active" data-subtarget="tt-latency" style="{sub_tab_style}">레이턴시 — 턴 감지·응답 속도</div>'
    )
    p.append(
        f'<div class="sub-tab" data-subtarget="tt-gate" style="{sub_tab_style}">Prepare 게이트 — 유사도 skip 판정</div>'
    )
    p.append("</div>")

    # ============================================================
    # Sub-tab: Latency
    # ============================================================
    p.append('<div id="tt-latency" class="sub-pane active">')

    # --- Description ---
    p.append(
        '<div class="panel mute" style="font-size:12px">'
        "발화 종료 후 첫 응답까지의 반응 속도를 측정. "
        "턴 감지(침묵 시작 → 턴 종료 판단)와 응답 생성(턴 감지 → 재생 시작)으로 구분."
        "</div>"
    )

    # --- Success/Fail cards ---
    total_tt = len(all_tt_turns)
    if total_tt:
        fail_count = len(failed_turns)
        p.append('<div class="cards">')
        p.append(
            f'<div class="card"><div class="card-label">턴 감지</div>'
            f'<div class="card-value">{total_tt - fail_count}<span class="unit">/{total_tt}</span></div>'
            f'<div class="card-sub">성공</div></div>'
        )
        if fail_count:
            early = sum(1 for t in failed_turns if t.get("error") == "early_turn_shift")
            late = sum(1 for t in failed_turns if t.get("error") == "late_turn_shift")
            premature = sum(1 for t in failed_turns if t.get("error") == "premature_turn_shift")
            no_recog = sum(1 for t in failed_turns if t.get("error") == "no_recognition")
            no_ts = sum(1 for t in failed_turns if t.get("error") == "no_turn_shift")
            parts = []
            if early:
                parts.append(f"조기 전환 {early}")
            if late:
                parts.append(f"지연 전환 {late}")
            if premature:
                parts.append(f"섣부른 전환 {premature}")
            if no_recog:
                parts.append(f"인식 실패 {no_recog}")
            if no_ts:
                parts.append(f"턴 감지 실패 {no_ts}")
            gen_fail = sum(1 for t in failed_turns if t.get("error") == "generation_failed")
            if gen_fail:
                parts.append(f"생성 실패 {gen_fail}")
            p.append(
                f'<div class="card"><div class="card-label">실패</div>'
                f'<div class="card-value bad">{fail_count}</div>'
                f'<div class="card-sub">{", ".join(parts)}</div></div>'
            )

        ts_pb = latency.get("turn_shift_to_playback_ms", {})
        if ts_pb:
            med = ts_pb.get("median_ms", 0)
            cls = _score_cls_lower_better(med, 1500, 3000)
            p.append(
                f'<div class="card"><div class="card-label">응답 속도</div>'
                f'<div class="card-value {cls}">{med:.0f}<span class="unit">ms</span></div>'
                f'<div class="card-sub">턴 감지 → 재생 (중위값)</div></div>'
            )

        cancel_total = sum(t.get("cancelled_turn_shifts", 0) for t in all_tt_turns)
        if cancel_total:
            cancel_turns = sum(1 for t in all_tt_turns if t.get("cancelled_turn_shifts"))
            p.append(
                f'<div class="card"><div class="card-label">잠정 전환 취소</div>'
                f'<div class="card-value warn">{cancel_total}<span class="unit">회</span></div>'
                f'<div class="card-sub">{cancel_turns}개 턴 — 조기 turn shift 신호</div></div>'
            )

        p.append("</div>")

    # --- Suite summary + Latency side by side ---
    suite_sorted = sorted(all_tt_turns, key=lambda t: suite_rank.get(t.get("suite_name", ""), 10**6))
    suite_groups = []
    for suite_name, group in groupby(suite_sorted, key=lambda t: t.get("suite_name", "")):
        suite_turns = list(group)
        success = sum(1 for t in suite_turns if not t.get("error"))
        total_s = len(suite_turns)
        lats = [
            t["latency"]["turn_shift_to_playback_ms"]
            for t in suite_turns
            if t.get("latency", {}).get("turn_shift_to_playback_ms")
        ]
        med = sorted(lats)[len(lats) // 2] if lats else None
        cat = _turn_category(suite_name)
        suite_groups.append((suite_name, success, total_s, med, cat))

    tt_suites = [(s, ok, tot, med) for s, ok, tot, med, cat in suite_groups if cat == "tt"]
    other_suites = [(s, ok, tot, med, cat) for s, ok, tot, med, cat in suite_groups if cat != "tt"]

    p.append('<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">')

    # Left: suite summary
    p.append('<div class="panel">')
    p.append('<div class="section-title">Suite별 요약</div>')

    # Header
    p.append(
        '<div style="display:flex;align-items:center;gap:16px;padding:3px 0;font-size:11px;color:#86868b;border-bottom:1px solid #e5e7eb;margin-bottom:4px">'
        '<span style="min-width:180px">Suite</span>'
        '<span style="min-width:120px">설명</span>'
        '<span style="min-width:60px">성공</span>'
        "<span>응답 중위값</span>"
        "</div>"
    )

    if tt_suites:
        for suite_name, success, total_s, med in tt_suites:
            desc = _get_suite_desc(scored, suite_name)
            all_ok = success == total_s
            tag_cls = "tag-ok" if all_ok else "tag-warn"
            med_str = f'<span class="mono">{med:.0f}ms</span>' if med else '<span class="mute">—</span>'
            p.append(
                f'<div style="display:flex;align-items:center;gap:16px;padding:3px 0;font-size:14px">'
                f'<span style="font-weight:500;min-width:180px">{suite_name}</span>'
                f'<span class="mute" style="min-width:120px">{_esc(desc)}</span>'
                f'<span style="min-width:60px"><span class="tag {tag_cls}">{success}/{total_s}</span></span>'
                f"{med_str}"
                f"</div>"
            )
    if other_suites:
        # Group by category
        from itertools import groupby as _gb

        other_sorted = sorted(other_suites, key=lambda x: suite_rank.get(x[0], 10**6))
        p.append(
            '<div style="margin-top:10px;padding-top:10px;border-top:1px solid #e5e7eb;font-size:14px;display:flex;gap:20px;flex-wrap:wrap">'
        )
        for cat, grp in _gb(other_sorted, key=lambda x: x[4]):
            items = list(grp)
            cat_label = _CATEGORY_LABELS.get(cat, cat)
            total_success = sum(s for _, s, _, _, _ in items)
            total_all = sum(t for _, _, t, _, _ in items)
            all_meds = [m for _, _, _, m, _ in items if m is not None]
            avg_med = sum(all_meds) / len(all_meds) if all_meds else None
            all_ok = total_success == total_all
            tag_cls = "tag-ok" if all_ok else "tag-warn"
            med_str = f' <span class="mono mute">{avg_med:.0f}ms</span>' if avg_med else ""
            p.append(
                f'<span>{cat_label} <span class="tag {tag_cls}">{total_success}/{total_all}</span>{med_str}</span>'
            )
        p.append("</div>")
    p.append("</div>")

    # Right: latency distribution
    plot_keys = ["turn_detection_delay_ms", "turn_shift_to_playback_ms", "llm_ttft_ms", "tts_ttfc_ms"]
    stats_only_keys = ["bridge_ms"]

    if latency:
        p.append('<div class="panel">')
        p.append('<div class="section-title">레이턴시 분포</div>')

        for key in plot_keys:
            stats = latency.get(key, {})
            if not stats:
                continue
            label = _LATENCY_LABELS.get(key.removesuffix("_ms"), key)

            values = []
            for t in tt_turns:
                if key == "turn_detection_delay_ms":
                    # expect_wait 스위트의 대기는 감지 속도가 아니라 timeout 설정값
                    if t.get("expect_wait"):
                        continue
                    v = t.get("turn_detection_delay_ms")
                else:
                    v = t.get("latency", {}).get(key)
                if v and v > 0:
                    values.append(v)

            p.append('<div style="margin-bottom:16px;padding-bottom:16px;border-bottom:1px solid #e5e7eb">')
            p.append(
                f'<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px">'
                f'<span style="font-weight:600;font-size:13px;min-width:100px">{label}</span>'
                f'<span class="mono" style="font-size:14px">{stats["median_ms"]:.0f}ms</span>'
                f'<span class="mute" style="font-size:12px">P95 {stats["p95_ms"]:.0f}ms · '
                f"{stats['min_ms']:.0f}–{stats['max_ms']:.0f}ms</span>"
                f"</div>"
            )
            p.append(_build_histogram(values, stats))
            p.append("</div>")

        for key in stats_only_keys:
            stats = latency.get(key, {})
            if not stats:
                continue
            label = _LATENCY_LABELS.get(key.removesuffix("_ms"), key)
            p.append(
                f'<div style="display:flex;align-items:baseline;gap:12px;padding:3px 0">'
                f'<span style="font-weight:600;font-size:13px;min-width:100px">{label}</span>'
                f'<span class="mono" style="font-size:14px">{stats["median_ms"]:.0f}ms</span>'
                f'<span class="mute" style="font-size:12px">P95 {stats["p95_ms"]:.0f}ms · '
                f"{stats['min_ms']:.0f}–{stats['max_ms']:.0f}ms</span>"
                f"</div>"
            )

        p.append("</div>")
    else:
        p.append('<div class="panel"><div class="empty">레이턴시 데이터 없음</div></div>')

    p.append("</div>")

    # --- Per-question results ---
    if all_tt_turns:
        sorted_tt = sorted(all_tt_turns, key=lambda t: q_rank.get(t["question_id"], 10**6))

        p.append('<div class="section">')
        p.append('<div class="section-title">개별 결과</div>')

        current_cat = None
        current_suite = None
        for t in sorted_tt:
            cat = _turn_category(t.get("suite_name", ""))
            suite = t.get("suite_name", "")

            if cat != current_cat:
                if current_cat is not None:
                    p.append("</div>")
                current_cat = cat
                current_suite = None
                cat_label = _CATEGORY_LABELS.get(cat, cat)
                p.append(f'<div><div class="cat-header">{cat_label}</div>')

            if suite != current_suite:
                current_suite = suite
                desc = _get_suite_desc(scored, suite)
                desc_str = f' <span class="mute">— {_esc(desc)}</span>' if desc else ""
                p.append(f'<div class="suite-header">{suite}{desc_str}</div>')

            latency_data = t.get("latency", {})
            lat = latency_data.get("turn_shift_to_playback_ms")
            td_delay = t.get("turn_detection_delay_ms")
            llm_ttft = latency_data.get("llm_ttft_ms")
            tts_ttfc = latency_data.get("tts_ttfc_ms")
            error = t.get("error")
            sys_text = t.get("system_text") or ""
            asr_text = t.get("asr_text") or ""
            has_asr_diff = asr_text and sys_text and asr_text != sys_text

            # Color class based on response latency
            if error or lat and lat > 3000:
                lat_cls = "bad"
            elif lat and lat > 2000:
                lat_cls = "warn"
            elif lat:
                lat_cls = "good"
            else:
                lat_cls = ""

            # Per-metric color (muted tones for non-primary)
            def _lat_color_muted(val, good, bad):
                if val <= good:
                    return "#6b9e7a"
                if val >= bad:
                    return "#c47a7a"
                return "#b8976b"

            # Latency chips with individual colors
            lat_chips = []
            if td_delay is not None:
                c = _lat_color_muted(td_delay, 500, 1500)
                lat_chips.append(f'<span style="color:{c}">감지 {td_delay:.0f}ms</span>')
            if lat:
                lat_chips.append(f'<span class="{lat_cls}">응답 {lat:.0f}ms</span>')
            if llm_ttft:
                c = _lat_color_muted(llm_ttft, 1000, 2500)
                lat_chips.append(f'<span style="color:{c}">LLM {llm_ttft:.0f}ms</span>')
            if tts_ttfc:
                c = _lat_color_muted(tts_ttfc, 1200, 2000)
                lat_chips.append(f'<span style="color:{c}">TTS {tts_ttfc:.0f}ms</span>')
            lat_str = (
                '<span class="mono" style="font-size:11px">' + " · ".join(lat_chips) + "</span>" if lat_chips else ""
            )

            p.append('<div class="item" style="display:grid;grid-template-columns:1fr 1fr;gap:4px 24px">')

            # Left column
            p.append("<div>")
            # Line 1: original text (bold) + ID & latency (small, inline)
            p.append(
                f'<div style="font-size:14px"><span style="font-weight:600">{_esc(t.get("input_text", ""))}</span></div>'
            )
            p.append(f'<div style="margin-top:2px"><span class="mute" style="font-size:11px">{t["question_id"]}</span>')
            if error:
                p.append(f' <span class="mute" style="font-size:11px">|</span> {_error_tag(error)}')
            ts_reason = t.get("turn_shift_reason")
            if ts_reason:
                p.append(f' <span class="tag tag-mute">{_esc(ts_reason)}</span>')
            cancels = t.get("cancelled_turn_shifts")
            if cancels:
                p.append(f' <span class="tag tag-warn">전환 취소 {cancels}회</span>')
            call_issues = t.get("call_issues")
            if call_issues:
                p.append(f' <span class="mute" style="font-size:11px">|</span> {_call_issues_tag(call_issues)}')
            if lat_str:
                p.append(f' <span class="mute" style="font-size:11px">|</span> {lat_str}')
            p.append("</div></div>")

            # Right column: asr text + system diff
            p.append("<div>")
            p.append(
                f'<div class="item-row"><span class="item-label mute">인식</span><span class="item-text">{_esc(asr_text)}</span></div>'
            )
            if has_asr_diff:
                p.append(
                    f'<div class="item-row"><span class="item-label mute">시스템</span><span class="item-text diff-mark">{_esc(sys_text)}</span></div>'
                )
            p.append("</div>")

            p.append("</div>")
            p.append(_detail_block(t))

        if current_cat is not None:
            p.append("</div>")
        p.append("</div>")

    p.append("</div>")  # close tt-latency

    # ============================================================
    # Sub-tab: Prepare similarity gate
    # ============================================================
    p.append('<div id="tt-gate" class="sub-pane" style="display:none">')
    p.append(_build_prepare_gate(scored))
    p.append("</div>")

    return "\n".join(p)


_GATE_DECISION_META = {
    "skip": ("#6b9e7a", "skip — 유사 판정, 재생성 생략"),
    "keep": ("#8aa9c9", "keep — PENDING 중 유사, 취소 안 함"),
    "regenerate": ("#b8976b", "regenerate — 의미 변화, 재생성"),
    "cancel": ("#c47a7a", "cancel — PENDING 중 의미 변화, 턴 전환 취소"),
}


def _build_prepare_gate(scored: dict) -> str:
    """Prepare similarity gate sub-tab: judge verdicts + similarity distributions."""
    scores = scored.get("scores", {})
    gate = scores.get("prepare_gate", {})
    turns = scored.get("turns", [])
    _, q_rank = _order_keys(scored)
    judged_turns = sorted((t for t in turns if "gate_judge" in t), key=lambda t: q_rank.get(t["question_id"], 10**6))
    event_turns = [t for t in turns if t.get("similarity_events")]

    if not gate and not judged_turns and not event_turns:
        return '<div class="empty">유사도 게이트 데이터 없음 (similarity_gate call record 필요)</div>'

    threshold = None
    for t in event_turns:
        for e in t["similarity_events"]:
            if e.get("threshold"):
                threshold = e["threshold"]
                break
        if threshold:
            break

    p = []

    # --- Description ---
    p.append(
        '<div class="panel mute" style="font-size:12px">'
        "ASR 업데이트 시 직전 prepare 텍스트와의 임베딩 유사도가 threshold"
        + (f"({threshold})" if threshold else "")
        + " 이상이면 응답 재생성을 생략(skip)한다. "
        "최종 인식 텍스트와 시스템 텍스트(LLM 입력)가 달라진 턴을 LLM Judge가 판정하여, "
        "의미가 달라졌는데 skip되어 응답이 부적절해진 경우를 harmful skip으로 집계."
        "</div>"
    )

    # --- Summary cards ---
    if gate:
        p.append('<div class="cards">')
        diff = gate.get("diff_turn_count", 0)
        total = gate.get("voice_turn_count", 0)
        p.append(
            f'<div class="card"><div class="card-label">텍스트 불일치</div>'
            f'<div class="card-value">{diff}<span class="unit">/{total}</span></div>'
            f'<div class="card-sub">인식 ≠ 시스템 (음성 턴)</div></div>'
        )
        judged = gate.get("judged_count", 0)
        if judged:
            harmful = gate.get("harmful_count", 0)
            cls = "bad" if harmful else "good"
            p.append(
                f'<div class="card"><div class="card-label">Harmful skip</div>'
                f'<div class="card-value {cls}">{harmful}<span class="unit">/{judged}</span></div>'
                f'<div class="card-sub">의미 변화 + 응답 부적절</div></div>'
            )
            mc = gate.get("meaning_changed_count", 0)
            p.append(
                f'<div class="card"><div class="card-label">의미 변화</div>'
                f'<div class="card-value">{mc}<span class="unit">/{judged}</span></div>'
                f'<div class="card-sub">Judge 판정</div></div>'
            )
        sa = gate.get("speculative_attempts") or {}
        if sa:
            p.append(
                f'<div class="card"><div class="card-label">재생성 횟수</div>'
                f'<div class="card-value">{sa["mean"]}</div>'
                f'<div class="card-sub">턴당 평균 prepare (최대 {sa["max"]})</div></div>'
            )
        p.append("</div>")

    # --- Similarity strip plot per decision ---
    sims_by_decision = gate.get("similarities_by_decision", {})
    if sims_by_decision:
        harmful_set = {round(s, 4) for s in gate.get("harmful_similarities", [])}
        all_sims = [s for v in sims_by_decision.values() for s in v]
        thresh = threshold or 0.85
        lo = max(0.0, min(min(all_sims), thresh) - 0.05)
        hi = 1.0

        def _pct(s: float) -> float:
            return (s - lo) / (hi - lo) * 100

        p.append('<div class="panel">')
        p.append('<div class="section-title">유사도 분포 (게이트 판정별)</div>')
        for decision in ("skip", "keep", "regenerate", "cancel"):
            sims = sims_by_decision.get(decision)
            if not sims:
                continue
            color, label = _GATE_DECISION_META[decision]
            p.append('<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">')
            p.append(
                f'<span style="font-size:12px;min-width:260px;color:{color};font-weight:500">{label} <span class="mute">({len(sims)})</span></span>'
            )
            p.append('<div style="position:relative;flex:1;height:18px;background:#f9fafb;border-radius:4px">')
            p.append(
                f'<div style="position:absolute;left:{_pct(thresh):.1f}%;top:0;bottom:0;'
                f'width:1px;background:#c47a7a"></div>'
            )
            for s in sims:
                is_harmful = decision in ("skip", "keep") and round(s, 4) in harmful_set
                border = "border:2px solid #991b1b;" if is_harmful else ""
                p.append(
                    f'<div title="{s:.3f}" style="position:absolute;left:{_pct(s):.1f}%;top:50%;'
                    f"transform:translate(-50%,-50%);width:8px;height:8px;border-radius:50%;"
                    f'background:{color};opacity:0.8;{border}"></div>'
                )
            p.append("</div></div>")
        p.append(
            f'<div style="position:relative;margin-left:272px;height:14px;font-size:10px" class="mute mono">'
            f'<span style="position:absolute;left:0">{lo:.2f}</span>'
            f'<span style="position:absolute;left:{_pct(thresh):.1f}%;transform:translateX(-50%);'
            f'color:#c47a7a">threshold {thresh}</span>'
            f'<span style="position:absolute;right:0">1.00</span>'
            f"</div>"
        )
        p.append("</div>")

    # --- Individual judged turns ---
    if judged_turns:
        p.append('<div class="section">')
        p.append('<div class="section-title">개별 판정 (인식 ≠ 시스템 텍스트)</div>')
        for t in judged_turns:
            gj = t["gate_judge"]
            if gj.get("harmful"):
                tag = '<span class="tag tag-fail">harmful skip</span>'
            elif gj.get("meaning_changed"):
                tag = '<span class="tag tag-warn">의미 변화 (응답은 적절)</span>'
            else:
                tag = '<span class="tag tag-ok">skip 적절</span>'

            sim_chips = []
            for e in t.get("similarity_events", []):
                s = e.get("similarity")
                d = e.get("decision", "?")
                if s is not None:
                    color = _GATE_DECISION_META.get(d, ("#6b7280", ""))[0]
                    sim_chips.append(f'<span class="mono" style="color:{color}">{d} {s:.3f}</span>')
            chips_str = " · ".join(sim_chips)

            p.append('<div class="item" style="margin-bottom:8px">')
            p.append(
                f'<div class="item-header"><span class="item-id">{t["question_id"]}</span> '
                f'<span class="mute" style="font-size:11px">{t.get("suite_name", "")}</span> {tag}'
                + (f" {_error_tag(t['error'])}" if t.get("error") else "")
                + (
                    f' <span class="mute" style="font-size:11px">|</span> <span style="font-size:11px">{chips_str}</span>'
                    if chips_str
                    else ""
                )
                + "</div>"
            )
            p.append(
                f'<div class="item-row"><span class="item-label mute">시스템</span><span class="item-text diff-mark">{_esc(t.get("system_text") or "")}</span></div>'
            )
            p.append(
                f'<div class="item-row"><span class="item-label mute">인식</span><span class="item-text">{_esc(t.get("asr_text") or "")}</span></div>'
            )
            p.append(
                f'<div class="item-row"><span class="item-label mute">응답</span><span class="item-text mute">{_esc((t.get("response_text") or "")[:200])}</span></div>'
            )
            if gj.get("reasoning"):
                p.append(f'<div class="reasoning">{_esc(gj["reasoning"])}</div>')
            p.append("</div>")
        p.append("</div>")
    elif gate.get("diff_turn_count"):
        p.append('<div class="empty">Judge 판정 없음 (judge 호출 실패 또는 미실행)</div>')

    return "\n".join(p)


def _get_question_text(scored: dict, qid: str) -> str:
    return scored.get("config", {}).get("question_texts", {}).get(qid, "")


def _build_interruption(scored: dict) -> str:
    scores = scored.get("scores", {})
    interruption = scores.get("interruption", {})
    turns = scored.get("turns", [])
    int_turns = [t for t in turns if "interrupt_delay_sec" in t]

    if not int_turns and not interruption:
        return '<div class="empty">인터럽션 데이터 없음</div>'

    _, q_rank = _order_keys(scored)

    p = []

    # --- Description ---
    p.append(
        '<div class="panel mute" style="font-size:12px">'
        "응답 재생 중 사용자가 끼어들었을 때 감지 여부를 측정. "
        "delay별로 재생 시작 후 끼어들기까지의 시간을 달리하여 테스트."
        "</div>"
    )

    # --- Summary cards ---
    if interruption and interruption.get("testable"):
        dr = interruption.get("detection_rate", 0)
        cls = _score_cls(dr, 0.7, 0.5)
        p.append('<div class="cards">')
        p.append(
            f'<div class="card"><div class="card-label">감지율</div>'
            f'<div class="card-value {cls}">{dr:.0%}</div>'
            f'<div class="card-sub">{interruption.get("detected", 0)}/{interruption.get("testable", 0)} 감지</div></div>'
        )
        int_lat = interruption.get("latency", {})
        if int_lat:
            p.append(
                f'<div class="card"><div class="card-label">감지 지연</div>'
                f'<div class="card-value">{int_lat.get("median_ms", 0):.0f}<span class="unit">ms</span></div>'
                f'<div class="card-sub">중위값</div></div>'
            )

        p.append("</div>")

    # --- Delay summary + Interrupt audio summary side by side ---
    by_delay = interruption.get("by_delay", {})
    if by_delay or int_turns:
        p.append('<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">')

        # Left: delay summary
        p.append('<div class="panel">')
        p.append('<div class="section-title">Delay별 요약</div>')
        if by_delay:
            p.append(
                '<div style="display:flex;align-items:center;gap:16px;padding:3px 0;font-size:11px;color:#86868b;border-bottom:1px solid #e5e7eb;margin-bottom:4px">'
                '<span style="min-width:50px">Delay</span>'
                '<span style="min-width:80px">감지율</span>'
                "<span>결과 분포</span>"
                "</div>"
            )
            for delay_str, b in by_delay.items():
                testable = b.get("testable", 0)
                truncated = b.get("truncated", 0)
                completed = b.get("completed", 0)
                na = b.get("na", 0)

                if testable == 0:
                    rate_str = '<span class="mute">N/A</span>'
                else:
                    rate = b.get("detected", 0) / testable
                    cls = _score_cls(rate, 0.7, 0.5)
                    rate_str = f'<span class="{cls}">{rate:.0%}</span> ({b.get("detected", 0)}/{testable})'

                dist_parts = []
                if truncated:
                    dist_parts.append(f'<span class="tag tag-ok">중단 {truncated}</span>')
                if completed:
                    dist_parts.append(f'<span class="tag tag-fail">미감지 {completed}</span>')
                if na:
                    dist_parts.append(f'<span class="tag tag-mute">N/A {na}</span>')

                p.append(
                    f'<div style="display:flex;align-items:center;gap:16px;padding:3px 0;font-size:14px">'
                    f'<span style="font-weight:500;min-width:50px">{delay_str}초</span>'
                    f'<span style="min-width:80px">{rate_str}</span>'
                    f"<span>{' '.join(dist_parts)}</span>"
                    f"</div>"
                )
        p.append("</div>")

        # Right: interrupt audio summary
        p.append('<div class="panel">')
        p.append('<div class="section-title">인터럽트 메시지별 요약</div>')
        if int_turns:
            from itertools import groupby as _gb2

            p.append(
                '<div style="display:flex;align-items:center;gap:16px;padding:3px 0;font-size:11px;color:#86868b;border-bottom:1px solid #e5e7eb;margin-bottom:4px">'
                '<span style="min-width:160px">메시지</span>'
                '<span style="min-width:80px">감지율</span>'
                "<span>결과 분포</span>"
                "</div>"
            )
            audio_sorted = sorted(int_turns, key=lambda t: q_rank.get(t.get("interrupt_audio", ""), 10**6))
            for audio_id, grp in _gb2(audio_sorted, key=lambda t: t.get("interrupt_audio", "")):
                audio_turns = list(grp)
                audio_text = _get_question_text(scored, audio_id)
                played_turns = [t for t in audio_turns if t.get("interrupt_played")]
                testable = len(played_turns)
                detected = sum(1 for t in played_turns if t.get("outcome") in ("truncated", "cancelled"))
                truncated = sum(1 for t in played_turns if t.get("outcome") == "truncated")
                completed = sum(1 for t in played_turns if t.get("outcome") == "completed")
                na_count = len(audio_turns) - testable

                if testable == 0:
                    rate_str = '<span class="mute">N/A</span>'
                else:
                    rate = detected / testable
                    cls = _score_cls(rate, 0.7, 0.5)
                    rate_str = f'<span class="{cls}">{rate:.0%}</span> ({detected}/{testable})'

                dist_parts = []
                if truncated:
                    dist_parts.append(f'<span class="tag tag-ok">중단 {truncated}</span>')
                if completed:
                    dist_parts.append(f'<span class="tag tag-fail">미감지 {completed}</span>')
                if na_count:
                    dist_parts.append(f'<span class="tag tag-mute">N/A {na_count}</span>')

                label = _esc(audio_text) if audio_text else audio_id
                p.append(
                    f'<div style="display:flex;align-items:center;gap:16px;padding:3px 0;font-size:14px">'
                    f'<span style="font-weight:500;min-width:160px">{label}</span>'
                    f'<span style="min-width:80px">{rate_str}</span>'
                    f"<span>{' '.join(dist_parts)}</span>"
                    f"</div>"
                )
        p.append("</div>")

        p.append("</div>")

    # --- Per-question: grouped by delay > interrupt audio (questions.json order within) ---
    if int_turns:
        sorted_int = sorted(
            int_turns,
            key=lambda t: (
                t.get("interrupt_delay_sec", 0),
                q_rank.get(t.get("interrupt_audio", ""), 10**6),
                q_rank.get(t["question_id"], 10**6),
            ),
        )

        p.append('<div class="section">')
        p.append('<div class="section-title">개별 결과</div>')

        current_delay = None
        current_audio = None
        for t in sorted_int:
            delay = t.get("interrupt_delay_sec", 0)
            audio_id = t.get("interrupt_audio", "")
            outcome = t.get("outcome")
            played = t.get("interrupt_played", False)
            q_text = _esc(t.get("input_text", ""))

            # Delay group header
            if delay != current_delay:
                current_delay = delay
                current_audio = None
                p.append(f'<div class="cat-header" style="font-size:14px;margin:16px 0 6px">Delay {delay:.1f}초</div>')

            # Interrupt audio sub-header
            if audio_id != current_audio:
                current_audio = audio_id
                audio_text = _get_question_text(scored, audio_id)
                p.append(f'<div class="suite-header">{audio_id} <span class="mute">— {_esc(audio_text)}</span></div>')

            # Latency info
            latency_data = t.get("latency", {})
            int_lat_ms = latency_data.get("interrupt_latency_ms")
            response_text = t.get("response_text", "")

            lat_chips = []
            if int_lat_ms and int_lat_ms > 0:
                lat_chips.append(f"감지 지연 {int_lat_ms:.0f}ms")
            lat_str = " · ".join(lat_chips)

            p.append('<div class="item" style="display:grid;grid-template-columns:1fr 1fr;gap:4px 24px">')

            # Left: question + metadata
            p.append("<div>")
            p.append('<div style="display:flex;align-items:center;gap:12px">')
            p.append(f'<span style="font-weight:600;font-size:14px">{q_text}</span>')
            p.append(f"{_outcome_tag(outcome)}")
            if not played:
                p.append('<span class="tag tag-mute">미재생</span>')
            p.append("</div>")
            p.append('<div style="margin-top:2px">')
            p.append(f'<span class="mute" style="font-size:11px">{t["question_id"]}</span>')
            if lat_str:
                p.append(
                    f' <span class="mute" style="font-size:11px">|</span> <span class="mono mute" style="font-size:11px">{lat_str}</span>'
                )
            p.append("</div>")
            p.append("</div>")

            # Right: truncated response
            p.append("<div>")
            if outcome == "truncated" and response_text:
                p.append(
                    f'<div class="item-row"><span class="item-label mute">응답</span><span class="item-text">{_esc(response_text)}</span></div>'
                )
            p.append("</div>")

            p.append("</div>")
            p.append(_detail_block(t))

        p.append("</div>")

    return "\n".join(p)


def _build_quality(scored: dict) -> str:
    scores = scored.get("scores", {})
    quality = scores.get("quality", {})
    turns = scored.get("turns", [])
    quality_turns = [t for t in turns if "quality_scores" in t]

    if not quality_turns:
        return '<div class="empty">응답 품질 데이터 없음</div>'

    suite_rank, q_rank = _order_keys(scored)

    from itertools import groupby

    p = []

    by_suite = quality.get("by_suite", {})

    # --- Description ---
    p.append(
        '<div class="panel mute" style="font-size:12px">'
        "LLM Judge가 각 응답을 공통 기준(관련성, 음성 적합성, 자연스러움)과 "
        "suite 고유 기준으로 1~5점 채점."
        "</div>"
    )

    # --- Summary card ---
    mean = quality.get("mean_score", 0)
    cls = _score_cls(mean, 4.0, 3.0)
    p.append('<div class="cards">')
    p.append(
        f'<div class="card"><div class="card-label">종합 평균</div>'
        f'<div class="card-value {cls}">{mean:.1f}<span class="unit">/5</span></div></div>'
    )
    p.append("</div>")

    # --- Suite summary + Criteria explanation side by side ---
    _SUITE_CRITERION = {
        "lq_factual": "correctness",
        "lq_advice": "helpfulness",
        "lq_casual": "engagement",
        "lq_empathy": "empathy",
        "lq_voice_adaptation": "format_adaptation",
        "lq_multi_turn": "context_coherence",
        "lq_wrong_premise": "correction_quality",
        "lq_impossible": "boundary_communication",
    }

    p.append('<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">')

    # Left: suite summary
    if by_suite:
        p.append('<div class="panel">')
        p.append('<div class="section-title">Suite별 요약</div>')
        p.append(
            '<div style="display:flex;align-items:center;gap:16px;padding:3px 0;font-size:11px;color:#86868b;border-bottom:1px solid #e5e7eb;margin-bottom:4px">'
            '<span style="min-width:180px">Suite</span>'
            '<span style="min-width:120px">설명</span>'
            '<span style="min-width:50px">점수</span>'
            "<span>고유 기준</span>"
            "</div>"
        )
        for suite, stats in sorted(by_suite.items(), key=lambda kv: suite_rank.get(kv[0], 10**6)):
            s_mean = stats.get("mean_score", 0)
            desc = _get_suite_desc(scored, suite)
            criterion = _SUITE_CRITERION.get(suite, "")
            criterion_label = _CRITERION_LABELS.get(criterion, criterion)
            p.append(
                f'<div style="display:flex;align-items:center;gap:16px;padding:3px 0;font-size:14px">'
                f'<span style="font-weight:500;min-width:180px">{suite}</span>'
                f'<span class="mute" style="min-width:120px">{_esc(desc)}</span>'
                f'<span style="min-width:50px">{_score_chip(s_mean)}</span>'
                f'<span class="mute" style="font-size:12px">{criterion_label}</span>'
                f"</div>"
            )
        p.append("</div>")

    # Right: criteria explanation
    p.append('<div class="panel">')
    p.append('<div class="section-title">평가 기준</div>')
    p.append('<div style="font-size:13px;line-height:1.8">')
    p.append(
        '<div><span style="font-weight:600">관련성</span> <span class="mute">— 질문의 핵심 의도에 맞는 응답인가</span></div>'
    )
    p.append(
        '<div><span style="font-weight:600">음성 적합성</span> <span class="mute">— 음성으로 듣기에 적절한 길이와 구조인가</span></div>'
    )
    p.append(
        '<div><span style="font-weight:600">자연스러움</span> <span class="mute">— 사람과 대화하는 듯 자연스러운 어투인가</span></div>'
    )
    p.append(
        '<div><span style="font-weight:600">고유 기준</span> <span class="mute">— suite별로 다른 핵심 평가 항목 (정확성, 공감, 맥락 유지 등)</span></div>'
    )
    p.append("</div>")
    p.append("</div>")

    p.append("</div>")

    # --- Per-question by suite (main view), questions.json order ---
    sorted_turns = sorted(quality_turns, key=lambda t: q_rank.get(t["question_id"], 10**6))
    for suite_name, group in groupby(sorted_turns, key=lambda t: t["suite_name"]):
        suite_turns = list(group)
        s_stats = by_suite.get(suite_name, {})
        s_mean = s_stats.get("mean_score", 0)
        desc = _get_suite_desc(scored, suite_name)
        desc_str = f' <span class="mute">— {_esc(desc)}</span>' if desc else ""
        is_multi = any(t.get("scenario_id") for t in suite_turns)

        p.append('<div class="suite-group">')
        p.append(
            f'<div class="suite-header">{suite_name}{desc_str} '
            f"{_score_chip(s_mean)} "
            f'<span class="mute">({len(suite_turns)}건)</span></div>'
        )

        if is_multi:
            scenario_sorted = sorted(suite_turns, key=lambda t: q_rank.get(t["question_id"], 10**6))
            for scenario_id, sc_group in groupby(scenario_sorted, key=lambda t: t.get("scenario_id", "")):
                sc_turns = list(sc_group)

                p.append('<div class="item" style="display:grid;grid-template-columns:1fr 1fr;gap:4px 24px">')

                # Left: interleaved question/response
                p.append("<div>")
                for t in sc_turns:
                    p.append(
                        f'<div class="item-row"><span class="item-label mute">질문</span><span class="item-text">{_esc(t.get("input_text", ""))}</span></div>'
                    )
                    if t.get("response_text"):
                        p.append(
                            f'<div class="item-row"><span class="item-label mute">응답</span><span class="item-text">{_esc(t["response_text"])}</span></div>'
                        )
                p.append("</div>")

                # Right: scores + reasoning
                all_scores: dict[str, list[float]] = {}
                all_reasoning = []
                for t in sc_turns:
                    for k, v in t.get("quality_scores", {}).items():
                        all_scores.setdefault(k, []).append(v)
                    if t.get("quality_reasoning"):
                        all_reasoning.append(t["quality_reasoning"])

                p.append("<div>")
                if all_scores:
                    avg_scores = {k: sum(vs) / len(vs) for k, vs in all_scores.items()}
                    p.append('<div class="item-scores">')
                    p.append(
                        f'<span class="mute" style="font-size:11px">{scenario_id}</span> <span class="mute" style="font-size:11px">|</span> '
                    )
                    for k, v in avg_scores.items():
                        label = _CRITERION_LABELS.get(k, k)
                        p.append(f'{_score_chip(v)} <span class="mute" style="font-size:10px">{label}</span>')
                    p.append("</div>")
                if all_reasoning:
                    p.append(f'<div class="reasoning">{_esc(all_reasoning[-1])}</div>')
                p.append("</div>")

                p.append("</div>")
                for t in sc_turns:
                    p.append(_detail_block(t, f"단계 상세 — {t['question_id']}"))
        else:
            for t in suite_turns:
                qs = t.get("quality_scores", {})
                reasoning = t.get("quality_reasoning", "")

                p.append('<div class="item" style="display:grid;grid-template-columns:1fr 1fr;gap:4px 24px">')

                # Left: question + response
                p.append("<div>")
                p.append(
                    f'<div class="item-row"><span class="item-label mute">질문</span><span class="item-text">{_esc(t.get("input_text", ""))}</span></div>'
                )
                p.append(
                    f'<div class="item-row"><span class="item-label mute">응답</span><span class="item-text">{_esc(t.get("response_text", ""))}</span></div>'
                )
                p.append("</div>")

                # Right: scores + reasoning
                p.append("<div>")
                p.append('<div class="item-scores">')
                p.append(
                    f'<span class="mute" style="font-size:11px">{t["question_id"]}</span> <span class="mute" style="font-size:11px">|</span> '
                )
                for k, v in qs.items():
                    label = _CRITERION_LABELS.get(k, k)
                    p.append(f'{_score_chip(v)} <span class="mute" style="font-size:10px">{label}</span>')
                p.append("</div>")
                if reasoning:
                    p.append(f'<div class="reasoning">{_esc(reasoning)}</div>')
                p.append("</div>")

                p.append("</div>")
                p.append(_detail_block(t))

        p.append("</div>")

    return "\n".join(p)


def _build_memory(scored: dict) -> str:
    scores = scored.get("scores", {})
    memory = scores.get("memory", {})
    turns = scored.get("turns", [])
    memory_turns = [t for t in turns if t.get("suite_name", "").startswith("mem_")]

    if not memory_turns and not memory:
        return '<div class="empty">장기기억 데이터 없음</div>'

    suite_rank, q_rank = _order_keys(scored)

    writer = memory.get("writer", {})
    recall = memory.get("retriever_recall", {})
    mem_quality = memory.get("quality", {})

    p = []

    # --- Sub-tabs ---
    p.append(
        '<div class="tabs" style="position:static;background:transparent;border-bottom:1px solid #d2d2d7;padding:0;margin-bottom:16px">'
    )
    p.append(
        '<div class="sub-tab active" data-subtarget="mem-writer" style="padding:8px 16px;font-size:12px;font-weight:500;color:#86868b;cursor:pointer;border-bottom:2px solid transparent">Writer — 기억 추출</div>'
    )
    p.append(
        '<div class="sub-tab" data-subtarget="mem-retriever" style="padding:8px 16px;font-size:12px;font-weight:500;color:#86868b;cursor:pointer;border-bottom:2px solid transparent">Retriever · Quality — 검색 및 활용</div>'
    )
    p.append("</div>")

    # ============================================================
    # Sub-tab: Writer
    # ============================================================
    p.append('<div id="mem-writer" class="sub-pane active">')

    # Writer description
    p.append(
        '<div class="panel mute" style="font-size:12px">'
        "대화 세션에서 중요한 정보를 에피소드로 추출하는 Writer의 품질을 평가. "
        "completeness(빠짐없이 추출), accuracy(사실 충실도), granularity(적절한 세부 수준)로 채점."
        "</div>"
    )

    # Writer cards
    if writer and writer.get("mean_score"):
        w = writer["mean_score"]
        cls = _score_cls(w, 4.0, 3.0)
        p.append('<div class="cards">')
        p.append(
            f'<div class="card"><div class="card-label">추출 품질</div>'
            f'<div class="card-value {cls}">{w:.1f}<span class="unit">/5</span></div>'
            f'<div class="card-sub">Writer 평균</div></div>'
        )
        p.append("</div>")

    # Writer sessions
    if writer and writer.get("by_session"):
        for s in writer["by_session"]:
            p.append('<div class="item" style="margin-bottom:8px">')

            p.append(
                f'<div class="item-header">'
                f'<span class="item-id">Session {s["session_index"]}</span>'
                f'<span class="mute" style="font-size:11px">|</span> '
            )
            for k in ["completeness", "accuracy", "granularity"]:
                v = s.get(k, 0)
                p.append(f'{_score_chip(v)} <span class="mute" style="font-size:10px">{k}</span> ')
            p.append(f'<span class="mute">{s.get("episode_count", 0)} episodes</span>')
            p.append("</div>")

            if s.get("reasoning"):
                p.append(f'<div class="reasoning">{_esc(s["reasoning"])}</div>')

            utts = s.get("utterances", [])
            eps = s.get("episodes", [])
            if utts or eps:
                p.append(
                    '<div class="collapsible-toggle mute" style="margin-top:6px;font-size:12px">Seed 대화 · 추출된 에피소드</div>'
                )
                p.append('<div class="collapsible-body" style="margin-top:4px">')
                p.append('<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">')

                if utts:
                    p.append('<div style="background:#f9fafb;border-radius:6px;padding:10px 12px">')
                    p.append(
                        '<div style="font-size:11px;font-weight:600;color:#86868b;margin-bottom:6px">Seed 대화</div>'
                    )
                    for u in utts:
                        if u["role"] == "user":
                            p.append(
                                f'<div style="font-size:12px;padding:4px 0;font-weight:500;display:flex;gap:6px"><span class="mute" style="min-width:52px;flex-shrink:0">사용자</span><span>{_esc(u["text"])}</span></div>'
                            )
                        else:
                            p.append(
                                f'<div style="font-size:12px;padding:4px 0 8px;color:#6b7280;border-bottom:1px solid #e5e7eb;margin-bottom:4px;display:flex;gap:6px"><span class="mute" style="min-width:52px;flex-shrink:0">봇</span><span>{_esc(u["text"])}</span></div>'
                            )
                    p.append("</div>")
                else:
                    p.append("<div></div>")

                if eps:
                    p.append('<div style="background:#f9fafb;border-radius:6px;padding:10px 12px">')
                    p.append(
                        '<div style="font-size:11px;font-weight:600;color:#86868b;margin-bottom:6px">추출된 에피소드</div>'
                    )
                    for ep in eps:
                        p.append(
                            f'<div style="font-size:12px;padding:5px 0 5px 10px;margin-bottom:4px;border-left:2px solid #86868b">{_esc(ep["text"])}</div>'
                        )
                    p.append("</div>")
                else:
                    p.append("<div></div>")

                p.append("</div>")
                p.append("</div>")

            p.append("</div>")

    p.append("</div>")

    # ============================================================
    # Sub-tab: Retriever & Quality
    # ============================================================
    p.append('<div id="mem-retriever" class="sub-pane" style="display:none">')

    # Summary cards
    scored_mem_turns = [t for t in memory_turns if t.get("memory_scores") or t.get("retriever_recall") is not None]

    p.append('<div class="cards">')
    if recall and recall.get("mean_recall") is not None:
        r = recall["mean_recall"]
        cls = _score_cls(r, 0.7, 0.4)
        p.append(
            f'<div class="card"><div class="card-label">Recall</div><div class="card-value {cls}">{r:.0%}</div></div>'
        )
    if mem_quality:
        if mem_quality.get("mean_precision") is not None:
            pr = mem_quality["mean_precision"]
            cls = _score_cls(pr, 0.5, 0.2)
            p.append(
                f'<div class="card"><div class="card-label">Precision</div>'
                f'<div class="card-value {cls}">{pr:.0%}</div></div>'
            )
        if mem_quality.get("mean_score"):
            mq = mem_quality["mean_score"]
            cls = _score_cls(mq, 4.0, 3.0)
            p.append(
                f'<div class="card"><div class="card-label">활용 품질</div>'
                f'<div class="card-value {cls}">{mq:.1f}<span class="unit">/5</span></div></div>'
            )
    p.append("</div>")

    # Suite summary + criteria explanation side by side
    from itertools import groupby

    p.append('<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">')

    _MEM_SUITE_DETAIL = {
        "mem_recall": "단일 사실을 정확히 기억하는지 확인",
        "mem_profile": "특정 주제에 대해 여러 세션의 정보를 종합",
        "mem_update": "이전 정보가 갱신된 경우 최신 정보를 반영하는지",
        "mem_no_hallucination": "기억에 없는 정보를 물었을 때 모른다고 답하는지",
        "mem_relevance": "기억을 맥락에 맞게 자연스럽게 활용하는지",
        "mem_multi_session": "주제 구분 없이 여러 세션을 종합 요약",
    }

    p.append('<div class="panel">')
    p.append('<div class="section-title">Suite별 요약</div>')
    if scored_mem_turns:
        p.append("<table>")
        p.append("<tr><th>Suite</th><th>테스트 내용</th><th>질문 수</th></tr>")
        suite_map: dict[str, list] = {}
        for t in scored_mem_turns:
            suite_map.setdefault(t["suite_name"], []).append(t)
        for suite_name in sorted(suite_map, key=lambda s: suite_rank.get(s, 10**6)):
            suite_turns = suite_map[suite_name]
            desc = _get_suite_desc(scored, suite_name)
            detail = _MEM_SUITE_DETAIL.get(suite_name, "")
            p.append(
                f'<tr><td style="font-weight:500">{desc}</td>'
                f'<td class="mute">{_esc(detail)}</td>'
                f"<td>{len(suite_turns)}</td></tr>"
            )
        p.append("</table>")
    p.append("</div>")

    p.append('<div class="panel">')
    p.append('<div class="section-title">평가 기준</div>')
    p.append('<div style="font-size:13px;line-height:1.8">')
    p.append('<div style="font-size:11px;font-weight:600;color:#86868b;margin-bottom:2px">검색 (자동 계산)</div>')
    p.append(
        '<div><span style="font-weight:600">Recall</span> <span class="mute">— 찾아야 할 에피소드 중 실제로 찾은 비율</span></div>'
    )
    p.append(
        '<div><span style="font-weight:600">Precision</span> <span class="mute">— 검색된 에피소드 중 관련 있는 비율</span></div>'
    )
    p.append('<div style="font-size:11px;font-weight:600;color:#86868b;margin:8px 0 2px">활용 (LLM Judge)</div>')
    p.append(
        '<div><span style="font-weight:600">응답 관련성</span> <span class="mute">— 질문에 맞는 응답인가</span></div>'
    )
    p.append(
        '<div><span style="font-weight:600">메모리 적절성</span> <span class="mute">— 기억을 자연스럽게 활용하는가</span></div>'
    )
    p.append(
        '<div><span style="font-weight:600">사실 정확성</span> <span class="mute">— 에피소드와 일치하는 응답인가</span></div>'
    )
    p.append(
        '<div><span style="font-weight:600">자연스러움</span> <span class="mute">— 대화체로 자연스러운가</span></div>'
    )
    p.append("</div>")
    p.append("</div>")

    p.append("</div>")

    # Per-probe results
    if scored_mem_turns:
        sorted_mem = sorted(scored_mem_turns, key=lambda t: q_rank.get(t["question_id"], 10**6))
        for suite_name, group in groupby(sorted_mem, key=lambda t: t["suite_name"]):
            suite_turns = list(group)
            desc = _get_suite_desc(scored, suite_name)
            desc_str = f' <span class="mute">— {_esc(desc)}</span>' if desc else ""
            p.append('<div class="suite-group">')
            p.append(f'<div class="suite-header">{suite_name}{desc_str}</div>')

            for t in suite_turns:
                ms = t.get("memory_scores", {})
                rc = t.get("retriever_recall")
                pr = t.get("retriever_precision")
                reasoning = t.get("memory_reasoning", "")

                p.append('<div class="item" style="display:grid;grid-template-columns:1fr 1fr;gap:4px 24px">')

                # Left: question + response
                p.append("<div>")
                p.append(
                    f'<div class="item-row"><span class="item-label mute">질문</span><span class="item-text">{_esc(t.get("input_text", ""))}</span></div>'
                )
                if t.get("response_text"):
                    p.append(
                        f'<div class="item-row"><span class="item-label mute">응답</span><span class="item-text">{_esc(t["response_text"])}</span></div>'
                    )
                p.append("</div>")

                # Right: scores + reasoning
                p.append("<div>")
                p.append('<div class="item-scores">')
                p.append(
                    f'<span class="mute" style="font-size:11px">{t["question_id"]}</span> <span class="mute" style="font-size:11px">|</span> '
                )
                if rc is not None:
                    cls = _score_cls(rc, 0.7, 0.4)
                    p.append(f'<span class="mono {cls}" style="font-size:11px">recall {rc:.0%}</span> ')
                if pr is not None:
                    cls = _score_cls(pr, 0.5, 0.2)
                    p.append(f'<span class="mono {cls}" style="font-size:11px">precision {pr:.0%}</span> ')
                for k, v in ms.items():
                    label = _MEMORY_CRITERION_LABELS.get(k, k)
                    p.append(f'{_score_chip(v)} <span class="mute" style="font-size:10px">{label}</span> ')
                p.append("</div>")
                if reasoning:
                    p.append(f'<div class="reasoning">{_esc(reasoning)}</div>')

                p.append("</div>")

                # Collapsible: retrieved episodes + missed targets (full width within the grid)
                retrieved = t.get("retrieved_episodes", [])
                target_ids = set(t.get("target_episode_ids", []))
                retrieved_ids = set()
                for ep in retrieved:
                    if isinstance(ep, dict):
                        eid = ep.get("episode_id") or ep.get("id")
                        if eid is not None:
                            retrieved_ids.add(eid)
                missed_ids = target_ids - retrieved_ids

                if retrieved or missed_ids:
                    label_parts = []
                    if retrieved:
                        label_parts.append(f"검색 {len(retrieved)}")
                    if missed_ids:
                        label_parts.append(f"미검색 {len(missed_ids)}")
                    p.append(
                        f'<div style="grid-column:1/-1"><div class="collapsible-toggle mute" style="margin-top:4px;font-size:11px">에피소드 ({" · ".join(label_parts)})</div>'
                    )
                    p.append('<div class="collapsible-body" style="margin-top:4px">')

                    if retrieved:
                        p.append(
                            '<div style="background:#f9fafb;border-radius:6px;padding:8px 10px;margin-bottom:4px">'
                        )
                        for ep in retrieved:
                            if isinstance(ep, dict):
                                ep_text = ep.get("text", str(ep))
                                eid = ep.get("episode_id") or ep.get("id")
                                score = ep.get("score")
                                score_str = (
                                    f' <span class="mono mute" style="font-size:10px">{score:.4f}</span>'
                                    if score
                                    else ""
                                )
                                is_target = eid in target_ids
                            else:
                                ep_text = str(ep)
                                score_str = ""
                                is_target = False
                            border_color = "#248a3d" if is_target else "#86868b"
                            target_mark = (
                                ' <span class="tag tag-ok" style="font-size:9px">target</span>' if is_target else ""
                            )
                            p.append(
                                f'<div style="font-size:12px;padding:4px 0 4px 10px;margin-bottom:3px;border-left:2px solid {border_color}">{_esc(ep_text)}{score_str}{target_mark}</div>'
                            )
                        p.append("</div>")

                    if missed_ids:
                        # Build episode text lookup from writer data
                        ep_texts: dict[int, str] = {}
                        for ws in writer.get("by_session", []):
                            for ep in ws.get("episodes", []):
                                ep_texts[ep["id"]] = ep["text"]

                        p.append('<div style="background:#fef2f2;border-radius:6px;padding:8px 10px">')
                        p.append(
                            '<div style="font-size:11px;font-weight:600;color:#991b1b;margin-bottom:4px">미검색 target 에피소드</div>'
                        )
                        for mid in sorted(missed_ids):
                            ep_text = ep_texts.get(mid, f"episode {mid}")
                            p.append(
                                f'<div style="font-size:12px;padding:4px 0 4px 10px;margin-bottom:3px;border-left:2px solid #dc2626">{_esc(ep_text)}</div>'
                            )
                        p.append("</div>")

                    p.append("</div></div>")

                p.append("</div>")
                p.append(_detail_block(t))

            p.append("</div>")

    p.append("</div>")

    return "\n".join(p)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_html(scored: dict) -> str:
    tab_defs = [
        ("overview", "Overview", _build_overview),
        ("asr", "ASR", _build_asr),
        ("turn-taking", "턴테이킹", _build_turn_taking),
        ("interruption", "인터럽션", _build_interruption),
        ("quality", "응답 품질", _build_quality),
        ("memory", "장기기억", _build_memory),
    ]

    # Tab bar
    tab_bar = []
    for i, (tid, label, _) in enumerate(tab_defs):
        active = " active" if i == 0 else ""
        tab_bar.append(f'<div class="tab{active}" data-target="{tid}">{label}</div>')

    # Tab panes
    panes = []
    for i, (tid, _, builder) in enumerate(tab_defs):
        active = " active" if i == 0 else ""
        panes.append(f'<div id="{tid}" class="tab-pane{active}">{builder(scored)}</div>')

    meta = scored.get("started_at", "")

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>Ray 평가 대시보드</title>
<style>{_CSS}</style></head><body>
<div class="header">
<h1>Ray 파이프라인 평가</h1>
<span class="meta">{meta}</span>
</div>
<div class="tabs">{"".join(tab_bar)}</div>
<div class="content">{"".join(panes)}</div>
<script>{_JS}</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval HTML dashboard")
    parser.add_argument("scored", help="Path to scored.json")
    parser.add_argument("--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    scored_path = Path(args.scored)
    if not scored_path.exists():
        print(f"Error: {scored_path} not found", file=sys.stderr)
        sys.exit(1)

    scored = json.loads(scored_path.read_text())
    html = build_html(scored)

    output_path = Path(args.output) if args.output else scored_path.parent / "dashboard.html"
    output_path.write_text(html)
    print(f"Dashboard saved: {output_path}")


if __name__ == "__main__":
    main()
