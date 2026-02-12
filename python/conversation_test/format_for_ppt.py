# -*- coding: utf-8 -*-
"""
테스트 결과 JSON을 PPT 붙여넣기용 이미지로 변환합니다.

사용법:
  텍스트 포맷:
    python format_for_ppt.py <JSON 파일> --style text

  PPT 이미지 포맷 (HTML → PNG):
    python format_for_ppt.py <JSON 파일> --style ppt

  출력 경로 지정:
    python format_for_ppt.py <JSON 파일> --style ppt -o ./my_output
"""

import os
import sys
import json
import argparse


def load_json(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── 텍스트 포맷 ──────────────────────────────────────────────────────────────

def format_text_style(data: dict) -> str:
    """줄 구분선이 있는 상세 텍스트 포맷"""
    meta = data["metadata"]
    lines = []

    lines.append("═" * 55)
    lines.append("[테스트 세팅]")
    lines.append(f"  프롬프트: {meta['prompt_name']}")
    lines.append(f"  모델:     {meta['model']}")
    lines.append(f"  도구:     {', '.join(meta['tools'])}")
    lines.append(f"  실행일시: {meta['timestamp'][:19].replace('T', ' ')}")
    if meta.get("model_presets"):
        presets_str = ", ".join(f"{k}={v}" for k, v in meta["model_presets"].items() if k != "model")
        if presets_str:
            lines.append(f"  프리셋:   {presets_str}")
    lines.append("═" * 55)
    lines.append("")

    for result in data["results"]:
        sid = result["scenario_id"]
        name = result["scenario_name"]
        category = result["category"]
        check = result["check_points"]

        lines.append(f"[{sid}] {name}")
        lines.append(f"카테고리: {category}")
        lines.append(f"검증 포인트: {check}")
        lines.append("─" * 55)

        for turn in result["conversation"]:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Ray:  {turn['ray']}")
            lines.append(f"      ({turn['response_time_sec']}s)")
            lines.append("")

        lines.append("─" * 55)
        lines.append("")

    return "\n".join(lines)


# ─── PPT 이미지 포맷 (HTML → PNG) ─────────────────────────────────────────────

SLIDE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Noto Sans KR', sans-serif;
    background: #F8F9FA;
    padding: 0;
    margin: 0;
  }}

  .slide {{
    width: {slide_w}px;
    min-height: 120px;
    background: #FFFFFF;
    padding: 36px 44px 32px 44px;
    position: relative;
  }}

  /* ── 세팅 헤더 (첫 슬라이드에만) ── */
  .setting-bar {{
    display: flex;
    gap: 24px;
    align-items: center;
    padding: 10px 20px;
    background: #F1F3F5;
    border-radius: 8px;
    margin-bottom: 28px;
    font-size: 13px;
    color: #495057;
    font-weight: 400;
  }}
  .setting-bar .label {{
    color: #868E96;
    font-weight: 500;
    margin-right: 4px;
  }}

  /* ── 시나리오 헤더 ── */
  .scenario-header {{
    margin-bottom: 20px;
  }}
  .scenario-id {{
    display: inline-block;
    background: #4263EB;
    color: #FFFFFF;
    font-size: 12px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 4px;
    margin-right: 10px;
    letter-spacing: 0.5px;
  }}
  .scenario-name {{
    font-size: 20px;
    font-weight: 700;
    color: #212529;
    vertical-align: middle;
  }}
  .scenario-meta {{
    margin-top: 6px;
    font-size: 12.5px;
    color: #868E96;
  }}

  /* ── 대화 영역 ── */
  .conversation {{
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}

  .turn {{
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}

  .user-row, .ray-row {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
  }}

  .role-tag {{
    flex-shrink: 0;
    width: 42px;
    font-size: 12px;
    font-weight: 600;
    padding: 3px 0;
    text-align: center;
    border-radius: 4px;
    margin-top: 1px;
  }}

  .role-tag.user {{
    background: #E8F4FD;
    color: #1971C2;
  }}

  .role-tag.ray {{
    background: #FFF3E0;
    color: #E8590C;
  }}

  .message-text {{
    font-size: 14.5px;
    line-height: 1.65;
    color: #343A40;
    flex: 1;
    padding-top: 1px;
  }}

  .message-text.ray-text {{
    color: #495057;
  }}

  .turn-divider {{
    border: none;
    border-top: 1px solid #F1F3F5;
    margin: 2px 0;
  }}

  .response-time {{
    font-size: 11px;
    color: #ADB5BD;
    margin-left: 52px;
    margin-top: -2px;
  }}
</style>
</head>
<body>
{content}
</body>
</html>"""


def build_slide_html(result: dict, metadata: dict, slide_width: int, include_settings: bool = True) -> str:
    """하나의 시나리오를 슬라이드 HTML로 변환"""
    parts = []

    parts.append(f'<div class="slide">')

    # 세팅 바
    if include_settings:
        meta = metadata
        parts.append(f'''<div class="setting-bar">
  <span><span class="label">Model</span> {meta["model"]}</span>
  <span><span class="label">Prompt</span> {meta["prompt_name"]}</span>
  <span><span class="label">Tools</span> {", ".join(meta["tools"])}</span>
  <span><span class="label">Date</span> {meta["timestamp"][:10]}</span>
</div>''')

    # 시나리오 헤더
    parts.append(f'''<div class="scenario-header">
  <span class="scenario-id">{result["scenario_id"]}</span>
  <span class="scenario-name">{result["scenario_name"]}</span>
  <div class="scenario-meta">{result["category"]}  ·  {result["check_points"]}</div>
</div>''')

    # 대화
    parts.append('<div class="conversation">')
    for i, turn in enumerate(result["conversation"]):
        parts.append('<div class="turn">')

        # User
        user_text = turn["user"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(f'''  <div class="user-row">
    <span class="role-tag user">User</span>
    <span class="message-text">{user_text}</span>
  </div>''')

        # Ray
        ray_text = turn["ray"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(f'''  <div class="ray-row">
    <span class="role-tag ray">Ray</span>
    <span class="message-text ray-text">{ray_text}</span>
  </div>''')

        # Response time
        parts.append(f'  <div class="response-time">{turn["response_time_sec"]}s</div>')

        parts.append('</div>')  # .turn

        # Divider (마지막 턴 제외)
        if i < len(result["conversation"]) - 1:
            parts.append('<hr class="turn-divider">')

    parts.append('</div>')  # .conversation
    parts.append('</div>')  # .slide

    return "\n".join(parts)


def generate_ppt_images(data: dict, output_dir: str, slide_width: int = 960):
    """시나리오별 HTML을 생성하고 PNG 이미지로 캡처합니다."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ playwright가 설치되어 있지 않습니다.")
        print("   설치: pip install playwright && playwright install chromium")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    metadata = data["metadata"]
    results = data["results"]

    print(f"📐 슬라이드 너비: {slide_width}px")
    print(f"📁 출력 폴더: {output_dir}")
    print(f"🎬 {len(results)}개 시나리오 렌더링 중...\n")

    saved_files = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for result in results:
            sid = result["scenario_id"]
            name = result["scenario_name"]

            # HTML 생성
            slide_content = build_slide_html(result, metadata, slide_width, include_settings=True)
            full_html = SLIDE_HTML_TEMPLATE.format(content=slide_content, slide_w=slide_width)

            # HTML 파일 저장 (디버깅용)
            html_path = os.path.join(output_dir, f"slide_{sid}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(full_html)

            # 렌더링 & 캡처
            page.set_viewport_size({"width": slide_width + 20, "height": 800})
            page.goto(f"file:///{os.path.abspath(html_path).replace(os.sep, '/')}")
            page.wait_for_load_state("networkidle")

            # .slide 요소를 정확히 캡처
            slide_el = page.query_selector(".slide")
            if slide_el:
                png_path = os.path.join(output_dir, f"slide_{sid}.png")
                slide_el.screenshot(path=png_path)
                saved_files.append(png_path)
                print(f"  ✅ [{sid}] {name} → {os.path.basename(png_path)}")
            else:
                print(f"  ⚠️ [{sid}] 렌더링 실패")

        browser.close()

    print(f"\n🎉 완료! {len(saved_files)}개 이미지 저장됨: {output_dir}")
    return saved_files


# ─── 메인 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="테스트 결과를 PPT용 포맷으로 변환")
    parser.add_argument("input", help="입력 JSON 파일 경로")
    parser.add_argument("--style", choices=["text", "ppt"], default="ppt",
                        help="출력 스타일 (text: 텍스트, ppt: HTML→이미지)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="출력 경로 (text: 파일, ppt: 폴더)")
    parser.add_argument("--width", type=int, default=960,
                        help="슬라이드 너비 픽셀 (기본: 960, PPT 16:9 기준 적정값)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    data = load_json(args.input)

    if args.style == "text":
        formatted = format_text_style(data)
        out_path = args.output or f"{os.path.splitext(args.input)[0]}_text.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(formatted)
        print(f"✅ 텍스트 변환 완료: {out_path}")

    elif args.style == "ppt":
        out_dir = args.output or f"{os.path.splitext(args.input)[0]}_slides"
        generate_ppt_images(data, out_dir, slide_width=args.width)


if __name__ == "__main__":
    main()
