# -*- coding: utf-8 -*-
"""
睿聘智模 v3.1  ·  app.py  Flask 后端
支持：会话管理、JD/简历输入、AI分析、面试对话、报告生成、HTML/PDF/DOCX导出
"""

import os
import uuid
import base64
import io
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

from config import (SERVER_HOST, SERVER_PORT, DEBUG_MODE, SECRET_KEY,
                    MAX_UPLOAD_MB, INTERVIEW_MODES)
from interviewer_engine import InterviewerEngine

# ── 关键修复：使用绝对路径，保证在任意工作目录下启动均能找到文件 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": "*"}})

# 内存 session 存储（每个 uuid → InterviewerEngine）
_sessions: dict[str, InterviewerEngine] = {}


def get_eng(sid: str) -> InterviewerEngine | None:
    return _sessions.get(sid)


# ── 静态文件 ──────────────────────────────────────────────────────
@app.route("/")
def index():
    # 优先从项目根目录提供 index.html（index.html 与 app.py 同级）
    return send_from_directory(BASE_DIR, "index.html")


# ── 会话创建 / 重置 ───────────────────────────────────────────────
@app.route("/api/session/create", methods=["POST"])
def session_create():
    data     = request.get_json(force=True) or {}
    settings = data.get("settings", {})
    sid      = str(uuid.uuid4())
    _sessions[sid] = InterviewerEngine(settings)
    return jsonify({"ok": True, "session_id": sid})


@app.route("/api/session/reset", methods=["POST"])
def session_reset():
    """退出当前面试，开启新面试（清理旧引擎）"""
    data     = request.get_json(force=True) or {}
    old_sid  = data.get("session_id", "")
    settings = data.get("settings", {})
    if old_sid in _sessions:
        del _sessions[old_sid]
    sid = str(uuid.uuid4())
    _sessions[sid] = InterviewerEngine(settings)
    return jsonify({"ok": True, "session_id": sid})


@app.route("/api/session/status", methods=["GET"])
def session_status():
    sid = request.args.get("sid", "")
    eng = get_eng(sid)
    if not eng:
        return jsonify({"ok": False, "error": "session not found"})
    return jsonify({
        "ok":              True,
        "has_jd":          bool(eng.jd),
        "has_resume":      bool(eng.resume),
        "has_plan":        bool(eng.plan),
        "current_q_idx":  eng.current_q_idx,
        "total_questions": len(eng.all_questions),
        "evaluations":     len(eng.evaluations),
        "completed":       (eng.current_q_idx >= len(eng.all_questions) > 0),
        "interview_mode":  eng.interview_mode,
        "duration_minutes":eng.duration_minutes,
        "max_followup":    eng.max_followup,
    })


# ── JD 输入 ───────────────────────────────────────────────────────
@app.route("/api/jd", methods=["POST"])
def submit_jd():
    data = request.get_json(force=True) or {}
    sid  = data.get("session_id", "")
    jd   = data.get("jd", "").strip()
    eng  = get_eng(sid)
    if not eng:
        return jsonify({"ok": False, "error": "会话不存在，请刷新"})
    if not jd:
        return jsonify({"ok": False, "error": "JD内容不能为空"})
    eng.jd = jd
    return jsonify({"ok": True, "length": len(jd)})


# ── 简历：文本粘贴 ────────────────────────────────────────────────
@app.route("/api/resume/text", methods=["POST"])
def submit_resume_text():
    data   = request.get_json(force=True) or {}
    sid    = data.get("session_id", "")
    resume = data.get("resume", "").strip()
    eng    = get_eng(sid)
    if not eng:
        return jsonify({"ok": False, "error": "会话不存在"})
    if not resume:
        return jsonify({"ok": False, "error": "简历内容不能为空"})
    eng.resume = resume
    return jsonify({"ok": True, "length": len(resume)})


# ── 简历：文件上传（PDF/DOCX/TXT/图片）──────────────────────────
@app.route("/api/resume/file", methods=["POST"])
def submit_resume_file():
    sid = request.form.get("session_id", "")
    eng = get_eng(sid)
    if not eng:
        return jsonify({"ok": False, "error": "会话不存在"})
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "未接收到文件"})

    f     = request.files["file"]
    fname = f.filename.lower()
    raw   = f.read()
    text  = ""

    try:
        if fname.endswith(".pdf"):
            import fitz
            doc  = fitz.open(stream=raw, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)

        elif fname.endswith((".docx", ".doc")):
            from docx import Document
            doc  = Document(io.BytesIO(raw))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif fname.endswith((".png",".jpg",".jpeg",".gif",".webp",".bmp")):
            ext  = fname.split(".")[-1]
            mime = "image/jpeg" if ext in ("jpg","jpeg") else f"image/{ext}"
            b64  = base64.b64encode(raw).decode()
            from interviewer_engine import extract_text_from_image
            text = extract_text_from_image(b64, mime)

        elif fname.endswith(".txt"):
            text = raw.decode("utf-8", errors="ignore")
        else:
            return jsonify({"ok": False, "error": f"不支持的文件格式"})

        text = text.strip()
        if not text:
            return jsonify({"ok": False, "error": "文件解析结果为空，请检查文件内容"})

        eng.resume = text
        return jsonify({"ok": True, "length": len(text), "preview": text[:300]})

    except ImportError as e:
        return jsonify({"ok": False, "error": f"缺少依赖库，请运行 pip install -r requirements.txt"})
    except Exception as e:
        return jsonify({"ok": False, "error": f"文件解析失败：{str(e)}"})


# ── AI 分析（岗位识别 + 权重 + 题数 + 方案）────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True) or {}
    sid  = data.get("session_id", "")
    eng  = get_eng(sid)
    if not eng:
        return jsonify({"ok": False, "error": "会话不存在"})
    if not eng.jd:
        return jsonify({"ok": False, "error": "请先输入JD"})
    if not eng.resume:
        return jsonify({"ok": False, "error": "请先上传简历"})

    try:
        pos_info   = eng.analyze_position()
        q_info     = eng.determine_question_count()
        plan       = eng.build_plan()

        return jsonify({
            "ok": True,
            "position_type":    eng.position_type,
            "seniority":        eng.seniority,
            "weights":          eng.weights,
            "weight_reasoning": pos_info.get("weight_reasoning", {}),
            "key_competencies": eng.key_competencies,
            "red_flags":        eng.red_flags,
            "q_allocation":     eng.q_allocation,
            "total_questions":  len(eng.all_questions),
            "q_reasoning":      q_info.get("reasoning",""),
            "plan": {
                "position_title":      plan.get("position_title",""),
                "match_score":         plan.get("match_score",0),
                "match_analysis":      plan.get("match_analysis",""),
                "candidate_highlights":plan.get("candidate_highlights",[]),
                "candidate_gaps":      plan.get("candidate_gaps",[]),
                "interview_strategy":  plan.get("interview_strategy",""),
                "dimensions":[{"id":d.get("dimension_id"),"name":d.get("name"),
                                "weight":d.get("weight"),"q_count":len(d.get("questions",[]))}
                               for d in plan.get("dimensions",[])],
            },
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ── 获取当前面试问题 ──────────────────────────────────────────────
@app.route("/api/interview/question", methods=["POST"])
def get_question():
    data = request.get_json(force=True) or {}
    sid  = data.get("session_id","")
    eng  = get_eng(sid)
    if not eng:
        return jsonify({"ok": False, "error": "会话不存在"})
    if not eng.all_questions:
        return jsonify({"ok": False, "error": "面试方案尚未构建"})

    result = eng.get_current_question_display()
    return jsonify({"ok": True, **result})


# ── 提交回答（主回答 or 追问回答）───────────────────────────────
@app.route("/api/interview/answer", methods=["POST"])
def submit_answer():
    data       = request.get_json(force=True) or {}
    sid        = data.get("session_id","")
    answer     = data.get("answer","").strip()
    is_followup= data.get("is_followup", False)
    eng        = get_eng(sid)

    if not eng:
        return jsonify({"ok": False, "error": "会话不存在"})
    if not answer:
        return jsonify({"ok": False, "error": "回答内容不能为空"})

    try:
        result = eng.process_followup_answer(answer) if is_followup else eng.process_answer(answer)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ── 生成最终报告 ──────────────────────────────────────────────────
@app.route("/api/interview/final", methods=["POST"])
def final_report():
    data = request.get_json(force=True) or {}
    sid  = data.get("session_id","")
    eng  = get_eng(sid)
    if not eng:
        return jsonify({"ok": False, "error": "会话不存在"})
    try:
        result = eng.generate_final_report()
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ── 导出报告 ─────────────────────────────────────────────────────
@app.route("/api/export/<fmt>", methods=["GET"])
def export_report(fmt: str):
    sid = request.args.get("sid","")
    eng = get_eng(sid)
    if not eng:
        return jsonify({"ok": False, "error": "会话不存在"}), 404

    title = eng.plan.get("position_title","面试报告").replace(" ","_")

    if fmt == "html":
        data = eng.export_html().encode("utf-8")
        return send_file(io.BytesIO(data), mimetype="text/html; charset=utf-8",
                         as_attachment=True,
                         download_name=f"睿聘智模_面试报告_{title}.html")

    elif fmt == "pdf":
        data = eng.export_pdf()
        if not data:
            return jsonify({"ok":False,
                            "error":"PDF导出失败，请运行: pip install reportlab"}), 500
        return send_file(io.BytesIO(data),
                         mimetype="application/pdf",
                         as_attachment=True,
                         download_name=f"睿聘智模_面试报告_{title}.pdf")

    elif fmt == "docx":
        data = eng.export_docx()
        if not data:
            return jsonify({"ok":False,
                            "error":"DOCX导出失败，请运行: pip install python-docx"}), 500
        return send_file(io.BytesIO(data),
                         mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                         as_attachment=True,
                         download_name=f"睿聘智模_面试报告_{title}.docx")

    return jsonify({"ok":False,"error":f"不支持的格式：{fmt}"}), 400


# ── 系统配置（供前端展示）────────────────────────────────────────
@app.route("/api/config", methods=["GET"])
def get_config():
    from config import (DEFAULT_INTERVIEW_MODE, DEFAULT_INTERVIEW_DURATION_MINUTES,
                        DEFAULT_MAX_FOLLOWUP_ROUNDS, DEFAULT_DEPTH_SCORE_THRESHOLD,
                        INTERVIEWER_NAME)
    return jsonify({
        "interviewer_name": INTERVIEWER_NAME,
        "default_mode":     DEFAULT_INTERVIEW_MODE,
        "default_duration": DEFAULT_INTERVIEW_DURATION_MINUTES,
        "default_followup": DEFAULT_MAX_FOLLOWUP_ROUNDS,
        "depth_threshold":  DEFAULT_DEPTH_SCORE_THRESHOLD,
        "available_modes":  {k: {"name":v["name"],"description":v["description"]}
                             for k,v in INTERVIEW_MODES.items()},
    })


# ── 启动 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  🎯  睿聘智模 v3.1  AI  智能面试系统")
    print("═"*60)
    print(f"  项目目录：{BASE_DIR}")
    print(f"  本地访问：http://localhost:{SERVER_PORT}")
    print(f"  外部分享：ngrok http {SERVER_PORT}")
    print("═"*60 + "\n")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=DEBUG_MODE)