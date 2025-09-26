import os
import re
import io
import json
import time
import fitz  # pymupdf
import streamlit as st
from datetime import datetime
from collections import deque
from typing import List, Tuple

# PDF feedback (Unicode PDF)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Groq LLM
from groq import Groq

# Optional: push audit to HF Datasets
from huggingface_hub import HfApi

APP_TITLE = "Перевірка якості робіт студентів"

# Паролі й ключі через секрети HF Spaces
SIMPLE_PASSWORD = os.environ.get("APP_PASSWORD", "class2025")
TEACHER_PASSWORD = os.environ.get("TEACHER_PASSWORD", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Опційно для аудиту у Datasets
HF_TOKEN = os.environ.get("HF_TOKEN", "")
AUDIT_DATASET_REPO = os.environ.get("AUDIT_DATASET_REPO", "")
AUDIT_LOCAL_PATH = "data/audit.jsonl"

# Менеджер запитів (2 req/min за замовчуванням)
REQUEST_WINDOW_SEC = 60
REQUEST_LIMIT = 2
_request_times = deque(maxlen=REQUEST_LIMIT * 2)

# Довідники
DEFAULT_DISCIPLINES = [
    "Системний аналіз",
    "Інформаційні технології аналізу даних",
    "Інтелектуальний аналіз даних",
]
REPORT_TYPES = ["Лабораторна робота", "Курсова робота", "Індивідуальне завдання"]

# Groq клієнт
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Шрифт для кирилиці в PDF
FONT_PATH = "fonts/DejaVuSans.ttf"
if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont("DejaVuSans", FONT_PATH))
    PDF_FONT_NAME = "DejaVuSans"
else:
    PDF_FONT_NAME = "Helvetica"  # fallback (можуть бути проблеми з кирилицею)

# ---------- Допоміжні функції ----------

@st.cache_resource
def load_prompts():
    with open("prompts.json", "r", encoding="utf-8") as f:
        return json.load(f)

def extract_text_pages(file_bytes: bytes) -> List[str]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return [page.get_text("text") for page in doc]

def count_images_in_pages(file_bytes: bytes, page_indices: List[int]) -> int:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    cnt = 0
    for i in page_indices:
        if 0 <= i < len(doc):
            cnt += len(doc[i].get_images(full=True))
    return cnt

def find_section_page_range(pages_text: List[str], title: str) -> Tuple[int, int]:
    if not title:
        return -1, -1
    title_norm = title.strip().lower()
    start = -1
    for i, txt in enumerate(pages_text):
        if title_norm in txt.lower():
            start = i
            break
    if start == -1:
        return -1, -1
    pattern = re.compile(r"^\s*(Розділ|Section|Висновки|Conclusions)\b",
                         re.IGNORECASE | re.MULTILINE)
    end = len(pages_text) - 1
    for j in range(start + 1, len(pages_text)):
        if pattern.search(pages_text[j]):
            end = j - 1
            break
    return start, end

def build_prompt(mapping, degree, discipline, report_type, work_no, variant):
    k_full = f"{degree}|{discipline}|{report_type}|{work_no}|{variant}"
    k_no_variant = f"{degree}|{discipline}|{report_type}|{work_no}"
    k_legacy = f"{degree}|{discipline}|{report_type}|{variant}"
    k_simple = f"{degree}|{discipline}|{report_type}"

    if k_full in mapping:
        return mapping[k_full], k_full
    if k_no_variant in mapping:
        return mapping[k_no_variant], k_no_variant
    if k_legacy in mapping:
        return mapping[k_legacy], k_legacy
    if k_simple in mapping:
        return mapping[k_simple], k_simple
    return None, k_full

def call_llm(system_prompt: str, report_text: str) -> str:
    """Виклик Llama 3.1 8B через Groq Chat Completions."""
    # Локальний ліміт запитів
    now = time.time()
    while _request_times and now - _request_times[0] > REQUEST_WINDOW_SEC:
        _request_times.popleft()
    if len(_request_times) >= REQUEST_LIMIT:
        return "Сервіс перевантажений, прошу зайдіть через 30 секунд."

    if client is None:
        return "Не встановлено GROQ_API_KEY у Secrets (Settings → Secrets)."

    try:
        messages = [
            {"role": "system",
             "content": system_prompt},
            {"role": "user",
             "content": "=== STUDENT REPORT (EXTRACT) ===\n" + report_text[:150000]}
        ]
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2,
            max_tokens=2048
        )
        _request_times.append(now)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        msg = str(e)
        if "429" in msg or "rate" in msg.lower():
            return "Сервіс перевантажений, прошу зайдіть через 30 секунд."
        return f"Помилка: {msg}"

def make_pdf_from_text(text: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    left, top, line_h = 40, height - 40, 14
    c.setFont(PDF_FONT_NAME, 11)
    y = top
    for line in text.splitlines():
        while len(line) > 110:
            part, line = line[:110], line[110:]
            if y < 40:
                c.showPage()
                c.setFont(PDF_FONT_NAME, 11)
                y = top
            c.drawString(left, y, part)
            y -= line_h
        if y < 40:
            c.showPage()
            c.setFont(PDF_FONT_NAME, 11)
            y = top
        c.drawString(left, y, line)
        y -= line_h
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

def append_audit(record: dict):
    os.makedirs(os.path.dirname(AUDIT_LOCAL_PATH), exist_ok=True)
    with open(AUDIT_LOCAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if HF_TOKEN and AUDIT_DATASET_REPO:
        try:
            api = HfApi(token=HF_TOKEN)
            api.upload_file(
                path_or_fileobj=AUDIT_LOCAL_PATH,
                path_in_repo="audit.jsonl",
                repo_id=AUDIT_DATASET_REPO,
                repo_type="dataset",
            )
        except Exception as e:
            print("[AUDIT UPLOAD ERROR]", e)

# ---------- UI ----------

st.set_page_config(page_title=APP_TITLE, page_icon="✅", layout="centered")
st.markdown(
    f"""
    <div style="border:2px solid #1f4ba8; padding:16px; border-radius:8px;">
      <h2 style="color:#163a7a; margin-top:0;">{APP_TITLE}</h2>
    """,
    unsafe_allow_html=True
)

# 1) Доступ
pwd = st.text_input("Пароль доступу", type="password")
if pwd != SIMPLE_PASSWORD:
    st.info("Введіть пароль, наданий викладачем.")
    st.stop()

# 2) Параметри
col1, col2 = st.columns(2)
degree = col1.selectbox("Рівень навчання", ["Бакалавр", "Магістр", "Доктор філософії"])
discipline = col2.selectbox("Дисципліна", DEFAULT_DISCIPLINES)
report_type = col1.selectbox("Вид звіту", REPORT_TYPES)

# Номер роботи
if report_type == "Лабораторна робота":
    work_no = col2.selectbox("Номер роботи", [str(i) for i in range(1, 8)])
else:
    work_no = col2.selectbox("Номер роботи", ["1"])

variant = col1.selectbox("Варіант", [str(i) for i in range(1, 21)])

uploaded = st.file_uploader("Завантажте PDF-звіт студента", type=["pdf"])
section_title = st.text_input("Назва розділу для підрахунку графіків (необов’язково)")

btn = st.button("Перевірити")

# Кнопка завантаження локального аудиту
if os.path.exists(AUDIT_LOCAL_PATH):
    with open(AUDIT_LOCAL_PATH, "rb") as f:
        st.download_button("Завантажити журнал аудиту (JSONL)", data=f,
                           file_name="audit.jsonl", mime="application/jsonl")

if btn:
    if uploaded is None:
        st.warning("Будь ласка, додайте PDF-файл звіту.")
        st.stop()

    # Зчитування PDF
    with st.spinner("Зчитування PDF..."):
        file_bytes = uploaded.getvalue()
        pages_text = extract_text_pages(file_bytes)
        full_text = "\n".join(pages_text)

    # Підрахунок графіків
    graphs_msg = ""
    if section_title.strip():
        start, end = find_section_page_range(pages_text, section_title)
        if start == -1:
            graphs_msg = f"Розділ «{section_title}» не знайдено у PDF."
            section_pages = []
        else:
            section_pages = list(range(start, end + 1))
            img_count = count_images_in_pages(file_bytes, section_pages)
            if img_count >= 5:
                graphs_msg = f"✅ У розділі «{section_title}» знайдено щонайменше 5 графіків/рисунків (всього: {img_count})."
            else:
                graphs_msg = f"⚠️ У розділі «{section_title}» знайдено лише {img_count} графіків/рисунків (<5)."

    # Пошук промпта
    with st.spinner("Підбір промпта..."):
        mapping = load_prompts()
        system_prompt, matched_key = build_prompt(mapping, degree, discipline, report_type, work_no, variant)

    if system_prompt is None:
        msg = (f"Для варіанту \"Рівень - {degree} | Дисципліна \"{discipline}\" | "
               f"{report_type} | № {work_no} | варіант {variant}\" промпту для перевірки не існує - "
               "перевірте параметри і виберіть інші.")
        st.error(msg)
        append_audit({
            "ts": datetime.utcnow().isoformat(),
            "user": "student",
            "degree": degree,
            "discipline": discipline,
            "report_type": report_type,
            "work_no": work_no,
            "variant": variant,
            "filename": uploaded.name,
            "section_title": section_title,
            "result": "NO_PROMPT",
            "message": msg
        })
        st.stop()

    # Виклик LLM (Groq)
    with st.spinner("Запит до Llama 3.1 (Groq)..."):
        result_text = call_llm(system_prompt, full_text)

    if graphs_msg:
        result_text = graphs_msg + "\n\n" + result_text

    st.subheader("Рекомендації та зауваження")
    st.text_area("Результат", value=result_text, height=320)

    # Завантаження результатів
    txt_bytes = result_text.encode("utf-8")
    st.download_button("Завантажити як TXT", data=txt_bytes,
                       file_name="lab_feedback.txt", mime="text/plain")

    pdf_out = make_pdf_from_text(result_text)
    st.download_button("Завантажити як PDF", data=pdf_out,
                       file_name="lab_feedback.pdf", mime="application/pdf")

    # Аудит
    append_audit({
        "ts": datetime.utcnow().isoformat(),
        "user": "student",
        "degree": degree,
        "discipline": discipline,
        "report_type": report_type,
        "work_no": work_no,
        "variant": variant,
        "filename": uploaded.name,
        "section_title": section_title,
        "graphs_note": graphs_msg,
        "prompt_key_used": matched_key,
        "result": "OK" if not result_text.startswith("Помилка") else "ERROR"
    })

# ---------- Кабінет викладача ----------
import pandas as pd

with st.expander("Кабінет викладача — перегляд журналу / експорт", expanded=False):
    t_pwd = st.text_input("Пароль викладача", type="password", key="teacher_pwd")
    if t_pwd and TEACHER_PASSWORD and t_pwd == TEACHER_PASSWORD:
        st.success("Доступ дозволено.")
        if not os.path.exists(AUDIT_LOCAL_PATH):
            st.info("Локальний журнал ще не створено (файл data/audit.jsonl відсутній).")
        else:
            rows = []
            with open(AUDIT_LOCAL_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
            if not rows:
                st.info("Журнал порожній.")
            else:
                df = pd.DataFrame(rows)
                if "ts" in df.columns:
                    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
                    df["date"] = df["ts"].dt.date

                f_col1, f_col2, f_col3 = st.columns(3)
                degree_filter = f_col1.multiselect("Рівень", sorted(df.get("degree", pd.Series(dtype=str)).dropna().unique()))
                discipline_filter = f_col2.multiselect("Дисципліна", sorted(df.get("discipline", pd.Series(dtype=str)).dropna().unique()))
                report_filter = f_col3.multiselect("Тип звіту", sorted(df.get("report_type", pd.Series(dtype=str)).dropna().unique()))

                f_col4, f_col5, f_col6 = st.columns(3)
                work_no_filter = f_col4.multiselect("№ роботи", sorted(df.get("work_no", pd.Series(dtype=str)).dropna().unique()))
                variant_filter = f_col5.multiselect("Варіант", sorted(df.get("variant", pd.Series(dtype=str)).dropna().unique()))
                result_filter = f_col6.multiselect("Статус результату", sorted(df.get("result", pd.Series(dtype=str)).dropna().unique()))

                date_min = df["date"].min() if "date" in df else None
                date_max = df["date"].max() if "date" in df else None
                if date_min is not None and date_max is not None:
                    date_from, date_to = st.date_input("Діапазон дат (UTC)", (date_min, date_max))
                else:
                    date_from, date_to = None, None

                fdf = df.copy()
                if degree_filter:
                    fdf = fdf[fdf["degree"].isin(degree_filter)]
                if discipline_filter:
                    fdf = fdf[fdf["discipline"].isin(discipline_filter)]
                if report_filter:
                    fdf = fdf[fdf["report_type"].isin(report_filter)]
                if work_no_filter:
                    fdf = fdf[fdf["work_no"].isin(work_no_filter)]
                if variant_filter:
                    fdf = fdf[fdf["variant"].isin(variant_filter)]
                if result_filter:
                    fdf = fdf[fdf["result"].isin(result_filter)]
                if date_from is not None and date_to is not None and "date" in fdf:
                    fdf = fdf[(fdf["date"] >= pd.to_datetime(date_from)) & (fdf["date"] <= pd.to_datetime(date_to))]

                st.caption(f"Записів після фільтрів: {len(fdf)}")
                st.dataframe(fdf.sort_values(by="ts", ascending=False), use_container_width=True)

                exp_col1, exp_col2 = st.columns(2)
                csv_bytes = fdf.to_csv(index=False).encode("utf-8-sig")
                exp_col1.download_button("Експорт у CSV", data=csv_bytes,
                                         file_name="audit_filtered.csv", mime="text/csv")

                xlsx_buffer = io.BytesIO()
                with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
                    fdf.to_excel(writer, index=False, sheet_name="audit")
                xlsx_buffer.seek(0)
                exp_col2.download_button("Експорт у XLSX", data=xlsx_buffer.getvalue(),
                    file_name="audit_filtered.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.caption("Введіть пароль викладача, щоб переглянути журнал.")

st.markdown('<div style="text-align:right;color:#163a7a;">Розроблено в НДЛ ШІК та НДЛ ПВШ кафедри САІТ ФІІТА ВНТУ у 2025 р.</div></div>', unsafe_allow_html=True)
