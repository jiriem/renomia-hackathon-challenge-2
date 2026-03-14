"""
Challenge 2: Vyčítání dat ze souborů (Document Data Extraction)

Input:  OCR text from insurance contract documents (main contract + amendments)
Output: Structured CRM fields extracted from the documents
"""

import os
import json
import hashlib
import logging
import re
import threading
import time

from google import genai
from google.genai import types
import psycopg2
from fastapi import FastAPI
from fastapi import HTTPException
import uvicorn

app = FastAPI(title="Challenge 2: Document Data Extraction")
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://hackathon:hackathon@localhost:5432/hackathon"
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


class GeminiTracker:
    """Wrapper around Gemini that tracks token usage."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.enabled = bool(api_key)
        if self.enabled:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
        self.model_name = model_name
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
        self._lock = threading.Lock()

    def generate(self, prompt, **kwargs):
        if not self.enabled:
            raise RuntimeError("Gemini API key not configured")

        config = {}
        generation_config = kwargs.pop("generation_config", None) or {}
        direct_config = kwargs.pop("config", None)
        if isinstance(generation_config, dict):
            config.update(generation_config)
        if isinstance(direct_config, dict):
            config.update(direct_config)
        elif direct_config is not None:
            config = direct_config

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config or None,
        )
        with self._lock:
            self.request_count += 1
            meta = getattr(response, "usage_metadata", None)
            if meta is None and hasattr(response, "model_dump"):
                dumped = response.model_dump()
                meta = dumped.get("usage_metadata") or dumped.get("usageMetadata")
            if meta:
                def _meta_get(obj, *names):
                    if obj is None:
                        return 0
                    for name in names:
                        if isinstance(obj, dict) and name in obj:
                            return obj.get(name) or 0
                        value = getattr(obj, name, None)
                        if value is not None:
                            return value
                    return 0

                self.prompt_tokens += _meta_get(
                    meta, "prompt_token_count", "promptTokenCount"
                )
                self.completion_tokens += (
                    _meta_get(
                        meta,
                        "candidates_token_count",
                        "candidatesTokenCount",
                        "response_token_count",
                        "responseTokenCount",
                    )
                )
                self.total_tokens += _meta_get(
                    meta, "total_token_count", "totalTokenCount"
                )
        return response

    def get_metrics(self):
        with self._lock:
            return {
                "gemini_request_count": self.request_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }

    def reset(self):
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.request_count = 0


gemini = GeminiTracker(GEMINI_API_KEY)


def get_db():
    return psycopg2.connect(DATABASE_URL)


@app.on_event("startup")
def init_db():
    for _ in range(15):
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )"""
            )
            conn.commit()
            cur.close()
            conn.close()
            return
        except Exception:
            time.sleep(1)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return gemini.get_metrics()


@app.post("/metrics/reset")
def reset_metrics():
    gemini.reset()
    return {"status": "reset"}


@app.post("/solve")
def solve(payload: dict):
    """
    Extract structured CRM fields from insurance contract documents.

    Input example:
    {
        "documents": [
            {
                "pdf_url": "https://storage.googleapis.com/.../smlouva.pdf",
                "filename": "smlouva_hlavni.pdf",
                "ocr_text": "... OCR extracted text of main contract ..."
            },
            {
                "pdf_url": "https://storage.googleapis.com/.../dodatek_1.pdf",
                "filename": "dodatek_1.pdf",
                "ocr_text": "... OCR text of amendment 1 ..."
            },
            {
                "pdf_url": "https://storage.googleapis.com/.../dodatek_2.pdf",
                "filename": "dodatek_2.pdf",
                "ocr_text": "... OCR text of amendment 2 ..."
            }
        ]
    }

    Expected output (all fields from CRM template):
    {
        "contractNumber": "POJ-2024-12345",
        "insurerName": "Generali Česká pojišťovna a.s.",
        "state": "accepted",              // enum: draft | accepted | cancelled
        "assetType": "other",              // enum: other | vehicle
        "concludedAs": "broker",           // enum: agent | broker
        "contractRegime": "individual",    // enum: individual | frame | fleet | coinsurance
        "startAt": "01.01.2024",           // DD.MM.YYYY
        "endAt": null,                     // DD.MM.YYYY or null (doba neurčitá)
        "concludedAt": "15.12.2023",       // DD.MM.YYYY
        "installmentNumberPerInsurancePeriod": 4,  // 1=yearly, 2=semi, 4=quarterly, 12=monthly
        "insurancePeriodMonths": 12,       // 12=yearly, 6=semi, 3=quarterly, 1=monthly
        "premium": {
            "currency": "czk",             // ISO 4217 lowercase
            "isCollection": false          // true if broker collects premium
        },
        "actionOnInsurancePeriodTermination": "auto-renewal",  // auto-renewal | policy-termination
        "noticePeriod": "six-weeks",       // enum or null
        "regPlate": null,                  // only for vehicle insurance
        "latestEndorsementNumber": "3",    // string, highest amendment number or null
        "note": null                       // special conditions summary or null
    }
    """
    # TODO: Implement your solution here
    #
    # Suggested approach:
    # 1. Concatenate OCR text from all documents (main contract + amendments)
    # 2. Send to Gemini with a structured extraction prompt
    # 3. Parse the response into the expected field format
    # 4. For amendments: use the latest values (amendments override base contract)
    # 5. For latestEndorsementNumber: find the highest amendment number

    documents = payload.get("documents", [])
    if not documents:
        logger.error("Solve called without any documents in payload")
        raise HTTPException(status_code=400, detail="Payload must include documents")

    if not getattr(solve, "_cache_cleared_after_start", False):
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("DELETE FROM cache")
            conn.commit()
            cur.close()
            conn.close()
            solve._cache_cleared_after_start = True
            logger.info("Cleared cache table on first solve after startup")
        except Exception:
            logger.exception("Failed to clear cache table on first solve after startup")

    cache_documents = []
    for document in documents:
        cache_documents.append(
            {
                "filename": document.get("filename"),
                "pdf_url": document.get("pdf_url"),
                "ocr_text": document.get("ocr_text"),
            }
        )

    cache_payload = json.dumps(
        cache_documents,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    cache_key = "challenge2:v1:" + hashlib.sha256(
        cache_payload.encode("utf-8")
    ).hexdigest()

    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT value FROM cache WHERE key = %s", (cache_key,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row is not None:
            logger.info("Cache hit for %s", cache_key)
            return row[0]
        logger.info("Cache miss for %s", cache_key)
    except Exception:
        logger.exception("Cache read failed for %s", cache_key)

    def detect_endorsement_number(document: dict):
        patterns = [
            r"dodatek\s*(?:č\.?|cislo|číslo|c\.?)?\s*(\d+)",
            r"endorsement\s*(?:no\.?|number)?\s*(\d+)",
        ]
        haystacks = [
            document.get("filename") or "",
            document.get("ocr_text") or "",
        ]
        for text in haystacks:
            lowered = text.lower()
            for pattern in patterns:
                match = re.search(pattern, lowered, re.IGNORECASE)
                if match:
                    return match.group(1)
        return None

    def build_nullable_schema(base_type: str):
        return {"type": [base_type, "null"]}

    def build_nullable_enum(values):
        return {"type": ["string", "null"], "enum": values + [None]}

    structured_documents = []
    endorsement_numbers = []
    for index, document in enumerate(documents, start=1):
        endorsement_number = detect_endorsement_number(document)
        if endorsement_number is not None:
            endorsement_numbers.append(int(endorsement_number))
        structured_documents.append(
            {
                "index": index,
                "filename": document.get("filename"),
                "pdf_url": document.get("pdf_url"),
                "endorsementNumber": endorsement_number,
                "ocrText": document.get("ocr_text") or "",
            }
        )

    latest_endorsement_number = (
        str(max(endorsement_numbers)) if endorsement_numbers else None
    )
    expected_keys = [
        "contractNumber",
        "insurerName",
        "state",
        "assetType",
        "concludedAs",
        "contractRegime",
        "startAt",
        "endAt",
        "concludedAt",
        "installmentNumberPerInsurancePeriod",
        "insurancePeriodMonths",
        "premium",
        "actionOnInsurancePeriodTermination",
        "noticePeriod",
        "regPlate",
        "latestEndorsementNumber",
        "note",
        "annualPremiumTotal",
        "liabilityLimitHealth",
        "liabilityLimitProperty",
        "insuranceScope",
    ]

    response_schema = {
        "type": "object",
        "required": expected_keys,
        "properties": {
            "contractNumber": build_nullable_schema("string"),
            "insurerName": build_nullable_schema("string"),
            "state": {"type": "string", "enum": ["draft", "accepted", "cancelled"]},
            "assetType": {"type": "string", "enum": ["other", "vehicle"]},
            "concludedAs": {"type": "string", "enum": ["agent", "broker"]},
            "contractRegime": {
                "type": "string",
                "enum": ["individual", "frame", "fleet", "coinsurance"],
            },
            "startAt": build_nullable_schema("string"),
            "endAt": build_nullable_schema("string"),
            "concludedAt": build_nullable_schema("string"),
            "installmentNumberPerInsurancePeriod": build_nullable_schema("integer"),
            "insurancePeriodMonths": build_nullable_schema("integer"),
            "premium": {
                "type": "object",
                "required": ["currency", "isCollection"],
                "properties": {
                    "currency": build_nullable_schema("string"),
                    "isCollection": {"type": "boolean"},
                },
            },
            "actionOnInsurancePeriodTermination": build_nullable_enum(
                ["auto-renewal", "policy-termination"]
            ),
            "noticePeriod": build_nullable_schema("string"),
            "regPlate": build_nullable_schema("string"),
            "latestEndorsementNumber": build_nullable_schema("string"),
            "note": build_nullable_schema("string"),
            "annualPremiumTotal": build_nullable_schema("integer"),
            "liabilityLimitHealth": build_nullable_schema("integer"),
            "liabilityLimitProperty": build_nullable_schema("integer"),
            "insuranceScope": build_nullable_schema("string"),
        },
    }

    prompt = f"""
You extract structured CRM data from OCR text of Czech insurance contracts.

Business rules:
- Input: The request contains a "documents" array. Each document may include filename, ocr_text, and optional pdf_url.
- Handle a single document, multiple documents, and documents without amendments.
- Document type detection: base contract contains the phrase "POJISTNÁ SMLOUVA"; amendment contains "DODATEK".
- Process all documents together as one contract file.
- A main contract may be followed by zero or more amendments/addenda.
- Amendments override earlier values from the main contract or earlier amendments.
- Prefer the value from the highest-numbered amendment when multiple amendments change the same field.
- latestEndorsementNumber should reflect the highest amendment number present in the documents; return null if there is no amendment. Return it as a STRING.
- Contract identity fields:
  - contractNumber example pattern: "POJISTNÁ SMLOUVA č. XXXX"
  - insurerName example pattern: "Pojistitel: XXXX"
- Dates must use DD.MM.YYYY exactly (zero-padded).
- If OCR text contains "doba neurčitá", endAt must be null.
- Installment mapping:
  - ročně = 1
  - pololetně = 2
  - čtvrtletně = 4
  - měsíčně = 12
- Insurance period length: extract from phrases like "Pojistné období: X měsíců" and return integer months.
- premium.currency must be lowercase ISO-style like czk or eur.
- premium.isCollection is true only if the documents explicitly state the broker collects the premium; otherwise return false.
- noticePeriod must be a lowercase hyphenated duration string, not natural language.
- Examples for noticePeriod: "six-weeks", "two-months", "one-month", "eight-weeks".
- Contract termination behavior: if text indicates automatic renewal (e.g., "Po uplynutí pojistného období se smlouva automaticky prodlužuje"),
  set actionOnInsurancePeriodTermination to "auto-renewal".
- Default constant fields (always return these exact values):
  - state = "accepted"
  - assetType = "other"
  - concludedAs = "broker"
  - contractRegime = "individual"
- Optional fields that may be missing must be null (never empty string):
  - regPlate, note, annualPremiumTotal, liabilityLimitHealth, liabilityLimitProperty, insuranceScope
- Null handling rules: use null instead of empty strings, 0, or placeholders like "unknown".
- Common scoring failure avoidance:
  - Do not use uppercase currency values.
  - Do not return dates in other formats.
  - Do not return latestEndorsementNumber as a number.
- Use only the allowed enum values from the schema.
- Do not invent values.
- Output must include ALL fields from the schema with correct types.

Document bundle:
{json.dumps(structured_documents, ensure_ascii=False)}
""".strip()

    try:
        response = gemini.generate(
            prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=response_schema,
            ),
        )
    except Exception:
        logger.exception("Gemini extraction request failed")
        raise HTTPException(status_code=502, detail="Gemini extraction failed")

    try:
        parsed = getattr(response, "parsed", None)
        if parsed is None:
            parsed = json.loads(response.text)
    except Exception:
        logger.exception("Failed to parse Gemini structured output: %r", getattr(response, "text", None))
        raise HTTPException(status_code=502, detail="Gemini returned invalid structured output")

    if not isinstance(parsed, dict):
        logger.error("Gemini structured output is not an object: %r", parsed)
        raise HTTPException(status_code=502, detail="Gemini returned non-object structured output")

    missing_keys = [key for key in expected_keys if key not in parsed]
    if missing_keys:
        logger.error("Gemini structured output is missing keys: %s; payload=%r", missing_keys, parsed)
        raise HTTPException(status_code=502, detail="Gemini returned incomplete structured output")

    premium = parsed.get("premium")
    if not isinstance(premium, dict):
        logger.error("Gemini structured output has invalid premium object: %r", premium)
        raise HTTPException(status_code=502, detail="Gemini returned invalid premium object")

    missing_premium_keys = [key for key in ("currency", "isCollection") if key not in premium]
    if missing_premium_keys:
        logger.error(
            "Gemini structured output is missing premium keys: %s; premium=%r",
            missing_premium_keys,
            premium,
        )
        raise HTTPException(status_code=502, detail="Gemini returned incomplete premium object")

    parsed["latestEndorsementNumber"] = latest_endorsement_number
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO cache (key, value)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value
            """,
            (cache_key, json.dumps(parsed, ensure_ascii=False)),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        logger.exception("Cache write failed for %s", cache_key)

    return parsed


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
