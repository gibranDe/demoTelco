
# app_flask.py (con logs para debug)
import os, re, json, time, uuid, requests, logging, traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError
from voyageai import Client as Voyage

# -------------------- Logging --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("rag-kpis")

def _rid():
    return uuid.uuid4().hex[:8]

# -------------------- Config ---------------------
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME     = os.getenv("DB_NAME", "demo_telmex")
COLL_INC    = os.getenv("COLL_INC", "incidencias")
COLL_KPIS   = os.getenv("COLL_KPIS", "kpis_cache")
VECTOR_IDX  = os.getenv("ATLAS_VECTOR_INDEX", "vector_index")
VOYAGE_KEY  = os.getenv("VOYAGE_API_KEY")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "voyage-3")
LLM_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI requerido")

log.info(f"Boot | db={DB_NAME} coll_inc={COLL_INC} coll_kpis={COLL_KPIS} vindex={VECTOR_IDX} emb_model={EMBED_MODEL} llm_model={LLM_MODEL}")
if not VOYAGE_KEY:
    log.warning("VOYAGE_API_KEY no configurada (no habrá embeddings de consulta)")
if not OPENAI_KEY:
    log.warning("OPENAI_API_KEY no configurada (no habrá síntesis LLM)")

client = MongoClient(MONGODB_URI, appname="rag-kpis")
db     = client[DB_NAME]
inc    = db[COLL_INC]
kpis   = db[COLL_KPIS]
voy    = Voyage(api_key=VOYAGE_KEY) if VOYAGE_KEY else None

app = Flask(__name__)
CORS(app)

# ------------ utils ------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _dt_days_ago(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)

def DATE_EXPR(field_name: str):
    f = f"${field_name}"
    f_str = {"$toString": f}

    ampm_parse = {
        "$let": {
            "vars": {
                "m": {
                    "$regexFind": {
                        "input": f_str,
                        "regex": r"^(\d{2})\/(\d{2})\/(\d{4}) (\d{2}):(\d{2}):(\d{2})\.(\d{6}) (AM|PM) ([\+\-]\d{2}:\d{2})$"
                    }
                }
            },
            "in": {
                "$cond": [
                    {"$ne": ["$$m", None]},
                    {
                        "$let": {
                            "vars": {
                                "mm":   {"$toInt": {"$arrayElemAt": ["$$m.captures", 0]}},
                                "dd":   {"$toInt": {"$arrayElemAt": ["$$m.captures", 1]}},
                                "yyyy": {"$toInt": {"$arrayElemAt": ["$$m.captures", 2]}},
                                "h12":  {"$toInt": {"$arrayElemAt": ["$$m.captures", 3]}},
                                "mi":   {"$toInt": {"$arrayElemAt": ["$$m.captures", 4]}},
                                "ss":   {"$toInt": {"$arrayElemAt": ["$$m.captures", 5]}},
                                "usec": {"$arrayElemAt": ["$$m.captures", 6]},
                                "ampm": {"$arrayElemAt": ["$$m.captures", 7]},
                                "tz":   {"$arrayElemAt": ["$$m.captures", 8]},
                            },
                            "in": {
                                "$dateFromParts": {
                                    "year":  "$$yyyy",
                                    "month": "$$mm",
                                    "day":   "$$dd",
                                    "hour": {
                                        "$let": {
                                            "vars": {
                                                "ispm": {"$eq": ["$$ampm", "PM"]},
                                                "isam": {"$eq": ["$$ampm", "AM"]},
                                            },
                                            "in": {
                                                "$cond": [
                                                    {"$and": ["$$ispm", {"$lt": ["$$h12", 12]}]},
                                                    {"$add": ["$$h12", 12]},
                                                    {
                                                        "$cond": [
                                                            {"$and": ["$$isam", {"$eq": ["$$h12", 12]}]},
                                                            0,
                                                            "$$h12",
                                                        ]
                                                    },
                                                ]
                                            },
                                        }
                                    },
                                    "minute": "$$mi",
                                    "second": "$$ss",
                                    "millisecond": { "$toInt": {"$substrBytes": ["$$usec", 0, 3]} },
                                    "timezone": "$$tz",
                                }
                            },
                        }
                    },
                    None
                ]
            }
        }
    }

    return {
        "$switch": {
            "branches": [
                {
                    "case": {
                        "$regexMatch": {
                            "input": f_str,
                            "regex": "(AM|PM)\\s+[\\+\\-]\\d{2}:\\d{2}$"
                        }
                    },
                    "then": ampm_parse
                },
                {
                    "case": { "$regexMatch": { "input": f_str, "regex": "T" } },
                    "then": { "$toDate": f }
                },
            ],
            "default": { "$toDate": f }
        }
    }

def _embed_query(text: str) -> List[float]:
    if not voy:
        raise ValueError("VOYAGE_API_KEY faltante")
    t0 = time.time()
    try:
        emb = voy.embed(texts=[text], model=EMBED_MODEL, input_type="query").embeddings[0]
        log.info(f"embed.query ok | len={len(emb)} | ms={(time.time()-t0)*1000:.0f}")
        return emb
    except Exception as e:
        log.error(f"embed.query error | {e}")
        raise

def _vector_search(qvec: List[float], top_k: int) -> List[Dict[str,Any]]:
    pipe = [
        {"$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": qvec,
            "numCandidates": max(top_k*5, 200),
            "limit": top_k
        }},
        {"$project": {
            "_id": 0,
            "NUMBER":1, "BRIEF_DESCRIPTION":1, "REGION":1, "LOCATION":1,
            "TELMEX_PRODUCT":1, "TYPE":1, "SEVERITY":1, "PRIORITY_CODE":1,
            "OPEN_TIME":1, "CLOSE_TIME":1, "OUTAGE":1,
            "score":{"$meta":"vectorSearchScore"}
        }}
    ]
    t0 = time.time()
    try:
        res = list(inc.aggregate(pipe, allowDiskUse=True))
        log.info(f"vsearch ok | hits={len(res)} | top_k={top_k} | ms={(time.time()-t0)*1000:.0f}")
        return res
    except PyMongoError as e:
        log.error(f"vsearch mongo error | {e}")
        log.debug("vsearch pipeline=" + json.dumps(pipe)[:1000])
        return []
    except Exception as e:
        log.error(f"vsearch error | {e}")
        return []

def _save_metric(metric:str, params:Dict[str,Any], items:List[Dict[str,Any]]):
    try:
        kpis.insert_one({
            "metric": metric,
            "params": params,
            "items": items,
            "ts": datetime.now(timezone.utc)
        })
        log.info(f"kpi.save ok | metric={metric} items={len(items)} params={params}")
    except Exception as e:
        log.error(f"kpi.save error | metric={metric} | {e}")

def _latest_metric(metric:str, params:Dict[str,Any]) -> Optional[Dict[str,Any]]:
    try:
        doc = kpis.find_one({"metric":metric, "params":params}, sort=[("ts", DESCENDING)])
        log.info(f"kpi.latest | metric={metric} cache={'hit' if doc else 'miss'} params={params}")
        return doc
    except Exception as e:
        log.error(f"kpi.latest error | {e}")
        return None

def _call_openai_chat(system: str, user: str) -> str:
    if not OPENAI_KEY:
        return "(OPENAI_API_KEY faltante — no puedo sintetizar)"
    t0 = time.time()
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type":"application/json"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role":"system","content":system},{"role":"user","content":user}],
                "temperature": 0.2,
                "max_tokens": 800
            },
            timeout=60
        )
        r.raise_for_status()
        j = r.json()
        log.info(f"openai.chat ok | model={LLM_MODEL} | ms={(time.time()-t0)*1000:.0f}")
        return j["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        log.error(f"openai.chat http error | {e} | body={getattr(e, 'response', None) and getattr(e.response,'text', '')}")
        return "(error llamando a OpenAI)"
    except Exception as e:
        log.error(f"openai.chat error | {e}")
        return "(error llamando a OpenAI)"

# ---------- KPI Pipelines ----------
def kpi_severity_ttr(days:int, topn:int):
    log.info(f"kpi.run severity_ttr | days={days} topn={topn}")
    since = _dt_days_ago(days)
    pipe = [
        {"$addFields": { "open_dt": DATE_EXPR("OPEN_TIME"), "close_dt": DATE_EXPR("CLOSE_TIME") }},
        {"$match": {"$expr": {"$gte": ["$open_dt", since]}}},
        {"$addFields":{
            "ttr_h": { "$divide": [ { "$subtract": [ "$close_dt", "$open_dt" ] }, 1000*60*60 ] },
            "sev_num": { "$toInt": { "$ifNull": [ "$SEVERITY", 0 ] } }
        }},
        {"$group":{
            "_id":"$TELMEX_PRODUCT",
            "incidencias":{"$sum":1},
            "criticas":{"$sum":{"$cond":[{"$lte":["$sev_num",2]},1,0]}},
            "ttr_avg_h":{"$avg":"$ttr_h"},
            "ttr_p50_h":{"$percentile":{"input":"$ttr_h","method":"approximate","p":[0.5]}},
            "ttr_p90_h":{"$percentile":{"input":"$ttr_h","method":"approximate","p":[0.9]}}
        }},
        {"$addFields":{
            "tasa_criticas": { "$cond": [ { "$gt": [ "$incidencias", 0 ] }, { "$divide": [ "$criticas", "$incidencias" ] }, 0 ] }
        }},
        {"$sort":{"tasa_criticas":-1,"incidencias":-1}},
        {"$limit": topn}
    ]
    t0=time.time()
    try:
        out = list(inc.aggregate(pipe, allowDiskUse=True))
        log.info(f"kpi.ok severity_ttr | items={len(out)} | ms={(time.time()-t0)*1000:.0f}")
        return out
    except Exception as e:
        log.error(f"kpi.err severity_ttr | {e}")
        log.debug("pipe=" + json.dumps(pipe, default=str)[:1500])
        return []

def kpi_resolution_effectiveness_fibra_region(region_regex:str, days:int, topn:int):
    log.info(f"kpi.run resolution_fibra_region | region={region_regex} days={days} topn={topn}")
    since = _dt_days_ago(days)
    pipe = [
        {"$addFields": { "open_dt": DATE_EXPR("OPEN_TIME"), "close_dt": DATE_EXPR("CLOSE_TIME") }},
        {"$match":{
            "$expr": { "$gte": ["$open_dt", since] },
            "$or":[
                {"TELMEX_PRODUCT":{"$regex":"fibra|fiber","$options":"i"}},
                {"TYPE":{"$regex":"fibra|fiber","$options":"i"}},
                {"BRIEF_DESCRIPTION":{"$regex":"fibra|fiber","$options":"i"}}
            ],
            "REGION":{"$regex":region_regex, "$options":"i"}
        }},
        {"$addFields":{
            "ttr_h": { "$divide": [ { "$subtract": [ "$close_dt", "$open_dt" ] }, 1000*60*60 ] }
        }},
        {"$group":{
            "_id":"$RESOLUTION_CODE",
            "incidencias":{"$sum":1},
            "sev12":{"$sum":{"$cond":[{"$in":["$SEVERITY",[1,2,"1","2"]]},1,0]}},
            "ttr_h":{"$avg":"$ttr_h"}
        }},
        {"$sort":{"incidencias":-1}},
        {"$limit": topn}
    ]
    t0=time.time()
    try:
        out = list(inc.aggregate(pipe, allowDiskUse=True))
        log.info(f"kpi.ok resolution_fibra_region | items={len(out)} | ms={(time.time()-t0)*1000:.0f}")
        return out
    except Exception as e:
        log.error(f"kpi.err resolution_fibra_region | {e}")
        log.debug("pipe=" + json.dumps(pipe, default=str)[:1500])
        return []

def kpi_hotspots_region(days:int, topn:int):
    log.info(f"kpi.run hotspots_region | days={days} topn={topn}")
    since = _dt_days_ago(days)
    pipe = [
        {"$addFields": { "open_dt": DATE_EXPR("OPEN_TIME") }},
        {"$match":{"$expr":{"$gte":["$open_dt", since]}}},
        {"$group":{
            "_id":"$REGION",
            "incidencias":{"$sum":1},
            "criticas":{"$sum":{"$cond":[{"$in":["$SEVERITY",[1,2,"1","2"]]},1,0]}},
            "avg_outage_h":{"$avg":{"$toDouble":{"$ifNull":["$OUTAGE",0]}}}
        }},
        {"$sort":{"incidencias":-1}},
        {"$limit": topn}
    ]
    t0=time.time()
    try:
        out = list(inc.aggregate(pipe, allowDiskUse=True))
        log.info(f"kpi.ok hotspots_region | items={len(out)} | ms={(time.time()-t0)*1000:.0f}")
        return out
    except Exception as e:
        log.error(f"kpi.err hotspots_region | {e}")
        log.debug("pipe=" + json.dumps(pipe, default=str)[:1500])
        return []

def kpi_estabilidad_segmento(days:int):
    log.info(f"kpi.run estabilidad_segmento | days={days}")
    since = _dt_days_ago(days)
    pipe = [
        {"$addFields": { "open_dt": DATE_EXPR("OPEN_TIME"), "close_dt": DATE_EXPR("CLOSE_TIME") }},
        {"$match":{"$expr":{"$gte":["$open_dt", since]}}},
        {"$addFields":{
            "segmento":{"$cond":[
                {"$regexMatch":{"input":{"$ifNull":["$TELMEX_PRODUCT",""]},"regex":"residencial|home|hogar","options":"i"}},
                "residencial","empresarial"
            ]},
            "ttr_h": { "$divide": [ { "$subtract": [ "$close_dt", "$open_dt" ] }, 1000*60*60 ] }
        }},
        {"$group":{
            "_id":"$segmento",
            "incidencias":{"$sum":1},
            "criticas":{"$sum":{"$cond":[{"$in":["$SEVERITY",[1,2,"1","2"]]},1,0]}},
            "ttr_avg_h":{"$avg":"$ttr_h"}
        }},
        {"$sort":{"_id":1}}
    ]
    t0=time.time()
    try:
        out = list(inc.aggregate(pipe, allowDiskUse=True))
        log.info(f"kpi.ok estabilidad_segmento | items={len(out)} | ms={(time.time()-t0)*1000:.0f}")
        return out
    except Exception as e:
        log.error(f"kpi.err estabilidad_segmento | {e}")
        log.debug("pipe=" + json.dumps(pipe, default=str)[:1500])
        return []

def kpi_top_locations_outage(days:int, topn:int):
    log.info(f"kpi.run top_locations_outage | days={days} topn={topn}")
    since = _dt_days_ago(days)
    pipe = [
        {"$addFields": { "open_dt": DATE_EXPR("OPEN_TIME") }},
        {"$match":{"$expr":{"$gte":["$open_dt", since]}}},
        {"$group":{
            "_id":"$LOCATION",
            "incidencias":{"$sum":1},
            "outage_avg_h":{"$avg":{"$toDouble":{"$ifNull":["$OUTAGE",0]}}},
            "prod":{"$addToSet":"$TELMEX_PRODUCT"}
        }},
        {"$sort":{"outage_avg_h":-1,"incidencias":-1}},
        {"$limit": topn}
    ]
    t0=time.time()
    try:
        out = list(inc.aggregate(pipe, allowDiskUse=True))
        log.info(f"kpi.ok top_locations_outage | items={len(out)} | ms={(time.time()-t0)*1000:.0f}")
        return out
    except Exception as e:
        log.error(f"kpi.err top_locations_outage | {e}")
        log.debug("pipe=" + json.dumps(pipe, default=str)[:1500])
        return []

def rebuild_kpis(days:int, topn:int) -> Dict[str,Any]:
    log.info(f"kpis.rebuild | days={days} topn={topn}")
    out = {}
    try:
        items = kpi_severity_ttr(days, topn)
        _save_metric("severity_ttr", {"days":days,"topn":topn}, items)
        out["severity_ttr"] = items

        items = kpi_resolution_effectiveness_fibra_region("norte|noreste|noroeste", days, topn)
        _save_metric("resolution_fibra_norte", {"days":days,"topn":topn}, items)
        out["resolution_fibra_norte"] = items

        items = kpi_hotspots_region(days, topn)
        _save_metric("hotspots_region", {"days":days,"topn":topn}, items)
        out["hotspots_region"] = items

        items = kpi_estabilidad_segmento(days)
        _save_metric("estabilidad_segmento", {"days":days}, items)
        out["estabilidad_segmento"] = items

        items = kpi_top_locations_outage(days, topn)
        _save_metric("top_locations_outage", {"days":days,"topn":topn}, items)
        out["top_locations_outage"] = items
    except Exception as e:
        log.error(f"kpis.rebuild error | {e}")
    return out

# ---------- intent map ----------
_INTENT_MAP = [
    (r"\b(monterrey|cdmx|ciudad de mexico|guadalajara|merida|norte|noreste|noroeste)\b", ["hotspots_region","top_locations_outage"]),
    (r"\b(severidad|criticas|prioridad|ttr|tiempo de resolucion)\b", ["severity_ttr"]),
    (r"\b(fibra|fiber)\b", ["resolution_fibra_norte"]),
    (r"\b(residencial|empresarial)\b", ["estabilidad_segmento"]),
]

def pick_kpis_for_question(q:str, days:int, topn:int) -> Dict[str,Any]:
    log.info(f"kpi.pick | q='{q[:80]}' days={days} topn={topn}")
    ql = q.lower() if q else ""
    chosen = set()
    for pat, names in _INTENT_MAP:
        if re.search(pat, ql):
            chosen.update(names)
    if not chosen:
        chosen = {"severity_ttr","hotspots_region"}
    res = {}
    for name in chosen:
        base = {"days":days}
        if name in {"severity_ttr","hotspots_region","top_locations_outage","resolution_fibra_norte"}:
            base["topn"]=topn
        doc = _latest_metric(name, base)
        if not doc:
            log.info(f"kpi.pick miss→compute | {name}")
            compute = {
                "severity_ttr": lambda: kpi_severity_ttr(days, topn),
                "hotspots_region": lambda: kpi_hotspots_region(days, topn),
                "top_locations_outage": lambda: kpi_top_locations_outage(days, topn),
                "resolution_fibra_norte": lambda: kpi_resolution_effectiveness_fibra_region("norte|noreste|noroeste", days, topn),
                "estabilidad_segmento": lambda: kpi_estabilidad_segmento(days),
            }[name]()
            _save_metric(name, base, compute)
            res[name] = compute
        else:
            res[name] = doc["items"]
    return res

# ---------- synthesis ----------
def synthesize(question:str, hits:List[Dict[str,Any]], kpi_ctx:Dict[str,Any]) -> str:
    log.info(f"synth.start | qlen={len(question)} hits={len(hits)} kpi_sets={len(kpi_ctx)}")
    lines = []
    for h in hits[:10]:
        lines.append(f"[{h.get('NUMBER','?')}] {h.get('REGION','')} • {h.get('TELMEX_PRODUCT','')} • sev={h.get('SEVERITY','')} — {h.get('BRIEF_DESCRIPTION','')}")
    context_hits = "\n".join(lines) if lines else "(sin incidentes relevantes)"

    kpi_parts = []
    for name, items in kpi_ctx.items():
        kpi_parts.append(f"\n== {name} ==")
        for it in items[:10]:
            kpi_parts.append(json.dumps(it, default=str))
    context_kpis = "\n".join(kpi_parts) if kpi_parts else "(sin KPIs)"

    system = (
        "Eres analista NOC Telco. Usa incidencias (cítalas por número) y KPIs (promedios, percentiles, tasas) "
        "para responder con hallazgos, interpretación y acciones priorizadas. Cita con [#IM...] cuando refieras un caso."
    )
    user = f"Pregunta:\n{question}\n\nIncidencias relevantes:\n{context_hits}\n\nKPIs:\n{context_kpis}\n\nEntrega respuesta ejecutiva y luego pasos técnicos."
    ans = _call_openai_chat(system, user)
    log.info("synth.done")
    return ans

# ----------- endpoints -----------
@app.get("/")
def root():
    rid = _rid()
    log.info(f"{rid} GET /")
    if os.path.exists("ui.html"):
        log.info(f"{rid} serve ui.html")
        return send_file("ui.html")
    return jsonify({"ok": True, "ts": _now_iso()})

@app.get("/health")
def health():
    rid = _rid()
    log.info(f"{rid} GET /health")
    try:
        inc.estimated_document_count()
        kpis.create_index([("metric", ASCENDING), ("params", ASCENDING), ("ts", DESCENDING)])
        log.info(f"{rid} health ok")
        return jsonify({"ok": True, "ts": _now_iso()})
    except Exception as e:
        log.error(f"{rid} health error | {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/metrics/rebuild")
def metrics_rebuild():
    rid = _rid()
    t0 = time.time()
    data = request.get_json(silent=True) or {}
    days = int(data.get("days", 30))
    topn = int(data.get("topn", 10))
    log.info(f"{rid} POST /metrics/rebuild | days={days} topn={topn}")
    out = rebuild_kpis(days, topn)
    log.info(f"{rid} rebuild done | metrics={list(out.keys())} | ms={(time.time()-t0)*1000:.0f}")
    return jsonify({"ok": True, "ts": _now_iso(), "metrics": list(out.keys())})

@app.get("/metrics/latest")
def metrics_latest():
    rid = _rid()
    t0 = time.time()
    days = int(request.args.get("days", 30))
    topn = int(request.args.get("topn", 10))
    log.info(f"{rid} GET /metrics/latest | days={days} topn={topn}")
    out = pick_kpis_for_question("", days, topn)
    log.info(f"{rid} latest done | sets={len(out)} | ms={(time.time()-t0)*1000:.0f}")
    return jsonify({"ok": True, "days": days, "topn": topn, "data": out})

@app.post("/ask")
def ask():
    rid = _rid()
    t0 = time.time()
    try:
        body = request.get_json(force=True)
    except Exception:
        log.error(f"{rid} /ask bad json")
        return jsonify({"error":"invalid json"}), 400

    q = (body.get("query") or body.get("question") or "").strip()
    top_k = int(body.get("top_k", 20))
    days  = int(body.get("days", 30))
    log.info(f"{rid} POST /ask | qlen={len(q)} top_k={top_k} days={days}")

    if not q:
        return jsonify({"error":"question/query requerido"}), 400

    try:
        qvec = _embed_query(q)
        hits = _vector_search(qvec, max(1, top_k))
        kctx = pick_kpis_for_question(q, days, 10)
        ans = synthesize(q, hits, kctx)
        log.info(f"{rid} ask done | hits={len(hits)} | ms={(time.time()-t0)*1000:.0f}")
        return jsonify({
            "answer": ans,
            "documents": hits,
            "kpis": kctx,
            "model": LLM_MODEL
        })
    except Exception as e:
        log.error(f"{rid} ask error | {e}")
        log.debug(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Requisitos: pip install flask flask-cors pymongo python-dotenv voyageai requests
    log.info("server.start | 0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000, debug=True)