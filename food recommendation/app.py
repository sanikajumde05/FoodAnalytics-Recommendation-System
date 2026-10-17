"""Flask backend:  python app.py  ->  http://127.0.0.1:5000"""
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from pipeline import load_data, rfm_segments, build_models, Recommender, evaluate_recommender

print("Loading & cleaning data ...");  orders, rest, users, df, DEMO = load_data()
print("Segmenting customers ...");     seg = rfm_segments(df)
print("Training models (1-3 min) ..."); res, best, imp, (yt, yp) = build_models(df)
print("Building & evaluating recommender ..."); rec = Recommender(orders, rest); ev = evaluate_recommender(rec)
Q = df.attrs["quality"]; print(f"Ready. Source: {Q['source']} | {len(df):,} clean orders | best model: {best}")

def top(s, n): return {str(k): round(float(v), 2) for k, v in s.nlargest(n).items()}
mo = df.dropna(subset=["order_date"]).groupby(df.order_date.dt.to_period("M")).agg(o=("sales_amount", "size"), r=("sales_amount", "sum"))
sel = np.random.default_rng(1).choice(len(yt), min(150, len(yt)), replace=False)
ANALYTICS = dict(
    demo=DEMO, quality=Q,
    kpi=dict(orders=len(df), revenue=round(float(df.sales_amount.sum())), avg_order_value=round(float(df.sales_amount.mean()), 1),
             customers=int(df.user_id.nunique()), restaurants=int(df.restaurant_id.nunique())),
    monthly_orders={str(k): int(v) for k, v in mo.o.items()}, monthly_revenue={str(k): round(float(v)) for k, v in mo.r.items()},
    top_restaurants=top(df.groupby("restaurant_name").size(), 10), top_cuisines=top(df.groupby("cuisine").size(), 8),
    top_states=top(df.groupby("state").size(), 8),
    dow={d: int((df.dow == i).sum()) for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])},
    breakdown={c: top(df.groupby(c).size(), 8) for c in df.attrs["user_cats"][:2]},
    segments=seg.attrs["profile"].to_dict("records"), k_scores=seg.attrs["k_scores"], k=seg.attrs["k"],
    models=res.to_dict("records"), best_model=best, importance={k: round(float(v), 4) for k, v in imp.head(8).items()},
    scatter=[dict(x=round(float(yt[i]), 1), y=round(float(yp[i]), 1)) for i in sel], recommender_eval=ev)
USERS = [str(u) for u in rec.u_idx if rec.M[rec.u_idx[u]].nnz >= 4][:40]
_loc = rest[rest.restaurant_id.isin(rec.r_ids)].groupby("state").city.agg(lambda s: sorted(set(s)))
FILTERS = dict(cuisines=sorted(set(rec.cuis)), states=sorted(_loc.index), cities={k: v for k, v in _loc.items()})

app = Flask(__name__, static_folder="static")
@app.after_request
def cors(r): r.headers["Access-Control-Allow-Origin"] = "*"; return r
@app.errorhandler(Exception)
def fail(e): return jsonify(error=f"{type(e).__name__}: {e}"), getattr(e, "code", 500)
@app.route("/")
def home(): return send_from_directory("static", "index.html")
@app.route("/api/health")
def health(): return jsonify(status="ok", source=Q["source"], demo=DEMO, orders=len(df))
@app.route("/api/analytics")
def analytics(): return jsonify(ANALYTICS)
@app.route("/api/users")
def api_users(): return jsonify(USERS)
@app.route("/api/filters")
def filters(): return jsonify(FILTERS)
@app.route("/api/cuisines")
def cuisines(): return jsonify(FILTERS["cuisines"])
@app.route("/api/recommend/<uid>")
def recommend(uid):
    uid = uid.strip(); a = request.args
    out, method = rec.recommend(uid, max(1, min(a.get("n", 10, type=int), 30)), a.get("cuisine") or None, a.get("state") or None, a.get("city") or None)
    return jsonify(method=method, segment=seg.segment.get(uid, "New user"), known_user=uid in rec.u_idx, recommendations=out)
if __name__ == "__main__": app.run(port=5000)
