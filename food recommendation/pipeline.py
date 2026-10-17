"""Shared data-science pipeline: cleaning + quality report, RFM/K-Means segments, order-value models, hybrid recommender."""
import os, glob, math, itertools
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, silhouette_score

# ------------------------------------------------------------------ DEMO (tests only; disabled unless FOOD_DEMO=1)
def make_demo(seed=7):
    rng = np.random.default_rng(seed)
    cu = ["North Indian", "South Indian", "Chinese", "Italian", "Fast Food", "Biryani", "Desserts", "Cafe", "Mexican", "Thai"]
    R, U, N = 300, 1500, 40000
    cost = rng.integers(100, 900, R); loc = [("Pune", "Maharashtra"), ("Mumbai", "Maharashtra"), ("Delhi", "Delhi"), ("Bangalore", "Karnataka")]
    rating = np.round(rng.uniform(3, 4.9, R), 1).astype(object); rating[rng.choice(R, 30, replace=False)] = "--"
    rest = pd.DataFrame({"id": np.arange(1, R + 1), "name": [f"Restaurant {i}" for i in range(1, R + 1)],
                         "city": ["%s,%s" % loc[i] for i in rng.integers(0, 4, R)], "rating": rating,
                         "rating_count": rng.integers(20, 2000, R).astype(str), "cost": ["₹ " + str(c) for c in cost], "cuisine": rng.choice(cu, R)})
    users = pd.DataFrame({"user_id": np.arange(1, U + 1), "Age": rng.integers(18, 60, U), "Gender": rng.choice(["Male", "Female"], U),
                          "Occupation": rng.choice(["Student", "Employee", "Self Employeed", "House wife"], U),
                          "Monthly Income": rng.choice(["No Income", "Below Rs.10000", "10001 to 25000", "25001 to 50000", "More than 50000"], U)})
    fav = [rng.choice(10, 2, replace=False) for _ in range(U)]; rc = pd.Categorical(rest.cuisine, categories=cu).codes
    pop = rng.dirichlet(np.ones(R)); uid = rng.integers(0, U, N); rid = np.empty(N, int)
    for k, u in enumerate(uid):
        w = pop * (1 + 8 * np.isin(rc, fav[u])); rid[k] = rng.choice(R, p=w / w.sum())
    qty = rng.integers(1, 5, N); amt = (qty * cost[rid] * rng.uniform(.9, 1.1, N)).round(2)
    amt[rng.choice(N, 20, replace=False)] *= 40; amt = amt.astype(object); amt[rng.choice(N, 100, replace=False)] = None
    orders = pd.DataFrame({"Order Date": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 365, N), "D"),
                           "Sales Qty": qty, "Sales Amount": amt, "Currency": "INR", "User ID": uid + 1, "r_id": rid + 1})
    orders.loc[:50, "r_id"] = R + 999                                   # orphan restaurant ids
    return pd.concat([orders, orders.sample(200, random_state=1)]), rest, users   # + duplicate rows

# ------------------------------------------------------------------ LOAD / CLEAN
ALIASES = dict(
    orders=({"user_id": ["user_id", "userid", "customer_id"], "restaurant_id": ["restaurant_id", "r_id", "rest_id", "restaurantid"],
             "sales_amount": ["sales_amount", "amount", "order_value", "total", "price"],
             "sales_quantity": ["sales_quantity", "sales_qty", "quantity", "qty"], "order_date": ["order_date", "date", "order_datetime"]},
            {"sales_amount": ["amount"], "sales_quantity": ["qty", "quant"], "order_date": ["date"]}),
    restaurant=({"restaurant_id": ["restaurant_id", "id", "r_id"], "restaurant_name": ["restaurant_name", "name"]}, {}),
    users=({"user_id": ["user_id", "id", "userid", "customer_id"]}, {}))

def _rename(df, tbl, required, log):
    mapping, hints = ALIASES[tbl]
    for target, cands in mapping.items():
        hit = next((c for c in cands if c in df.columns), None)
        if hit is None and target in hints: hit = next((c for c in df.columns if any(h in c for h in hints[target])), None)
        if hit and hit != target: df = df.rename(columns={hit: target}); log.append(f"{tbl}: '{hit}' -> '{target}'")
    miss = [c for c in required if c not in df.columns]
    if miss: raise SystemExit(f"ERROR [{tbl}.csv] required columns {miss} not found. Columns present: {list(df.columns)}")
    return df

_idstr = lambda s: s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
_num = lambda s: pd.to_numeric(s.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
_txt = lambda s, d="Unknown": s.astype(str).str.strip().replace({"": d, "nan": d, "None": d, "NaN": d})

def load_data(folder="data"):
    """Returns orders, restaurants, users, merged df, demo_flag. df.attrs['quality'] holds the cleaning report."""
    files = {k: [p for p in glob.glob(f"{folder}/*.csv") if k in os.path.basename(p).lower()] for k in ("order", "restaurant", "user")}
    demo = not all(files.values())
    if demo and os.environ.get("FOOD_DEMO") != "1":
        raise SystemExit("ERROR: real dataset not found. Put the Kaggle CSVs (orders, restaurant, users) in the 'data/' folder. "
                         "Synthetic fallback is disabled so results always come from real data.")
    o, r, u = make_demo() if demo else (pd.read_csv(files[k][0]) for k in ("order", "restaurant", "user"))
    q = dict(source="SYNTHETIC DEMO (testing only)" if demo else ", ".join(os.path.basename(files[k][0]) for k in files),
             raw_orders=len(o), raw_restaurants=len(r), raw_users=len(u)); log = []
    for d in (o, r, u): d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
    o = _rename(o, "orders", ("user_id", "restaurant_id", "sales_amount"), log)
    r = _rename(r, "restaurant", ("restaurant_id",), log); u = _rename(u, "users", ("user_id",), log)
    for d, c in ((o, "user_id"), (o, "restaurant_id"), (r, "restaurant_id"), (u, "user_id")): d[c] = _idstr(d[c])
    # ---- orders
    n = len(o); o = o.drop_duplicates(); q["duplicate_orders_removed"] = n - len(o)
    o["sales_amount"] = pd.to_numeric(o["sales_amount"], errors="coerce")
    q["orders_missing_amount_removed"] = int(o.sales_amount.isna().sum()); o = o.dropna(subset=["sales_amount"])
    q["orders_nonpositive_amount_removed"] = int((o.sales_amount <= 0).sum()); o = o[o.sales_amount > 0].copy()
    o["order_date"] = pd.to_datetime(o["order_date"], errors="coerce") if "order_date" in o else pd.NaT
    q["orders_invalid_date"] = int(o.order_date.isna().sum())
    if "sales_quantity" in o: o["sales_quantity"] = pd.to_numeric(o["sales_quantity"], errors="coerce").fillna(1).clip(lower=1)
    q1, q3 = o.sales_amount.quantile([.25, .75]); cap = q3 + 3 * (q3 - q1)
    if not np.isfinite(cap) or cap <= q3: cap = o.sales_amount.quantile(.995)
    q["outlier_cap_value"] = round(float(cap), 2); q["outlier_orders_capped_for_modelling"] = int((o.sales_amount > cap).sum())
    # ---- restaurants
    for c in ("rating", "cost", "rating_count"):
        r[c] = _num(r[c]) if c in r else np.nan
    n = len(r); r = r.drop_duplicates("restaurant_id"); q["duplicate_restaurants_removed"] = n - len(r)
    r["cuisine"] = _txt(r["cuisine"].astype(str).str.split(",").str[0]).str.title() if "cuisine" in r else "Unknown"
    if "state" not in r:
        sp = (r["city"].astype(str) if "city" in r else pd.Series("Unknown", index=r.index)).str.split(",", n=1, expand=True)
        r["city"] = sp[0]; r["state"] = sp[1] if sp.shape[1] > 1 else "Unknown"
    r["city"], r["state"] = _txt(r["city"]).str.title(), _txt(r["state"]).str.title()
    if "restaurant_name" not in r: r["restaurant_name"] = "Restaurant " + r.restaurant_id
    r["restaurant_name"] = _txt(r["restaurant_name"], "Unnamed")
    q["restaurant_ratings_imputed"] = int(r.rating.isna().sum()); q["restaurant_costs_imputed"] = int(r.cost.isna().sum())
    for c in ("rating", "cost"):
        r[c] = r[c].fillna(r.groupby("cuisine")[c].transform("median")).fillna(r[c].median()).fillna(0)
    r["rating_count"] = r.rating_count.fillna(0)
    # ---- users
    n = len(u); u = u.drop_duplicates("user_id").drop(columns=[c for c in ("name", "email", "password") if c in u], errors="ignore")
    q["duplicate_users_removed"] = n - len(u)
    cats = [c for c in u.columns if c != "user_id" and not pd.api.types.is_numeric_dtype(u[c]) and u[c].nunique() <= 20]
    nums = [c for c in u.columns if c != "user_id" and pd.api.types.is_numeric_dtype(u[c])]
    for c in cats: u[c] = _txt(u[c]).str.title()
    keep = ["restaurant_id", "restaurant_name", "cuisine", "city", "state", "rating", "rating_count", "cost"]
    df = o.merge(r[keep], on="restaurant_id", how="left").merge(u[["user_id"] + cats + nums], on="user_id", how="left")
    q["orders_with_unknown_restaurant"] = int(df.restaurant_name.isna().sum()); q["orders_with_unknown_user"] = int(df[(cats + nums)[0]].isna().sum()) if cats + nums else 0
    for c in ("restaurant_name", "cuisine", "city", "state"): df[c] = df[c].fillna("Unknown")
    for c in ("rating", "cost", "rating_count"): df[c] = df[c].fillna(r[c].median())
    for c in cats: df[c] = df[c].fillna("Unknown")
    for c in nums: df[c] = df[c].fillna(u[c].median())
    df["month"], df["dow"] = df.order_date.dt.month.fillna(0), df.order_date.dt.dayofweek.fillna(0)
    df["is_weekend"] = (df.dow >= 5).astype(int)
    q.update(orders_clean=len(df), restaurants_clean=len(r), users_clean=len(u), column_mapping=log or ["no renaming needed"])
    df.attrs.update(user_cats=cats, user_nums=nums, quality=q)
    return o, r, u, df, demo

# ------------------------------------------------------------------ SEGMENTATION
SEG_INFO = {"Champions": ("Frequent, high-spending, recently active", "Reward with loyalty perks / early access"),
            "Loyal Regulars": ("Order often at moderate spend", "Upsell bundles and subscriptions"),
            "Big Spenders": ("Few orders but high basket value", "Offer premium restaurants and occasion deals"),
            "Occasional Buyers": ("Low frequency and spend", "Nudge with personalised recommendations"),
            "At-Risk / Lapsed": ("Have not ordered recently", "Run win-back discounts")}

def rfm_segments(df):
    """K-Means on log-scaled Recency/Frequency/Monetary/AvgOrderValue/Variety; k chosen by silhouette; rule-based names."""
    has = df.order_date.notna().any(); snap = df.order_date.max() + pd.Timedelta(days=1) if has else None
    g = df.groupby("user_id").agg(frequency=("sales_amount", "size"), monetary=("sales_amount", "sum"),
                                  last=("order_date", "max"), variety=("restaurant_id", "nunique"))
    g["recency"] = (snap - g["last"]).dt.days.fillna(0) if has else 0.0
    g["aov"] = g.monetary / g.frequency; F = ["recency", "frequency", "monetary", "aov", "variety"]
    Z = StandardScaler().fit_transform(np.log1p(g[F])); samp = np.random.default_rng(0).choice(len(Z), min(5000, len(Z)), replace=False)
    labs, ks = {}, {}
    for k in range(3, min(7, len(g))):
        labs[k] = KMeans(k, n_init=10, random_state=42).fit_predict(Z); ks[k] = round(float(silhouette_score(Z[samp], labs[k][samp])), 4)
    k = max(ks, key=ks.get); g["cluster"] = labs[k]
    c = g.groupby("cluster")[F].mean(); rk = c.rank(pct=True); rk["recency"] = 1 - rk["recency"] + 1 / len(c)
    def name(i):
        r_, f, m, a = rk.loc[i, ["recency", "frequency", "monetary", "aov"]]
        if f >= .66 and m >= .66 and r_ >= .5: return "Champions"
        if r_ <= .34: return "At-Risk / Lapsed"
        if a >= .66 and f < .66: return "Big Spenders"
        return "Loyal Regulars" if f >= .5 else "Occasional Buyers"
    nm = {i: name(i) for i in c.index}
    for b in set(nm.values()):
        ids = [i for i in nm if nm[i] == b]
        if len(ids) > 1:
            for t, i in enumerate(sorted(ids, key=lambda x: -c.loc[x, "monetary"])): nm[i] = f"{b} {'ABCDEF'[t]}"
    g["segment"] = g.cluster.map(nm)
    tmp = df.assign(segment=df.user_id.map(g.segment)); tc = tmp.groupby("segment").cuisine.agg(lambda s: s.mode().iat[0])
    p = g.groupby("segment").agg(customers=("frequency", "size"), recency=("recency", "mean"), frequency=("frequency", "mean"),
                                 monetary=("monetary", "mean"), aov=("aov", "mean"), variety=("variety", "mean")).round(1)
    p["share_pct"] = (100 * p.customers / p.customers.sum()).round(1); p["top_cuisine"] = tc
    base = lambda s: s[:-2] if s[:-2] in SEG_INFO else s
    p["description"] = [SEG_INFO.get(base(s), ("", ""))[0] for s in p.index]; p["action"] = [SEG_INFO.get(base(s), ("", ""))[1] for s in p.index]
    g.attrs.update(profile=p.sort_values("monetary", ascending=False).reset_index(), k_scores=ks, k=int(k))
    return g.drop(columns="last")

# ------------------------------------------------------------------ ORDER-VALUE MODELS
def build_models(df, seed=42):
    """Compares v2-style setup (raw target, base features) with enhanced setup (capped log-target + train-only target encodings)."""
    cap = df.attrs["quality"]["outlier_cap_value"]; d = df.copy(); d["amt"] = d.sales_amount.clip(upper=cap)
    if len(d) > 120000: d = d.sample(120000, random_state=seed)
    num = [c for c in ["sales_quantity", "cost", "rating", "rating_count", "month", "dow", "is_weekend"] + df.attrs["user_nums"] if c in d]
    cats = ["cuisine", "state"] + df.attrs["user_cats"]; top = d.cuisine.value_counts().index[:15]
    d["cuisine"] = d.cuisine.where(d.cuisine.isin(top), "Other"); d["state"] = d.state.where(d.state.isin(d.state.value_counts().index[:10]), "Other")
    Xb = pd.concat([d[num].apply(pd.to_numeric, errors="coerce").fillna(0), pd.get_dummies(d[cats].astype(str), prefix_sep="=")], axis=1).astype(float)
    itr, ite = train_test_split(d.index, test_size=.2, random_state=seed)
    ytr_raw, ytr, yte = d.sales_amount.loc[itr], d.amt.loc[itr], d.sales_amount.loc[ite]; gm = ytr.mean()
    def enc(key, nm, m=5):                       # leave-one-out smoothed mean of train targets (no test leakage)
        g = d.loc[itr, key].to_frame().assign(y=ytr).groupby(key).y.agg(["sum", "count"])
        s, c = d.loc[itr, key].map(g["sum"]), d.loc[itr, key].map(g["count"])
        te = d.loc[ite, key]
        return ((s - ytr + m * gm) / (c - 1 + m)).rename(nm), ((te.map(g["sum"]).fillna(0) + m * gm) / (te.map(g["count"]).fillna(0) + m)).rename(nm)
    e = [enc("user_id", "user_avg_value"), enc("restaurant_id", "restaurant_avg_value")]
    Xtr_b, Xte_b = Xb.loc[itr], Xb.loc[ite]
    Xtr_e, Xte_e = pd.concat([Xtr_b] + [a for a, _ in e], axis=1), pd.concat([Xte_b] + [b for _, b in e], axis=1)
    ttr = lambda m: TransformedTargetRegressor(regressor=m, func=np.log1p, inverse_func=np.expm1)
    mk = {"Linear Regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
          "Random Forest": lambda: RandomForestRegressor(100, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=seed),
          "Gradient Boosting": lambda: HistGradientBoostingRegressor(max_iter=200, random_state=seed)}
    rows, fit = [dict(model="Mean baseline", setup="reference", R2=0.0, MAE=round(mean_absolute_error(yte, np.full(len(yte), ytr_raw.mean())), 2),
                      RMSE=round(float(np.sqrt(mean_squared_error(yte, np.full(len(yte), ytr_raw.mean())))), 2))], {}
    def add(label, setup, m, Xtr, Xte, y):
        m.fit(Xtr, y); p = m.predict(Xte); fit[label] = (m, p, Xte)
        rows.append(dict(model=label, setup=setup, R2=round(r2_score(yte, p), 4), MAE=round(mean_absolute_error(yte, p), 2),
                         RMSE=round(float(np.sqrt(mean_squared_error(yte, p))), 2)))
    for n, f in mk.items(): add(f"{n} (v2 setup)", "base features, raw target", f(), Xtr_b, Xte_b, ytr_raw)
    for n, f in mk.items(): add(f"{n} (enhanced)", "+ target encodings, capped log-target", ttr(f()), Xtr_e, Xte_e, ytr)
    sub = Xtr_e.sample(min(40000, len(Xtr_e)), random_state=seed)
    rs = RandomizedSearchCV(ttr(HistGradientBoostingRegressor(random_state=seed)), {
        "regressor__learning_rate": [.03, .05, .1, .2], "regressor__max_leaf_nodes": [15, 31, 63], "regressor__max_iter": [200, 400],
        "regressor__min_samples_leaf": [20, 50, 100], "regressor__l2_regularization": [0, .1, 1]},
        n_iter=8, cv=3, scoring="neg_root_mean_squared_error", random_state=seed, n_jobs=-1).fit(sub, ytr.loc[sub.index])
    add("Gradient Boosting (enhanced + tuned)", "+ randomised search (3-fold CV)",
        ttr(HistGradientBoostingRegressor(random_state=seed, **{k.split("__", 1)[1]: v for k, v in rs.best_params_.items()})), Xtr_e, Xte_e, ytr)
    res = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best = res[res.model != "Mean baseline"].model.iloc[0]; m, p, Xte = fit[best]
    s = Xte.sample(min(2000, len(Xte)), random_state=seed)
    pi = permutation_importance(m, s, yte.loc[s.index], n_repeats=3, random_state=seed, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=Xte.columns).groupby(lambda c: c.split("=")[0]).sum().sort_values(ascending=False)
    return res, best, imp, (yte.values, p)

# ------------------------------------------------------------------ HYBRID RECOMMENDER
REASON = dict(cf="Customers with similar taste ordered here", rt="Highly rated by diners", pp="Popular across the platform")

class Recommender:
    """score = w_cf*collaborative + w_cu*cuisine affinity + w_rt*smoothed rating + w_pp*popularity + w_lc*city/state affinity."""
    W = dict(cf=.6, cu=.25, rt=.15, pp=0., lc=0.)
    def __init__(self, orders, rest):
        u = orders.user_id.astype("category"); r = orders.restaurant_id.astype("category")
        self.u_idx = {x: i for i, x in enumerate(u.cat.categories)}; self.r_ids = r.cat.categories
        self.M = csr_matrix((np.ones(len(orders)), (u.cat.codes, r.cat.codes)), shape=(len(u.cat.categories), len(self.r_ids)))
        self.info = rest.set_index("restaurant_id").reindex(self.r_ids); inf = self.info
        txt = lambda c: inf[c].fillna("Unknown").values.astype(str)
        self.cuis, self.city, self.state = txt("cuisine"), txt("city"), txt("state")
        self.name = inf.restaurant_name.fillna("Unknown restaurant").values
        self.Cu, self.cuc = self._ind(self.cuis); self.Ci, self.cic = self._ind(self.city); self.St, self.stc = self._ind(self.state)
        rt = inf.rating.fillna(inf.rating.mean() if inf.rating.notna().any() else 0); cnt = inf.rating_count.fillna(0)
        C = max(float(np.median(cnt[cnt > 0])) if (cnt > 0).any() else 1., 1.); self.rt = ((rt * cnt + rt.mean() * C) / (cnt + C)).values / 5
        self.pop = np.asarray(self.M.sum(0)).ravel(); self.pp = np.log1p(self.pop) / np.log1p(self.pop).max()
    @staticmethod
    def _ind(a):
        codes, _ = pd.factorize(a); return csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes))), codes
    def components(self, vec, self_idx=None, k=30):
        if vec.nnz == 0: return None
        sims = cosine_similarity(vec, self.M).ravel()
        if self_idx is not None: sims[self_idx] = 0
        k = min(k, len(sims)); top = np.argpartition(sims, -k)[-k:]
        cf = np.asarray(self.M[top].T @ sims[top]).ravel(); cf = cf / (cf.max() or 1)
        def aff(Mat, codes):
            a = (vec @ Mat).toarray().ravel(); return (a / (a.sum() or 1))[codes]
        cu = aff(self.Cu, self.cuc); cu = cu / (cu.max() or 1)
        lc = .6 * aff(self.Ci, self.cic) + .4 * aff(self.St, self.stc); lc = lc / (lc.max() or 1)
        return dict(cf=cf.astype(np.float32), cu=cu.astype(np.float32), rt=self.rt, pp=self.pp, lc=lc.astype(np.float32))
    def score(self, comp, W=None):
        W = W or self.W; return sum(W[c] * comp[c] for c in W if W[c])
    def recommend(self, uid, n=10, cuisine=None, state=None, city=None):
        i = self.u_idx.get(uid); vec = self.M[i] if i is not None else csr_matrix((1, len(self.r_ids)))
        comp = self.components(vec, i)
        if comp is None: W, s = dict(pp=.5, rt=.5), .5 * self.pp + .5 * self.rt; comp = dict(pp=self.pp, rt=self.rt)
        else: W, s = self.W, self.score(comp)
        s = np.array(s, dtype=float); s[vec.indices] = -9
        for arr, val in ((self.cuis, cuisine), (self.state, state), (self.city, city)):
            if val: s = np.where(arr == val, s, -9)
        out = []
        for j in np.argsort(-s)[:n]:
            if s[j] <= -9: break
            best = max((c for c in W if W[c]), key=lambda c: W[c] * float(comp[c][j]))
            why = {"cu": f"Matches your taste for {self.cuis[j]}", "lc": f"Popular in your area ({self.city[j]})"}.get(best, REASON.get(best))
            if i is None: why = "Popular & highly rated (new user)"
            rt = self.info.rating.iloc[j]
            out.append(dict(restaurant_id=str(self.r_ids[j]), name=str(self.name[j]), cuisine=str(self.cuis[j]), city=str(self.city[j]),
                            state=str(self.state[j]), rating=None if pd.isna(rt) else float(rt), score=round(float(s[j]), 3), reason=why))
        return out, ("hybrid" if i is not None else "cold-start")

def evaluate_recommender(rec, ks=(5, 10), sample=None, seed=42, tune=True):
    """Hold out 20% of each test user's visited restaurants; report Precision@K, Recall@K, NDCG@K. Weights are tuned on
    one user group (NDCG@10) and every reported number comes from a different, untouched group."""
    rng = np.random.default_rng(seed); cand = np.where(np.diff(rec.M.indptr) >= 4)[0]; R = len(rec.r_ids)
    if len(cand) < 20: return dict(error="Too few users with >=4 distinct restaurants to evaluate", users_test=0)
    n = sample or (150 if R < 30000 else 60); pick = rng.choice(cand, min(2 * n, len(cand)), replace=False); half = len(pick) // 2
    def prep(users):
        out = []
        for i in users:
            row = rec.M[i]; held = rng.choice(row.indices, max(1, int(round(.2 * len(row.indices)))), replace=False)
            vec = row.tolil(); vec[0, held] = 0; vec = vec.tocsr(); vec.eliminate_zeros()
            out.append((rec.components(vec, i), set(held.tolist()), vec.indices))
        return out
    val, test = prep(pick[:half]), prep(pick[half:]); K = max(ks); disc = [1 / math.log2(i + 2) for i in range(K)]
    def run(data, W):
        acc = {f"{m}@{k}": [] for k in ks for m in ("precision", "recall", "ndcg")}
        for comp, held, seen in data:
            s = np.asarray(rec.score(comp, W), dtype=float); s[seen] = -9
            top = np.argpartition(-s, K)[:K]; top = top[np.argsort(-s[top])]; hit = [t in held for t in top]
            for k in ks:
                h = sum(hit[:k]); acc[f"precision@{k}"].append(h / k); acc[f"recall@{k}"].append(h / len(held))
                acc[f"ndcg@{k}"].append(sum(d for d, x in zip(disc[:k], hit[:k]) if x) / sum(disc[:min(len(held), k)]))
        return {m: round(float(np.mean(v)), 4) for m, v in acc.items()}
    if tune:
        grid = [dict(cf=a, cu=b, rt=c, pp=d, lc=e) for a, b, c, d, e in itertools.product([.4, .7, 1], [0, .25, .5], [0, .15, .3], [0, .15], [0, .2, .4])]
        rec.W = max(grid, key=lambda W: run(val, W)["ndcg@10"])
    res = {"Popularity only": run(test, dict(pp=1)), "Content only (cuisine + rating)": run(test, dict(cu=.7, rt=.3)),
           "Collaborative only": run(test, dict(cf=1)), "Hybrid v2 weights (CF+cuisine+rating)": run(test, dict(cf=.6, cu=.25, rt=.15)),
           "Tuned hybrid (+ popularity, city/state)": run(test, rec.W)}
    return dict(ks=list(ks), users_tune=len(val), users_test=len(test), weights={k: v for k, v in rec.W.items()}, results=res,
                protocol="Random 20% of each user's visited restaurants held out; relevant = held-out restaurants; users with >=4 distinct restaurants.")
