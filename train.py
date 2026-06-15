# Datathon 2026 - ana pipeline: features -> 5-fold CV -> LightGBM -> submission
# Kullanim:
#   python train.py --name baseline                 # sadece tabular
#   python train.py --name fe --fe                  # + feature engineering
#   python train.py --name fe_text --fe --text      # + TF-IDF/SVD metin ozellikleri
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).parent
DATA = ROOT / "datathon-2026"
SUBS = ROOT / "submissions"
OOF = ROOT / "oof"

TARGET = "career_success_score"
ID_COL = "student_id"
TEXT_COL = "mentor_feedback_text"
CAT_COLS = ["department", "university_tier", "target_role", "hobby",
            "preferred_social_media_platform"]

TECH_SCORES = ["coding_score", "problem_solving_score", "data_structures_score",
               "sql_score", "machine_learning_score", "backend_score",
               "frontend_score", "cloud_score", "devops_score"]
SOFT_SCORES = ["communication_score", "teamwork_score", "leadership_score",
               "presentation_score"]

# hedef role gore en alakali teknik skorlar (role-skill eslesmesi)
ROLE_SKILLS = {
    "Backend Developer": ["backend_score", "sql_score", "data_structures_score"],
    "Software Developer": ["coding_score", "data_structures_score", "problem_solving_score"],
    "Frontend Developer": ["frontend_score", "coding_score"],
    "Data Scientist": ["machine_learning_score", "sql_score", "problem_solving_score"],
    "Data Analyst": ["sql_score", "problem_solving_score"],
    "Cloud Engineer": ["cloud_score", "devops_score"],
    "AI Engineer": ["machine_learning_score", "coding_score"],
    "DevOps Engineer": ["devops_score", "cloud_score"],
    "Cybersecurity Analyst": ["problem_solving_score", "backend_score"],
    "MLOps Engineer": ["machine_learning_score", "devops_score", "cloud_score"],
    "Product Analyst": ["sql_score", "problem_solving_score"],
}


def load_data():
    tr = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    te = pd.read_csv(DATA / "test_x.csv", encoding="utf-8-sig")
    return tr, te


def year_weights(tr, te):
    # test, son yillara agirlik veriyor (2025-26 ~%42); egitim/skorlamada
    # satirlari test yil dagilimina gore agirlikla -> LB'ye esdeger metrik
    te_p = te["application_year"].value_counts(normalize=True)
    tr_p = tr["application_year"].value_counts(normalize=True)
    w = tr["application_year"].map(te_p / tr_p).fillna(0).to_numpy()
    return w / w.mean()


def add_fe(df):
    df = df.copy()
    df["interview_rate"] = df["interviews_attended"] / df["applications_sent"].replace(0, np.nan)
    df["award_rate"] = df["hackathon_awards"] / df["hackathon_count"].replace(0, np.nan)
    df["intern_months_per"] = df["internship_duration_months"] / df["internship_count"].replace(0, np.nan)
    df["years_since_grad"] = df["application_year"] - df["graduation_year"]
    df["tech_mean"] = df[TECH_SCORES].mean(axis=1)
    df["tech_max"] = df[TECH_SCORES].max(axis=1)
    df["tech_std"] = df[TECH_SCORES].std(axis=1)
    df["soft_mean"] = df[SOFT_SCORES].mean(axis=1)
    df["github_impact"] = df["github_repo_count"] * df["github_avg_stars"]
    df["total_projects"] = (df["real_client_project_count"] + df["freelance_project_count"]
                            + df["github_repo_count"])
    # hedef role uygun skorlarin ortalamasi
    role_mean = np.full(len(df), np.nan)
    for role, skills in ROLE_SKILLS.items():
        m = (df["target_role"] == role).to_numpy()
        if m.any():
            role_mean[m] = df.loc[m, skills].mean(axis=1).to_numpy()
    df["role_skill_mean"] = role_mean
    df["role_skill_gap"] = df["role_skill_mean"] - df["tech_mean"]
    df["text_len"] = df[TEXT_COL].str.len()
    df["text_word_count"] = df[TEXT_COL].str.split().str.len()
    return df


CORE_SCORES = ["project_quality_score", "technical_interview_score",
               "communication_score", "portfolio_score"]


def add_fe2(df):
    # son yillarda coklu-zayiflik hedefi cokertiyor (artik analizi 11 Haz):
    # carpimsal / min-bazli etkilesimleri aciktan ver
    df = df.copy()
    tech_mean = df[TECH_SCORES].mean(axis=1)
    core = df[CORE_SCORES].copy()
    core["tech_mean"] = tech_mean
    df["core_min"] = core.min(axis=1)
    df["core_geo"] = np.exp(np.log(core.clip(lower=1)).mean(axis=1))
    df["pq_x_ti"] = df["project_quality_score"] * df["technical_interview_score"] / 100
    df["tm_x_comm"] = tech_mean * df["communication_score"] / 100
    df["pq_x_tm"] = df["project_quality_score"] * tech_mean / 100
    all_scores = TECH_SCORES + SOFT_SCORES + CORE_SCORES
    df["weak50_count"] = (df[all_scores] < 50).sum(axis=1)
    df["weak40_count"] = (df[all_scores] < 40).sum(axis=1)
    yr = df["application_year"] - 2019
    df["yr_x_core"] = yr * core.mean(axis=1)
    df["yr_x_weak"] = yr * df["weak50_count"]
    return df


POS_PATTERNS = ["dikkat çek", "güçlü", "ileri düzey", "umut verici", "potansiyel",
                "etkili", "başarılı", "yetkin", "uzman", "tutku", "üst düzey"]
NEG_PATTERNS = ["geliştirme", "gelişim göster", "çalışması gerek", "ihtiyaç",
                "eksik", "zayıf", "yetersiz", "daha fazla"]
CONTRAST = ["ancak", "fakat", " ama "]


def add_keyword_features(df):
    df = df.copy()
    txt = df[TEXT_COL].str.lower()
    for i, p in enumerate(POS_PATTERNS):
        df[f"kw_pos_{i}"] = txt.str.count(p)
    for i, p in enumerate(NEG_PATTERNS):
        df[f"kw_neg_{i}"] = txt.str.count(p)
    df["kw_pos_total"] = df[[f"kw_pos_{i}" for i in range(len(POS_PATTERNS))]].sum(axis=1)
    df["kw_neg_total"] = df[[f"kw_neg_{i}" for i in range(len(NEG_PATTERNS))]].sum(axis=1)
    df["kw_balance"] = df["kw_pos_total"] - df["kw_neg_total"]
    df["kw_contrast"] = sum(txt.str.count(p) for p in CONTRAST)
    return df


def add_text_features(tr, te, n_components=64, seed=42):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import make_pipeline
    from scipy.sparse import hstack

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=50000, sublinear_tf=True)
    Xw = word.fit_transform(tr[TEXT_COL])
    Xc = char.fit_transform(tr[TEXT_COL])
    X = hstack([Xw, Xc]).tocsr()
    Xt = hstack([word.transform(te[TEXT_COL]), char.transform(te[TEXT_COL])]).tocsr()
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    Ztr = svd.fit_transform(X)
    Zte = svd.transform(Xt)
    cols = [f"txt_svd_{i}" for i in range(n_components)]
    tr = pd.concat([tr.reset_index(drop=True), pd.DataFrame(Ztr, columns=cols)], axis=1)
    te = pd.concat([te.reset_index(drop=True), pd.DataFrame(Zte, columns=cols)], axis=1)
    return tr, te


def add_embedding_features(tr, te, n_components=64, seed=42):
    # embed.py'nin cache'ledigi sentence-transformer embedding'leri (SVD ile indirgenir)
    from sklearn.decomposition import TruncatedSVD
    Etr = np.load(OOF / "emb_train.npy")
    Ete = np.load(OOF / "emb_test.npy")
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    Ztr = svd.fit_transform(Etr)
    Zte = svd.transform(Ete)
    cols = [f"emb_{i}" for i in range(n_components)]
    tr = pd.concat([tr.reset_index(drop=True), pd.DataFrame(Ztr, columns=cols)], axis=1)
    te = pd.concat([te.reset_index(drop=True), pd.DataFrame(Zte, columns=cols)], axis=1)
    return tr, te


def add_llm_features(tr, te):
    # llm_judge_colab.py'nin cikardigi yapilandirilmis LLM-judge feature'lari
    L = np.load(OOF / "llm_train.npy")
    Lte = np.load(OOF / "llm_test.npy")
    cols = ["llm_genel", "llm_teknik", "llm_iletisim", "llm_liderlik",
            "llm_proje", "llm_ovgu", "llm_elestiri", "llm_ustduzey"]
    tr = pd.concat([tr.reset_index(drop=True), pd.DataFrame(L, columns=cols)], axis=1)
    te = pd.concat([te.reset_index(drop=True), pd.DataFrame(Lte, columns=cols)], axis=1)
    return tr, te


def build_xy(tr, te, use_fe, use_text, use_kw=False, use_emb=False, keep_text=False,
             use_fe2=False, use_llm=False):
    if use_fe:
        tr, te = add_fe(tr), add_fe(te)
    if use_fe2:
        tr, te = add_fe2(tr), add_fe2(te)
    if use_kw:
        tr, te = add_keyword_features(tr), add_keyword_features(te)
    if use_llm:
        tr, te = add_llm_features(tr, te)
    if use_text:
        tr, te = add_text_features(tr, te)
    if use_emb:
        tr, te = add_embedding_features(tr, te)
    y = tr[TARGET].to_numpy()
    drop = [ID_COL, TARGET] + ([] if keep_text else [TEXT_COL])
    feats = [c for c in tr.columns if c not in drop]
    X_tr = tr[feats].copy()
    X_te = te[feats].copy()
    for c in CAT_COLS:
        X_tr[c] = X_tr[c].astype("category")
        X_te[c] = X_te[c].astype("category").cat.set_categories(X_tr[c].cat.categories)
    return X_tr, y, X_te, feats


def make_lgb(seed):
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        objective="regression", n_estimators=5000, learning_rate=0.03,
        num_leaves=63, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
        min_child_samples=30, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, verbose=-1)


def fit_predict_fold(model_type, X_tr, y_tr, X_va, y_va, X_te, seed, gpu=False,
                     w_tr=None, w_va=None, l2=None):
    if model_type == "lgb":
        import lightgbm as lgb
        model = make_lgb(seed)
        model.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)], eval_sample_weight=[w_va],
                  callbacks=[lgb.early_stopping(200, verbose=False),
                             lgb.log_evaluation(0)])
        return model.predict(X_va), model.predict(X_te), model.best_iteration_
    if model_type == "xgb":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=5000, learning_rate=0.03, max_depth=6,
            colsample_bytree=0.8, subsample=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
            device="cuda" if gpu else "cpu",
            enable_categorical=True, early_stopping_rounds=200,
            random_state=seed, verbosity=0)
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)],
                  sample_weight_eval_set=None if w_va is None else [w_va],
                  verbose=False)
        return model.predict(X_va), model.predict(X_te), model.best_iteration
    if model_type == "cat":
        from catboost import CatBoostRegressor, Pool
        cat_feats = [c for c in CAT_COLS if c in X_tr.columns]
        text_feats = [TEXT_COL] if TEXT_COL in X_tr.columns else []

        def to_pool(X, y=None, w=None):
            X = X.copy()
            for c in cat_feats:
                X[c] = X[c].astype(str)
            return Pool(X, y, weight=w, cat_features=cat_feats,
                        text_features=text_feats)

        extra = {} if l2 is None else {"l2_leaf_reg": l2}
        model = CatBoostRegressor(
            iterations=5000, learning_rate=0.03, depth=6,
            loss_function="RMSE", random_seed=seed, verbose=0,
            task_type="GPU" if gpu else "CPU",
            early_stopping_rounds=200, **extra)
        model.fit(to_pool(X_tr, y_tr, w_tr), eval_set=to_pool(X_va, y_va, w_va))
        return (model.predict(to_pool(X_va)), model.predict(to_pool(X_te)),
                model.get_best_iteration())
    raise ValueError(model_type)


def run_cv(X, y, X_te, model_type, n_splits=5, seed=42, gpu=False, w=None, l2=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_te))
    for fold, (itr, iva) in enumerate(kf.split(X)):
        va_pred, te_pred, best_iter = fit_predict_fold(
            model_type, X.iloc[itr], y[itr], X.iloc[iva], y[iva], X_te, seed, gpu,
            None if w is None else w[itr], None if w is None else w[iva], l2)
        oof[iva] = va_pred
        test_pred += te_pred / n_splits
        print(f"  fold {fold}: mse={mean_squared_error(y[iva], oof[iva]):.4f} "
              f"best_iter={best_iter}")
    return oof, test_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fe", action="store_true", help="feature engineering ekle")
    ap.add_argument("--text", action="store_true", help="TF-IDF/SVD metin ozellikleri ekle")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="lgb", choices=["lgb", "xgb", "cat"])
    ap.add_argument("--kw", action="store_true", help="metinden anahtar kelime sayimlari")
    ap.add_argument("--emb", action="store_true", help="cache'li sentence embedding'leri")
    ap.add_argument("--gpu", action="store_true", help="xgb/cat icin GPU kullan")
    ap.add_argument("--yw", action="store_true",
                    help="test yil dagilimina gore sample weight ile egit")
    ap.add_argument("--fe2", action="store_true",
                    help="carpimsal/min etkilesim feature'lari")
    ap.add_argument("--l2", type=float, default=None,
                    help="catboost l2_leaf_reg")
    ap.add_argument("--llm", action="store_true", help="LLM-judge feature'lari")
    args = ap.parse_args()

    tr, te = load_data()
    w = year_weights(tr, te)
    # catboost metni native isler (text_features), ham kolonu birakiyoruz
    X, y, X_te, feats = build_xy(tr, te, args.fe, args.text, use_kw=args.kw,
                                 use_emb=args.emb, keep_text=(args.model == "cat"),
                                 use_fe2=args.fe2, use_llm=args.llm)
    print(f"{args.name}: {X.shape[1]} feature, {len(X)} train, {len(X_te)} test")

    oof, test_pred = run_cv(X, y, X_te, args.model, seed=args.seed, gpu=args.gpu,
                            w=w if args.yw else None, l2=args.l2)
    oof_c = np.clip(oof, 0, 100)
    cv = mean_squared_error(y, oof_c)
    lb_est = mean_squared_error(y, oof_c, sample_weight=w)
    print(f"CV MSE (esit): {cv:.4f}   LB tahmini (yil-agirlikli): {lb_est:.4f}")

    OOF.mkdir(exist_ok=True)
    np.save(OOF / f"{args.name}_oof.npy", oof_c)
    np.save(OOF / f"{args.name}_test.npy", np.clip(test_pred, 0, 100))

    sub = pd.DataFrame({ID_COL: te[ID_COL], TARGET: np.clip(test_pred, 0, 100)})
    assert len(sub) == len(te) and sub[TARGET].notna().all()
    path = SUBS / f"sub_{args.name}_lb{lb_est:.3f}.csv"
    sub.to_csv(path, index=False)
    print(f"yazildi: {path}")

    with open(ROOT / "experiments.md", "a", encoding="utf-8") as f:
        f.write(f"| {dt.date.today()} | {args.name} | model={args.model} fe={args.fe} "
                f"fe2={args.fe2} text={args.text} kw={args.kw} emb={args.emb} "
                f"yw={args.yw} seed={args.seed} | {cv:.4f} | {lb_est:.4f} |\n")


if __name__ == "__main__":
    main()
