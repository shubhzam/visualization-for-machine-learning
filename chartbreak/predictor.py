"""Optional baseline and failure classifiers with database-disjoint evaluation."""
from .analysis import EVALUATED


def evaluate_predictors(records, folds=5):
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    if any(r['mode']=='demo' for r in records):
        raise ValueError('Predictor evaluation requires real model results, not simulated fixtures.')
    scored=[r for r in records if r['outcome'] in EVALUATED]
    frame=pd.DataFrame([dict(**r['features'],model=r['model'],strategy=r['strategy'],db_id=r['db_id'],failure=int(r['outcome']!='pass')) for r in scored])
    if frame.empty or frame.db_id.nunique()<folds:
        raise ValueError(f'Need results from at least {folds} distinct databases.')
    y=frame.failure; groups=frame.db_id
    categorical=['chart_type','hardness','model','strategy']
    numeric=['join','sort','binning','three_channels','num_tables','query_length']
    results=[]
    for name in ['baseline','logistic','gradient_boosted']:
        cats=categorical[:2] if name=='baseline' else categorical
        nums=[] if name=='baseline' else numeric
        transforms=[('categories',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)]
        if nums: transforms.append(('numbers',StandardScaler(),nums))
        for fold,(train,test) in enumerate(GroupKFold(n_splits=folds).split(frame,y,groups)):
            if y.iloc[train].nunique()<2 or y.iloc[test].nunique()<2:
                results.append(dict(model=name,fold=fold,status='skipped_one_class'));continue
            estimator=HistGradientBoostingClassifier(random_state=42) if name=='gradient_boosted' else LogisticRegression(max_iter=1000,random_state=42)
            pipe=make_pipeline(ColumnTransformer(transforms),estimator)
            pipe.fit(frame.iloc[train],y.iloc[train]);prob=pipe.predict_proba(frame.iloc[test])[:,1]
            results.append(dict(model=name,fold=fold,status='ok',n_test=len(test),
                auroc=float(roc_auc_score(y.iloc[test],prob)),ap=float(average_precision_score(y.iloc[test],prob))))
    return results
