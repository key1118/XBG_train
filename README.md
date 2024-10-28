def train_lgb(input_x,
              input_y,
              input_id,
              params,
              list_nfold=[0,1,2,3,4],
              n_splits=5,
             ):
    train_oof = np.zeros(len(input_x))
    metrics = []
    imp = pd.DataFrame()

    # cross-validation
    cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(input_x, input_y))
    for nfold in list_nfold:
        print("-"*20, nfold, "-"*20)
        
        # make dataset
        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
        x_tr, y_tr, id_tr = input_x.loc[idx_tr, :], input_y[idx_tr], input_id.loc[idx_tr, :]
        x_va, y_va, id_va = input_x.loc[idx_va, :], input_y[idx_va], input_id.loc[idx_va, :]
        print(x_tr.shape, x_va.shape)
        
        # train
        model = lgb.LGBMClassifier(**params)
        #model.fit(x_tr,
                 # y_tr,
                  #eval_set=[(x_tr, y_tr), (x_va, y_va)],
                  #early_stopping_rounds=100,
                  #verbose=100
                # )
        
         # 2024/02/14環境で動かしたい場合はこのコードを利用してください。
        model.fit(x_tr,
                  y_tr,
                  eval_set=[(x_tr,y_tr), (x_va,y_va)],
                   callbacks=[
                       lgb.early_stopping(stopping_rounds=100, verbose=True),
                       lgb.log_evaluation(100),
                   ],
                  )
        
        fname_lgb = "model_lgb_fold{}.pickle".format(nfold)
        with open(fname_lgb, "wb") as f:
            pickle.dump(model, f, protocol=4)
        
        # evaluate
        y_tr_pred = model.predict_proba(x_tr)[:,1]
        y_va_pred = model.predict_proba(x_va)[:,1]
        metric_tr = roc_auc_score(y_tr, y_tr_pred)
        metric_va = roc_auc_score(y_va, y_va_pred)
        metrics.append([nfold, metric_tr, metric_va])
        print("[auc] tr:{:.4f}, va:{:.4f}".format(metric_tr, metric_va))
        
        # oof
        train_oof[idx_va] = y_va_pred
        
        # imp
        _imp = pd.DataFrame({"col":input_x.columns, "imp":model.feature_importances_, "nfold":nfold})
        imp = pd.concat([imp, _imp])
      
    print("-"*20, "result", "-"*20)
    # metric
    metrics = np.array(metrics)
    print(metrics)
    print("[cv] tr:{:.4f}+-{:.4f}, va:{:.4f}+-{:.4f}".format(
        metrics[:,1].mean(), metrics[:,1].std(),
        metrics[:,2].mean(), metrics[:,2].std(),
    ))
    print("[oof] {:.4f}".format(
        roc_auc_score(input_y, train_oof)
    ))
    
    # oof
    train_oof = pd.concat([
        input_id,
        pd.DataFrame({"pred":train_oof})
    ], axis=1)
    
    # importance
    imp = imp.groupby("col")["imp"].agg(["mean", "std"]).reset_index(drop=False)
    imp.columns = ["col", "imp", "imp_std"]
    
    return train_oof, imp, metrics
