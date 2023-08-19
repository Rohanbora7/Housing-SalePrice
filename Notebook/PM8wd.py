def outliers(df):
    df = standardize(df)
    outliers = []
    cat,con = catconsep(df)
    for i in con:
        outliers.extend(list(df[df[i]>3].index))
        outliers.extend(list(df[df[i]<-3].index))
    from numpy import unique
    Q = list(unique(outliers))
    return Q

def catconsep(df):
    cat = []
    con = []
    for i in df.columns:
        if(df[i].dtypes == "object"):
            cat.append(i)
        else:
            con.append(i)
    return cat,con


def standardize(df):
    import pandas as pd
    cat,con = catconsep(df)
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X1 = pd.DataFrame(ss.fit_transform(df[con]),columns=con)
    return X1

def replacer(df):
    cat,con = catconsep(df)
    for i in con:
        x = df[i].mean()
        df[i]=df[i].fillna(x)

    for i in cat:
        x = df[i].mode()[0]
        df[i]=df[i].fillna(x)
        
        
def preprocessing(df):
    cat,con = catconsep(df)
    from sklearn.preprocessing import MinMaxScaler
    ss = MinMaxScaler()
    import pandas as pd
    X1 = pd.DataFrame(ss.fit_transform(df[con]),columns=con)
    X2 = pd.get_dummies(df[cat])
    Xnew = X1.join(X2)
    return Xnew

def ANOVA(df,cat,con):
    from statsmodels.formula.api import ols
    eqn = str(con) + " ~ " + str(cat)
    model = ols(eqn,df).fit()
    from statsmodels.stats.anova import anova_lm
    Q = anova_lm(model)
    return round(Q.iloc[0:1,4:5].values[0][0],5)

def chisq(df,cat1,cat2):
    import pandas as pd
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(df[cat1],df[cat2])
    a,b,c,d = chi2_contingency(ct)
    return round(b,5)

def prep_NN(X,Y):
    from warnings import filterwarnings
    filterwarnings("ignore")
    from PM8wd import catconsep
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder,MinMaxScaler
    le = LabelEncoder()
    Ynew = pd.DataFrame(le.fit_transform(Y),columns=Y.columns)
    cat,con = catconsep(X)
    mm = MinMaxScaler()
    X1 = pd.DataFrame(mm.fit_transform(X[con]),columns=con)
    X2 = pd.get_dummies(X[cat])
    Xnew = X1.join(X2)
    return Xnew,Ynew


def reg_model(mo,Xnew,Y):
    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)
    model = mo.fit(xtrain,ytrain)
    tr_pred = model.predict(xtrain)
    ts_pred = model.predict(xtest)
    from sklearn.metrics import mean_absolute_error
    tr_err = round(mean_absolute_error(ytrain,tr_pred),2)
    ts_err = round(mean_absolute_error(ytest,ts_pred),2)
    return tr_err,ts_err
    
def cl_model(mo,Xnew,Y):
    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)
    model = mo.fit(xtrain,ytrain)
    tr_pred = model.predict(xtrain)
    ts_pred = model.predict(xtest)
    from sklearn.metrics import accuracy_score
    tr_acc = round(accuracy_score(ytrain,tr_pred),2)
    ts_acc = round(accuracy_score(ytest,ts_pred),2)
    return tr_acc,ts_acc

def replacer(df):
    for i in df.columns:
        if(df[i].dtypes == "object"):
            x = df[i].mode()[0]
            df[i] = df[i].fillna(x)
        else:
            x = df[i].mean()
            df[i] = df[i].fillna(x)