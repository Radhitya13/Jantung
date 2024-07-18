import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Fungsi untuk evaluasi model
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates the performance of a trained model on test data using various metrics.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        "precision_0": report["0"]["precision"],
        "precision_1": report["1"]["precision"],
        "recall_0": report["0"]["recall"],
        "recall_1": report["1"]["recall"],
        "f1_0": report["0"]["f1-score"],
        "f1_1": report["1"]["f1-score"],
        "macro_avg_precision": report["macro avg"]["precision"],
        "macro_avg_recall": report["macro avg"]["recall"],
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "accuracy": accuracy_score(y_test, y_pred)
    }
    
    df = pd.DataFrame(metrics, index=[model_name]).round(2)
    return df

# Fungsi untuk tuning hyperparameters
def tune_clf_hyperparameters(clf, param_grid, X_train, y_train, scoring='recall', n_splits=3):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    clf_grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    clf_grid.fit(X_train, y_train)
    best_hyperparameters = clf_grid.best_params_
    return clf_grid.best_estimator_, best_hyperparameters

# Streamlit sidebar
st.sidebar.title("Pengaturan")

uploaded_file = st.sidebar.file_uploader("Unggah file CSV", type="csv")
target_variable = st.sidebar.text_input("Masukkan nama variabel target")

dataset_name = "heart.csv"


# Pengaturan model
st.sidebar.subheader("Pengaturan Model")
models = st.sidebar.multiselect("Pilih model", ["Decision Tree", "Random Forest", "KNN", "SVM"], default=["Decision Tree", "Random Forest"])

if uploaded_file is not None and target_variable:
    df = pd.read_csv(uploaded_file)
    
    st.write("### Dataframe")
    st.write(df.head())

    st.write("### Info Data")
    buffer = pd.io.common.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    continuous_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_variable in continuous_features:
        continuous_features.remove(target_variable)
    
    features_to_convert = [feature for feature in df.columns if feature not in continuous_features]
    df[features_to_convert] = df[features_to_convert].astype('object')
    
    df.describe().T
    df.describe(include='object')
    
    df_continuous = df[continuous_features]
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    for i, col in enumerate(df_continuous.columns):
        x = i // 3
        y = i % 3
        values, bin_edges = np.histogram(df_continuous[col], 
                                         range=(np.floor(df_continuous[col].min()), np.ceil(df_continuous[col].max())))
        
        graph = sns.histplot(data=df_continuous, x=col, bins=bin_edges, kde=True, ax=ax[x, y],
                             edgecolor='none', color='red', alpha=0.6, line_kws={'lw': 3})
        ax[x, y].set_xlabel(col, fontsize=15)
        ax[x, y].set_ylabel('Count', fontsize=12)
        ax[x, y].set_xticks(np.round(bin_edges, 1))
        ax[x, y].set_xticklabels(ax[x, y].get_xticks(), rotation=45)
        ax[x, y].grid(color='lightgrey')
        
        for j, p in enumerate(graph.patches):
            ax[x, y].annotate('{}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() + 1),
                              ha='center', fontsize=10, fontweight="bold")
        
        textstr = '\n'.join((
            r'$\mu=%.2f$' % df_continuous[col].mean(),
            r'$\sigma=%.2f$' % df_continuous[col].std()
        ))
        ax[x, y].text(0.75, 0.9, textstr, transform=ax[x, y].transAxes, fontsize=12, verticalalignment='top',
                      color='white', bbox=dict(boxstyle='round', facecolor='#ff826e', edgecolor='white', pad=0.5))

    ax[1,2].axis('off')
    plt.suptitle('Distribution of Continuous Variables', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    st.pyplot(fig)

    categorical_features = df.columns.difference(continuous_features)
    df_categorical = df[categorical_features]

    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15, 18))
    for i, col in enumerate(categorical_features):
        row = i // 2
        col_idx = i % 2
        value_counts = df[col].value_counts(normalize=True).mul(100).sort_values()
        value_counts.plot(kind='barh', ax=ax[row, col_idx], width=0.8, color='red')
        
        for index, value in enumerate(value_counts):
            ax[row, col_idx].text(value, index, str(round(value, 1)) + '%', fontsize=15, weight='bold', va='center')
        
        ax[row, col_idx].set_xlim([0, 95])
        ax[row, col_idx].set_xlabel('Frequency Percentage', fontsize=12)
        ax[row, col_idx].set_title(f'{col}', fontsize=20)

    ax[4,1].axis('off')
    plt.suptitle('Distribution of Categorical Variables', fontsize=22)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    st.pyplot(fig)

    sns.set_palette(['#ff826e', 'red'])
    fig, ax = plt.subplots(len(continuous_features), 2, figsize=(15,15), gridspec_kw={'width_ratios': [1, 2]})
    for i, col in enumerate(continuous_features):
        graph = sns.barplot(data=df, x=target_variable, y=col, ax=ax[i,0])
        sns.kdeplot(data=df[df[target_variable]==0], x=col, fill=True, linewidth=2, ax=ax[i,1], label='0')
        sns.kdeplot(data=df[df[target_variable]==1], x=col, fill=True, linewidth=2, ax=ax[i,1], label='1')
        ax[i,1].set_yticks([])
        ax[i,1].legend(title='Heart Disease', loc='upper right')
        
        for cont in graph.containers:
            graph.bar_label(cont, fmt='         %.3g')
            
    plt.suptitle('Continuous Features vs Target Distribution', fontsize=22)
    plt.tight_layout()                     
    st.pyplot(fig)

    categorical_features = [feature for feature in categorical_features if feature != target_variable]
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15,10))
    for i,col in enumerate(categorical_features):
        cross_tab = pd.crosstab(index=df[col], columns=df[target_variable])
        cross_tab_prop = pd.crosstab(index=df[col], columns=df[target_variable], normalize='index')
        cmp = ListedColormap(['#ff826e', 'red'])
        x, y = i//4, i%4
        cross_tab_prop.plot(kind='bar', ax=ax[x,y], stacked=True, width=0.8, colormap=cmp,
                            legend=False, ylabel='Proportion', sharey=True)
        
        for idx, val in enumerate([*cross_tab.index.values]):
            for (proportion, count, y_location) in zip(cross_tab_prop.loc[val],cross_tab.loc[val],cross_tab_prop.loc[val].cumsum()):
                ax[x,y].text(idx, y_location- proportion / 2, f'{count}', color='black', weight='bold')
        
        ax[x,y].set_title(col, fontsize=18, y=1.1)
        ax[x,y].tick_params(axis='x', rotation=45)
    
    handles, labels = ax[1,3].get_legend_handles_labels()
    plt.legend(handles, ['No','Yes'], bbox_to_anchor=(1.05, 1), loc='upper left', title='Heart Disease')
    plt.suptitle('Proportion of Target variable by Categorical Features', fontsize=22)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    st.pyplot(fig)

    df[continuous_features] = df[continuous_features].apply(lambda x: boxcox(x + 1)[0] if (x<=0).any() else boxcox(x)[0], axis=0)
    
    df[features_to_convert] = df[features_to_convert].astype('category')
    df = pd.get_dummies(df)
    
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    
    st.write("### Hasil Evaluasi Model")
    
    all_results = []
    
    if "Decision Tree" in models:
        dt_clf = DecisionTreeClassifier(random_state=0)
        param_grid = {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5, 10]
        }
        best_dt_clf, best_dt_params = tune_clf_hyperparameters(dt_clf, param_grid, X_train, y_train)
        results = evaluate_model(best_dt_clf, X_test, y_test, "Decision Tree")
        st.write("### Decision Tree")
        st.write(f"Best Parameters: {best_dt_params}")
        st.write(results)
        all_results.append(results)
    
    if "Random Forest" in models:
        rf_clf = RandomForestClassifier(random_state=0)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5, 10]
        }
        best_rf_clf, best_rf_params = tune_clf_hyperparameters(rf_clf, param_grid, X_train, y_train)
        results = evaluate_model(best_rf_clf, X_test, y_test, "Random Forest")
        st.write("### Random Forest")
        st.write(f"Best Parameters: {best_rf_params}")
        st.write(results)
        all_results.append(results)
    
    if "KNN" in models:
        knn_clf = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        best_knn_clf, best_knn_params = tune_clf_hyperparameters(knn_clf, param_grid, X_train, y_train)
        results = evaluate_model(best_knn_clf, X_test, y_test, "KNN")
        st.write("### KNN")
        st.write(f"Best Parameters: {best_knn_params}")
        st.write(results)
        all_results.append(results)
    
    if "SVM" in models:
        svm_clf = SVC(random_state=0)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }
        best_svm_clf, best_svm_params = tune_clf_hyperparameters(svm_clf, param_grid, X_train, y_train)
        results = evaluate_model(best_svm_clf, X_test, y_test, "SVM")
        st.write("### SVM")
        st.write(f"Best Parameters: {best_svm_params}")
        st.write(results)
        all_results.append(results)

    final_results = pd.concat(all_results)
    st.write("### Hasil Akhir")
    st.write(final_results)
