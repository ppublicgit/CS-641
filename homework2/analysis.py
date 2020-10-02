import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.tree import DecisionTreeClassifier

import dtree


def get_indices_dict(df):
    indices = {}
    for county in np.unique(df["county"].values):
        rf_indices = df.loc[(df["rf_irr"] == 0) & (df["county"] == county)].index.values
        irr_indices = df.loc[(df["rf_irr"] == 1) & (df["county"] == county)].index.values
        indices[county + "_rf"] = rf_indices
        indices[county + "_irr"] = irr_indices

    return indices


def split_indices(alist, train=0.25, validate=0.25, test=0.5, **kwargs):
    def almost_equals(left, right, tol=0.0001):
        if abs(left - right) < tol:
            return True

    if not almost_equals(train+validate+test, 1):
        raise ValueError("Train + validate + test should equal 1")

    copylist = copy.copy(alist)

    seed = kwargs.get("seed", 42)
    np.random.seed(seed)
    np.random.shuffle(copylist)

    num = len(copylist)
    train_stop = int(num*train)
    validate_stop = int(num*validate) + train_stop

    return copylist[:train_stop], copylist[train_stop:validate_stop], copylist[validate_stop:]


def get_train_val_test_df(df, train=0.25, validate=0.25, test=0.5, drop_cols=None, **kwargs):
    if drop_cols is not None:
        df = df.drop(columns=drop_cols)

    train = []
    validate = []
    test = []
    indices = get_indices_dict(df)
    for key, val in indices.items():
        tr, va, te = split_indices(val, **kwargs)
        train.append(tr)
        validate.append(va)
        test.append(te)

    train = np.concatenate(train)
    validate = np.concatenate(validate)
    test = np.concatenate(test)

    train_df = df.iloc[train]
    validate_df = df.iloc[validate]
    test_df = df.iloc[test]

    return train_df, validate_df, test_df


def extract_in_out(df, county=None):
    if county is None:
        x_in = df.drop(columns=["rf_irr", "county"]).to_numpy()
        y_out = df.loc[:, ["rf_irr"]].to_numpy()
    elif isinstance(county, str):
        x_in = df.loc[df["county"]==county].drop(columns=["rf_irr", "county"]).to_numpy()
        y_out = df.loc[df["county"]==county].loc[:, ["rf_irr"]].to_numpy()
    else:
        x_in = df.loc[df["county"].isin(county)].drop(columns=["rf_irr", "county"]).to_numpy()
        y_out = df.loc[df["county"].isin(county)].loc[:, ["rf_irr"]].to_numpy()
    return x_in, y_out


def calc_conf_mat(predictions, actual):
    conf_mat = np.zeros((2,2), dtype=int)
    for i in range(len(predictions)):
        if predictions[i] == actual[i] and actual[i] == 1:
            conf_mat[1,1] += 1
        elif predictions[i] == actual[i] and actual[i] == 0:
            conf_mat[0,0] += 1
        elif predictions[i] != actual[i] and predictions[i] == 1:
            conf_mat[1,0] += 1
        else:
            conf_mat[0,1] += 1
    return conf_mat

def print_conf_mat(conf_mat):
    print("                        Actual Class")
    print("                     Rainfed | Irrigated")
    print(f"Predicted Rainfed   |  {conf_mat[0,0]:<4}  |  {conf_mat[0,1]:<4}  ")
    print(f"Predicted Irrigated |  {conf_mat[1,0]:<4}  |  {conf_mat[1,1]:<4}  ")


def get_split_error(conf_mat):
    rf_er = 1 - conf_mat[0,0]/(conf_mat[0,0]+conf_mat[1,0])
    ir_er = 1 - conf_mat[1,1]/(conf_mat[0,1]+conf_mat[1,1])
    return rf_er, ir_er


def main(county=None, drop_cols=None):

    df = pd.read_csv(os.path.join(os.getcwd(), "data", "cleaned_data", "all_data.csv"), index_col=0)

    df.reset_index(inplace=True)

    df = df.drop(columns=["index"])

    train_df, validate_df, test_df = get_train_val_test_df(df, 0.25, 0.25, 0.5, drop_cols=drop_cols)

    X_train, y_train = extract_in_out(train_df, county)
    X_validate, y_validate = extract_in_out(validate_df, county)
    X_test, y_test = extract_in_out(test_df, county)

    train_gini_errors = []
    val_gini_errors = []
    max_depth = list(range(1,15))
    num_gini_nodes = []
    for md in max_depth:
        clf_gini_tree = DecisionTreeClassifier(criterion='gini', max_depth=md, random_state=1)
        clf_gini_tree.fit(X_train, y_train)
        num_gini_nodes.append(clf_gini_tree.tree_.node_count)
        train_gini_errors.append(1-clf_gini_tree.score(X_train, y_train))
        val_gini_errors.append(1-clf_gini_tree.score(X_validate, y_validate))

    train_entropy_errors = []
    val_entropy_errors = []
    test_entropy_errors = []
    num_entropy_nodes = []
    for md in max_depth:
        clf_entropy_tree = DecisionTreeClassifier(criterion='entropy', max_depth=md, random_state=1)
        clf_entropy_tree.fit(X_train, y_train)
        num_entropy_nodes.append(clf_entropy_tree.tree_.node_count)
        train_entropy_errors.append(1-clf_entropy_tree.score(X_train, y_train))
        val_entropy_errors.append(1-clf_entropy_tree.score(X_validate, y_validate))
        test_entropy_errors.append(1-clf_entropy_tree.score(X_test, y_test))

    if county is None:
        title = "Madison, Houston, Limestone"
    elif isinstance(county, str):
        title = county
    else:
        title = county[0] + ", " + county[1]

    if drop_cols is not None:
        title += f"\nDropped Cols {drop_cols}"

    fig, axes = plt.subplots(2,2, squeeze=False)
    fig.suptitle(title)
    axes[0,0].set_title("Error vs Max Depth")
    axes[0,0].plot(max_depth, train_gini_errors, label="train_gini")
    axes[0,0].plot(max_depth, val_gini_errors, label="val_gini")
    axes[0,0].plot(max_depth, train_entropy_errors, label="train_entropy")
    axes[0,0].plot(max_depth, val_entropy_errors, label="val_entropy")
    axes[0,0].plot(max_depth, test_entropy_errors, label="test_entropy")
    axes[0,0].legend()
    axes[0,0].grid(True)
    axes[0,0].set_xlabel("Max Depth")
    axes[0,0].set_ylabel("Error Rate")
    axes[1,0].set_title("Error vs Number of Nodes")
    axes[1,0].plot(num_gini_nodes, train_gini_errors, label="train_gini")
    axes[1,0].plot(num_gini_nodes, val_gini_errors, label="val_gini")
    axes[1,0].plot(num_entropy_nodes, train_entropy_errors, label="train_entropy")
    axes[1,0].plot(num_entropy_nodes, val_entropy_errors, label="val_entropy")
    axes[1,0].plot(num_entropy_nodes, test_entropy_errors, label="test_entropy")
    axes[1,0].grid(True)
    axes[1,0].legend()
    axes[1,0].set_xlabel("Num Nodes")
    axes[1,0].set_ylabel("Error Rate")


    train_gini_predictions = []
    val_gini_predictions = []
    test_gini_predictions = []
    split_train_error = []
    split_val_error = []
    split_test_error = []
    num_gini_nodes = []
    for i, md in enumerate(max_depth):
        clf_gini_tree = DecisionTreeClassifier(criterion='gini', max_depth=md, random_state=1)
        clf_gini_tree.fit(X_train, y_train)
        num_gini_nodes.append(clf_gini_tree.tree_.node_count)
        train_gini_predictions.append(clf_gini_tree.predict(X_train))
        val_gini_predictions.append(clf_gini_tree.predict(X_validate))
        test_gini_predictions.append(clf_gini_tree.predict(X_test))

        train_conf_mat = calc_conf_mat(train_gini_predictions[i], y_train)
        val_conf_mat = calc_conf_mat(val_gini_predictions[i], y_validate)
        test_conf_mat = calc_conf_mat(test_gini_predictions[i], y_test)

        split_train_error.append(get_split_error(train_conf_mat))
        split_val_error.append(get_split_error(val_conf_mat))
        split_test_error.append(get_split_error(test_conf_mat))

        print(f"========= Max-Depth {md} =========")
        print_conf_mat(train_conf_mat)
        print("")
        print_conf_mat(val_conf_mat)

    rf_tr_error, ir_tr_error, rf_vl_error, ir_vl_error, rf_te_error, ir_te_error = [], [] ,[] ,[], [], []

    for i in range(len(split_train_error)):
        rf_tr_error.append(split_train_error[i][0])
        ir_tr_error.append(split_train_error[i][1])
        rf_vl_error.append(split_val_error[i][0])
        ir_vl_error.append(split_val_error[i][1])
        rf_te_error.append(split_test_error[i][0])
        ir_te_error.append(split_test_error[i][1])

    axes[0,1].set_title("Rainfed/Irrigated Prediction Error vs Max Depth")
    axes[0,1].plot(max_depth, rf_tr_error, label="rainfed_train")
    axes[0,1].plot(max_depth, ir_tr_error, label="irrigated_train")
    axes[0,1].plot(max_depth, rf_vl_error, label="rainfed_val")
    axes[0,1].plot(max_depth, ir_vl_error, label="irrigated_val")
    axes[0,1].plot(max_depth, rf_te_error, label="rainfed_test")
    axes[0,1].plot(max_depth, ir_te_error, label="irrigated_test")
    axes[0,1].legend()
    axes[0,1].grid(True)
    axes[0,1].set_xlabel("Max Depth")
    axes[0,1].set_ylabel("Error Rate")
    axes[1,1].set_title("Rainfed/Irrigated Prediction Error vs Number of Nodes")
    axes[1,1].plot(num_gini_nodes, rf_tr_error, label="rainfed_train")
    axes[1,1].plot(num_gini_nodes, ir_tr_error, label="irrigated_train")
    axes[1,1].plot(num_gini_nodes, rf_vl_error, label="rainfed_val")
    axes[1,1].plot(num_gini_nodes, ir_vl_error, label="irrigated_val")
    axes[1,1].plot(num_gini_nodes, rf_te_error, label="rainfed_test")
    axes[1,1].plot(num_gini_nodes, ir_te_error, label="irrigated_test")
    axes[1,1].legend()
    axes[1,1].set_xlabel("Num Nodes")
    axes[1,1].set_ylabel("Error Rate")
    axes[1,1].grid(True)

    plt.show(block=False)


if __name__ == "__main__":
    counties = [None, "Madison", "Limestone", "Houston", ["Madison", "Limestone"]]
    drop_cols = [None, ["Longitude"], ["Latitude", "Longitude"]]
    for county in counties:
        for dc in drop_cols:
            main(county, dc)
    input("Enter to exit...")
