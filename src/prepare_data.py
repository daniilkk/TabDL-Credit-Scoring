import os
import re
import argparse
from typing import List, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def get_loan_types(df: pd.DataFrame) -> List[str]:
    loan_types = set()

    for text_value in df['Type_of_Loan']:
        if pd.isnull(text_value):
            continue

        values = text_value.split(', ')
        for value in values:
            if value.startswith('and '):
                value = value[4:]
            value = '_'.join(value.split(' '))
            loan_types.add(value)

    loan_types_sorted = sorted(list(loan_types))
    loan_types_sorted.insert(0, 'Sum')

    return loan_types_sorted


def split_type_of_loan(df: pd.DataFrame, loan_types_sorted: List[str]):
    def extract_counter(text_value: str) -> List[int]:
        loan_types_counter = Counter({key: 0 for key in loan_types_sorted})
        if not pd.isnull(text_value):
            values = text_value.split(', ')

            for value in values:
                if value.startswith('and '):
                    value = value[4:]
                value = '_'.join(value.split(' '))
                
                loan_types_counter[value] += 1
                loan_types_counter['Sum'] += 1

        return [loan_types_counter[loan_type] for loan_type in loan_types_sorted]

    new_type_of_loan_columns = list(zip(*df['Type_of_Loan'].map(extract_counter)))

    for idx, loan_type in enumerate(loan_types_sorted):
        df['Type_of_Loan_' + loan_type] = new_type_of_loan_columns[idx]

    df.drop(columns='Type_of_Loan', inplace=True)


def convert_credit_history_age_to_months(df: pd.DataFrame):
    def convert_to_months(text_value: str) -> str:
        split_data = text_value.split(" ")
        return str(int(split_data[0]) * 12 + int(split_data[3]))

    df['Credit_History_Age'] = df['Credit_History_Age'].map(convert_to_months)


def split_payment_behaviour(df: pd.DataFrame):
    df.drop(df[(df['Payment_Behaviour'] == '!@9#%8')].index, inplace=True)

    def split(payment_behaviour: str) -> Tuple[str, str]:
        split = payment_behaviour.split('_')

        return split[0], split[2]

    df["Payment_Behaviour_Spent"], df["Payment_Behaviour_Value"] = zip(*df['Payment_Behaviour'].map(split))
    df.drop(columns='Payment_Behaviour', inplace=True)


def plot_feature_hist(df, feature):
    plt.hist(df[feature], bins=10 if feature == 'Num_Bank_Accounts' else 20, log=True)

    plt.title(feature)

    hist_figures_root = 'figures/hists'
    save_path = os.path.join(hist_figures_root, f'{feature}.png')
    plt.savefig(fname=save_path, dpi=300)
    plt.clf()

def replace_outliers(df: pd.DataFrame, feature_name: str, l_threshold: float, r_threshold: float):
    df_no_outliers = df[(df[feature_name] >= l_threshold) & (df[feature_name] <= r_threshold)]
    
    median_feature_for_person = dict(
        df_no_outliers[[feature_name, 'Customer_ID']]
        .groupby(['Customer_ID']).median()[feature_name]
    )

    def replace_if_needed(row):
        if row[feature_name] < l_threshold or row[feature_name] > r_threshold:
            return median_feature_for_person.get(row['Customer_ID'], np.nan)
        else:
            return row[feature_name]

    df[feature_name] = df[[feature_name, 'Customer_ID']].apply(replace_if_needed, axis=1)

def parse_numeric_features(df: pd.DataFrame, num_features: List[str]):
    def remove_underscore(val):
        if '_' in str(val):
            return val.strip('_')
        else:
            return val
    
    for feature_name in num_features:
        df[feature_name] = df[feature_name].apply(remove_underscore)

        # df[feature_name] = df[feature_name].map(lambda x: re.sub('\D', '', str(x)))


def main(args: argparse.Namespace):
    df = pd.read_csv(args.source, low_memory=False)

    df.drop(columns=['ID', 'Name', 'SSN', 'Monthly_Inhand_Salary', 'Num_of_Loan'], inplace=True)

    df.replace('_', np.NaN, inplace=True)
    df.replace('', np.NaN, inplace=True)

    # Occurs only in "Occupation" feature.
    df.replace('_______', 'Unemployed', inplace=True)

    loan_types_sorted = get_loan_types(df)
    loan_types_features = ['Type_of_Loan_' + loan_type for loan_type in loan_types_sorted]
    split_type_of_loan(df, loan_types_sorted)    

    df.dropna(inplace=True)

    convert_credit_history_age_to_months(df)
    
    num_features = [
        'Age',
        'Annual_Income',
        'Num_Bank_Accounts',
        'Num_Credit_Card',
        'Interest_Rate',
        'Delay_from_due_date',
        'Num_of_Delayed_Payment',
        'Changed_Credit_Limit',
        'Num_Credit_Inquiries',
        'Amount_invested_monthly',
        'Outstanding_Debt',
        'Credit_Utilization_Ratio',
        'Credit_History_Age',
        'Total_EMI_per_month',
        'Amount_invested_monthly',
        'Monthly_Balance',
        *loan_types_features
    ]

    df = df[df['Monthly_Balance'] != '__-333333333333333333333333333__']
    
    parse_numeric_features(df, num_features)
    df = df.astype({feature_name: float for feature_name in num_features})

    # df = cap_num_features(df)

    outliers_thresholds = {
        'Age': (18, 80),
        'Amount_invested_monthly': (-np.inf, 1100),
        'Annual_Income': (-np.inf, 200000),
        'Interest_Rate': (-np.inf, 35),
        'Num_Bank_Accounts': (1, 11),
        'Num_Credit_Card': (1, 11),
        'Num_Credit_Inquiries': (-np.inf, 17),
        'Num_of_Delayed_Payment': (0, 28),
        'Total_EMI_per_month': (-np.inf, 1701.98),
    }

    for feature_name, thresholds in outliers_thresholds.items():
        replace_outliers(df, feature_name, *thresholds)

    df.dropna(inplace=True)

    for feature in num_features:
        plot_feature_hist(df, feature)

    split_payment_behaviour(df)
    
    # Change 'Standard' label to 'Good', so there are only 'Good' and 'Bad' now.
    df['Credit_Score'] = df['Credit_Score'].replace({'Standard': 'Good'})

    # df.drop(columns='Credit_Mix', inplace=True)

    df.to_csv(args.destination, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, required=True, help='Path to the raw dataset')
    parser.add_argument('-d', '--destination', type=str, required=True, help='Path for saving the cleaned dataset')

    main(parser.parse_args())
