
# provider_audit_pipeline.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['audit_date'] = pd.to_datetime(df['audit_date'])
    df['audit_month'] = df['audit_date'].dt.to_period('M')
    return df

def calculate_fields(df):
    df['MRR_rate'] = df['records_received'] / df['records_requested']
    df['hit_rate'] = df['findings'] / (df['findings'] + df['no_findings'])
    df['recovery_rate'] = df['recovery_amount'] / df['invoiced_amount']
    return df

def provider_summary(df):
    summary = df.groupby('providertaxid').agg(
        total_claims=('claim_id', 'count'),
        records_requested=('records_requested', 'sum'),
        records_received=('records_received', 'sum'),
        findings=('findings', 'sum'),
        no_findings=('no_findings', 'sum'),
        overpayment_avg=('overpayment_amount', 'mean'),
        invoiced_amount=('invoiced_amount', 'sum'),
        recovery_amount=('recovery_amount', 'sum'),
        total_disputes=('dispute_status', lambda x: (x != 'No Dispute').sum()),
    )
    summary['MRR_rate'] = summary['records_received'] / summary['records_requested']
    summary['hit_rate'] = summary['findings'] / (summary['findings'] + summary['no_findings'])
    summary['dispute_rate'] = summary['total_disputes'] / summary['total_claims']
    summary['recovery_rate'] = summary['recovery_amount'] / summary['invoiced_amount']
    return summary

def plot_distributions(df):
    sns.histplot(df['MRR_rate'], kde=True)
    plt.title('MRR Rate Distribution')
    plt.show()

    sns.histplot(df['hit_rate'], kde=True)
    plt.title('Hit Rate Distribution')
    plt.show()

    sns.boxplot(x='dispute_status', y='overpayment_amount', data=df)
    plt.title('Overpayment by Dispute Status')
    plt.show()

def new_providers_monthly(df, program='APC2'):
    df_program = df[df['program'] == program]
    df_program['audit_month'] = df_program['audit_date'].dt.to_period('M')
    first_seen = df_program.groupby('providertaxid')['audit_month'].min()
    new_provider_counts = first_seen.value_counts().sort_index()
    new_provider_counts.plot(kind='bar', title=f'New {program} Providers Per Month')
    plt.ylabel('Count')
    plt.xlabel('Month')
    plt.xticks(rotation=45)
    plt.show()

def program_overlap(df):
    apc1_providers = set(df[df['program'] == 'APC1']['providertaxid'].unique())
    apc2_providers = set(df[df['program'] == 'APC2']['providertaxid'].unique())
    only_apc1 = apc1_providers - apc2_providers
    only_apc2 = apc2_providers - apc1_providers
    both = apc1_providers & apc2_providers
    return len(only_apc1), len(only_apc2), len(both)

def high_volume_providers(df):
    claim_counts = df.groupby('providertaxid')['claim_id'].count()
    top_providers = claim_counts[claim_counts > claim_counts.quantile(0.95)].index
    df_high_vol = df[df['providertaxid'].isin(top_providers)]
    return df_high_vol.groupby('providertaxid')[['MRR_rate', 'hit_rate', 'recovery_rate']].mean()

def recovery_trend(df):
    monthly_recovery = df.groupby(df['audit_date'].dt.to_period('M')).agg(
        invoiced=('invoiced_amount', 'sum'),
        recovered=('recovery_amount', 'sum')
    )
    monthly_recovery['recovery_rate'] = monthly_recovery['recovered'] / monthly_recovery['invoiced']
    monthly_recovery[['invoiced', 'recovered']].plot(kind='line', title='Monthly Invoiced vs Recovered')
    plt.ylabel('Amount')
    plt.show()

# Example usage
if __name__ == '__main__':
    df = load_data('your_data.csv')
    df = calculate_fields(df)
    summary = provider_summary(df)
    plot_distributions(df)
    new_providers_monthly(df, 'APC2')
    overlap = program_overlap(df)
    print("Overlap:", overlap)
    top_behavior = high_volume_providers(df)
    print(top_behavior.head())
    recovery_trend(df)
