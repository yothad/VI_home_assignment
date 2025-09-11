import pandas as pd

class PreprocessData:
    """Class to handle data preprocessing tasks."""
    def __init__(self, raw_dfs):
        self.raw_dfs = raw_dfs
        self.prep_dfs = {}
        self.labels_df = raw_dfs['churn_labels']

    def preprocess_claims(self, df):
        """Pivot claims data to create binary features for each ICD code."""
        claims_df = df.groupby(['member_id', 'icd_code']).max().reset_index()
        pivot_df = claims_df.pivot(index='member_id', columns='icd_code', values='diagnosis_date').notnull().astype(int).reset_index()
        pivot_df.columns = [f'icd_{col}' if col != 'member_id' else col for col in pivot_df.columns]
        pivot_df.index.name = None
        return pivot_df
    
    def preprocess_app_usage(self, df):
        """Aggregate app usage data to count total app usage per member."""
        return df.groupby('member_id').size().reset_index(name='app_usage_count')
    
    def preprocess_web_visits(self, df):
        """Pivot web visits data to create binary features for each page title."""
        web_df = df.groupby(['member_id', 'title']).size().reset_index(name='count')
        web_pivot_df = web_df.pivot(index='member_id', columns='title', values='count').fillna(0).reset_index()
        web_pivot_df.columns = [f'web_{col}' if col != 'member_id' else col for col in web_pivot_df.columns]
        web_pivot_df.index.name = None
        return web_pivot_df
    
    def calc_subscription_length(self):
        """Calculate subscription length in days."""
        self.labels_df['signup_date'] = pd.to_datetime(self.labels_df['signup_date'])
        self.labels_df['subscription_length'] = (pd.Timestamp.today() - self.labels_df['signup_date']).dt.days
        self.labels_df = self.labels_df.drop(columns={'signup_date'})

    def _execute(self):
        self.calc_subscription_length()
        if 'claims' in self.raw_dfs:
            self.prep_dfs['claims'] = self.preprocess_claims(self.raw_dfs['claims'])
            self.labels_df = self.labels_df.merge(self.prep_dfs['claims'], on='member_id', how='left').fillna(0)
        if 'app_usage' in self.raw_dfs:
            self.prep_dfs['app_usage'] = self.preprocess_app_usage(self.raw_dfs['app_usage'])
            self.labels_df = self.labels_df.merge(self.prep_dfs['app_usage'], on='member_id', how='left').fillna(0)
        if 'web_visits' in self.raw_dfs:
            self.prep_dfs['web_visits'] = self.preprocess_web_visits(self.raw_dfs['web_visits'])
            self.labels_df = self.labels_df.merge(self.prep_dfs['web_visits'], on='member_id', how='left').fillna(0)
