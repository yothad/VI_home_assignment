import pandas as pd

class ExportData:
    """Class to handle exporting predictions for live members."""
    def __init__(self, 
                 path,
                 churn_labels_df,
                 selected_features,
                 model,
                 prob_threshold=0.5):
        self.churn_labels_df = churn_labels_df
        self.model = model
        self.path = path
        self.prob_threshold = prob_threshold
        self.selected_features = selected_features

    def predict_live_members(self):
        """Predict churn risk for live members who haven't churned or been outreached."""
        X_live = self.churn_labels_df.copy()
        X_live = X_live[(X_live['churn'] == 0) & (X_live['outreach'] == 0)]
        member_ids = X_live['member_id'].values # Keep member_id separately if you need it later
        X_live = X_live[self.selected_features]
        y_live_probs = self.model.predict_proba(X_live)[:, 1]
        X_live["churn_risk"] = y_live_probs
        results_df = pd.DataFrame({
            'member_id': member_ids,
            'churn_risk': y_live_probs
        })
        results_df = results_df.sort_values("churn_risk", ascending=False).reset_index(drop=True).reset_index()
        return results_df

    def export_to_csv(self, X_live, path):
        """Export members with churn risk above the threshold to a CSV file."""
        export_df = X_live[X_live['churn_risk'] > self.prob_threshold]
        export_df.to_csv(path, index=False)
        print(f"{len(export_df)} members exported to {path}")

    def _execute(self):
        X_live = self.predict_live_members()
        self.export_to_csv(X_live, self.path)
        return X_live
