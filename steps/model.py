import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support


class ModelChurn:
    """Class to handle churn prediction model training, feature selection, and evaluation."""
    def __init__(self, 
                 churn_labels_df,
                 test_size=0.3333,
                 random_state=42,
                 eval_metric='logloss',
                 n_features_to_select=10,
                 step=1,
                 tune_metric='f1'):
        self.churn_labels_df = churn_labels_df
        self.test_size = test_size
        self.random_state = random_state
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.eval_metric = eval_metric
        self.tune_metric = tune_metric

    def split_data(self):
        """Split data into training and testing sets."""
        features = [col for col in self.churn_labels_df.columns if col != "churn"]
        X = self.churn_labels_df[features]
        y = self.churn_labels_df['churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test
    
    def smote_resample(self, X_train, y_train):
        """Apply SMOTE to balance the training data."""
        smote = SMOTE(random_state=self.random_state)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        return X_train_sm, y_train_sm

    def feature_selection_rfe(self, X_train_sm, y_train_sm):
        """Perform Recursive Feature Elimination (RFE) with XGBoost to select top features."""
        model_for_rfe = XGBClassifier(eval_metric=self.eval_metric, random_state=self.random_state)
        rfe = RFE(estimator=model_for_rfe, n_features_to_select=self.n_features_to_select, step=self.step)
        rfe.fit(X_train_sm, y_train_sm)

        selected_features = X_train_sm.columns[rfe.support_].tolist()
        print("Selected features:", selected_features)
        return selected_features
    
    def train_model(self, X_train_sm, y_train_sm, selected_features):
        """Train the final XGBoost model on the selected features."""
        X_train_selected = X_train_sm[selected_features]
        model = XGBClassifier(eval_metric=self.eval_metric, random_state=self.random_state)
        model.fit(X_train_selected, y_train_sm)
        return model
    
    def tune_model_threshold(self, y_train, y_prob):
        """Tune the classification threshold based on the specified metric."""
        thresholds = np.arange(0, 1.01, 0.01)
        best_thresh = 0.5
        best_score = 0

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_train, y_pred, average='binary', zero_division=0
            )

            if self.tune_metric == 'f1' and f1 > best_score:
                best_score = f1
                best_thresh = thresh

        return best_thresh, best_score

    def test_model(self, model, X_test, y_test, selected_features):
        """Evaluate the model on the test set and tune the threshold."""
        X_test_selected = X_test[selected_features]
        y_prob = model.predict_proba(X_test_selected)[:, 1]
        best_threshold, best_metric_score = self.tune_model_threshold(y_test, y_prob)
        print(f"Best threshold by f1: {best_threshold:.2f} with score: {best_metric_score:.3f}")
        y_pred_tuned = (y_prob >= best_threshold).astype(int)
        return y_pred_tuned, y_prob
    
    def _execute(self):
        # --- Train/Test Split ---
        X_train, X_test, y_train, y_test = self.split_data()
        
        # --- SMOTE on Training Only ---
        X_train_sm, y_train_sm = self.smote_resample(X_train, y_train)
        
        # --- RFE with XGBoost: select top features ---
        selected_features = self.feature_selection_rfe(X_train_sm, y_train_sm)
        
        # --- Train Final Model on Selected Features ---
        model = self.train_model(X_train_sm, y_train_sm, selected_features)
        
        # --- Evaluate on Test Set ---
        y_pred_tuned, y_prob = self.test_model(model, X_test, y_test, selected_features)
        
        print("Tuned threshold model performance:")
        print(classification_report(y_test, y_pred_tuned))
        print("ROC AUC:", roc_auc_score(y_test, y_prob))
        
        return model, selected_features, X_test, y_test, y_pred_tuned, y_prob
