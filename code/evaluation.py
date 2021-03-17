from abc import abstractmethod
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats.stats import pearsonr
from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score, \
    accuracy_score, hamming_loss, matthews_corrcoef, confusion_matrix, roc_curve, \
    explained_variance_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


class Evaluation:
    def __init__(self, y_score, y_true, return_result=False):
        """
        Constructor.
        :param y_score: target prediction
        :param y_true: ground truth of target
        :param return_result: return result in dict if True
        """
        self.y_score = y_score
        self.y_true = y_true
        self.return_result = return_result

    @abstractmethod
    def detailed_metrics(self):
        pass


class RegEvaluation(Evaluation):
    def __init__(self, y_score, y_true, return_result=False):
        """
        Constructor.
        :param y_score: target prediction, in list or Series
        :param y_true: ground truth of target, in list or Series
        :param return_result: return result in dict if True
        """
        super().__init__(y_score, y_true, return_result)

    def detailed_metrics(self):
        to_display = dict()
        to_display['Explained_Variance_Score'] = explained_variance_score(self.y_true, self.y_score)
        to_display['Mean_Absolute_Error'] = mean_absolute_error(self.y_true, self.y_score)
        to_display['Mean_Absolute_Percentage_Error'] = mean_absolute_percentage_error(self.y_true, self.y_score)
        to_display['Mean_Squared_Error'] = mean_squared_error(self.y_true, self.y_score)
        to_display['Root_Mean_Squared_Error'] = to_display['Mean_Squared_Error'] ** 0.5
        to_display['Pearson_Coefficient'] = pearsonr(self.y_true, self.y_score)[0]
        to_display['R2_Score'] = r2_score(self.y_true, self.y_score)

        for k, v in to_display.items():
            print(str(k) + ': ' + str(v))

        if self.return_result:
            return to_display

    def scatter_plot(self):
        minimum = min(min(self.y_true), min(self.y_score))
        maximum = max(max(self.y_true), max(self.y_score))
        dummy = [x for x in range(minimum, maximum, 1)]
        plt.plot(dummy, dummy, linestyle='--', label='Diagonal Line')
        plt.scatter(self.y_true, self.y_score, marker='.', label='Model', c='orange')
        # axis labels
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()

    def error_distribution(self):
        error = np.array(self.y_score - self.y_true)
        to_display = dict()
        to_display['min_raw'] = np.min(error)
        to_display['max_raw'] = np.max(error)
        to_display['min_clipped'] = np.quantile(error, q=0.02)
        to_display['max_clipped'] = np.quantile(error, q=0.98)
        error_clipped = np.array([x for x in error if to_display['max_clipped'] >= x >= to_display['min_clipped']])
        to_display['q25'] = np.quantile(error_clipped, q=0.25)
        to_display['q75'] = np.quantile(error_clipped, q=0.75)
        to_display['median'] = np.median(error_clipped)
        to_display['avg'] = np.average(error_clipped)
        to_display['std'] = np.std(error_clipped)

        for k, v in to_display.items():
            print(str(k) + ': ' + str(v))

        sns.displot(data=error_clipped)

        if self.return_result:
            return to_display


class BinClsEvaluation(Evaluation):

    def __init__(self, y_score, y_true, return_result=False):
        """
        Constructor.
        :param y_score: target prediction, a score from 0.0 to 1.0, in list or Series
        :param y_true: ground truth of target, 0 or 1, in list or Series
        :param return_result: return result in dict if True
        """
        super().__init__(y_score, y_true, return_result)
        self.result = dict()
        self.best_cutoff, self.decision_table = self.get_best_cutoff()

    def get_best_cutoff(self):
        """
        Find the best cutoff and the decision table
        """
        cutoff = []
        accuracy = []
        precision = []
        recall = []
        f1 = []

        for value in range(0, 102, 2):
            # print(value/100)
            cutoff.append(value / 100)
            accuracy.append(accuracy_score(self.y_true, self.y_score > value / 100))
            precision.append(precision_score(self.y_true, self.y_score > value / 100))
            recall.append(recall_score(self.y_true, self.y_score > value / 100))
            f1.append(f1_score(self.y_true, self.y_score > value / 100))

        best_cutoff = cutoff[np.argmax(f1)]

        decision_table = pd.DataFrame([accuracy, precision, recall, f1]).T
        decision_table.index = cutoff
        decision_table.columns = ['accuracy', 'precision', 'recall', 'f1']

        return best_cutoff, decision_table

    def confusion_matrix(self, cutoff=None, cost_tp=1, cost_tn=0, cost_fp=-0.3, cost_fn=0):
        """
        Confusion matrix, cost matrix and related metrics.
        :param cutoff: cut-off threshold of y_pred
        :param cost_tp: cost of tp
        :param cost_tn: cost of tn
        :param cost_fp: cost of fp
        :param cost_fn: cost of fn
        :return:
        """
        if cutoff is None:
            cutoff = self.best_cutoff

        to_display = dict()
        to_display['cutoff'] = cutoff

        to_display['tn'], to_display['fp'], to_display['fn'], to_display['tp'] = \
            confusion_matrix(self.y_true, self.y_score > cutoff).ravel()

        to_display['gain_tn'] = to_display['tn'] * cost_tn
        to_display['gain_tp'] = to_display['tp'] * cost_tp
        to_display['gain_fn'] = to_display['fn'] * cost_fn
        to_display['gain_fp'] = to_display['fp'] * cost_fp
        to_display['gain_all'] = to_display['gain_tn'] + to_display['gain_tp'] + to_display['gain_fn'] + to_display[
            'gain_fp']
        to_display['gain_per_record'] = to_display['gain_all'] / len(self.y_true)

        acc = accuracy_score(self.y_true, self.y_score > cutoff)
        pcs = precision_score(self.y_true, self.y_score > cutoff)
        rec = recall_score(self.y_true, self.y_score > cutoff)
        f1 = f1_score(self.y_true, self.y_score > cutoff)

        for k, v in to_display.items():
            print(str(k) + ': ' + str(v))

        data = pd.Series(data=[acc, pcs, rec, f1], index=['accuracy', 'precision', 'recall', 'f1 score'])
        sns.barplot(data.values, data.index, orient='h')

        if self.return_result:
            return to_display

    def decision_chart(self):
        """
        Plot accuracy, recall, precision, f1_score for each cutoff.
        :return:
        """
        sns.lineplot(data=self.decision_table)

    def roc_curve(self):
        """
        Plot roc curve.
        """
        ns_probs = [0 for _ in range(len(self.y_true))]
        # calculate scores
        ns_auc = roc_auc_score(self.y_true, ns_probs)
        model_auc = roc_auc_score(self.y_true, self.y_score)
        # summarize scores
        print('No Skill: ROC AUC=%.3f' % ns_auc)
        print('Model: ROC AUC=%.3f' % model_auc)
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(self.y_true, ns_probs)
        model_fpr, model_tpr, _ = roc_curve(self.y_true, self.y_score)
        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(model_fpr, model_tpr, marker='.', label='Model')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()

        if self.return_result:
            return model_auc

    def density_chart(self):
        df_plot = pd.DataFrame(data=[self.y_score, self.y_true]).T
        df_plot.columns = ['proba', 'label']
        sns.displot(data=df_plot, x="proba", hue="label", kind='kde')

    def calibration_curve(self, bins=10):
        fraction_of_positives, mean_predicted_value = calibration_curve(self.y_true, self.y_score, n_bins=bins)
        dummy = [x / 100 for x in range(0, 101, 1)]
        plt.plot(dummy, dummy, linestyle='--', label='Perfectly calibrated')
        plt.plot(mean_predicted_value, fraction_of_positives, marker='.', label="Model")
        plt.xlabel('Average of Predicted Probability for Positive Class')
        plt.ylabel('Frequency of Positive Class')
        plt.legend()
        plt.show()

    def detailed_metrics(self, cutoff=None):
        """
        Evaluation for a binary classification task
        :param cutoff: cut-off threshold of y_pred
        """
        if cutoff is None:
            cutoff = self.best_cutoff

        to_display = dict()

        # threshold independent
        to_display['Auc_Roc'] = roc_auc_score(self.y_true, self.y_score)
        to_display['Log_Loss'] = log_loss(self.y_true, self.y_score)
        to_display['Brier_Score_Loss'] = brier_score_loss(self.y_true, self.y_score)

        # threshold dependent
        to_display['Accuracy'] = accuracy_score(self.y_true, self.y_score > cutoff)
        to_display['Precision'] = precision_score(self.y_true, self.y_score > cutoff)
        to_display['Recall'] = recall_score(self.y_true, self.y_score > cutoff)
        to_display['F1_score'] = f1_score(self.y_true, self.y_score > cutoff)
        to_display['Hamming_Loss'] = hamming_loss(self.y_true, self.y_score > cutoff)
        to_display['Matthews_Corrcoef'] = matthews_corrcoef(self.y_true, self.y_score > cutoff)

        for k, v in to_display.items():
            print(str(k) + ': ' + str(v))

        if self.return_result:
            return to_display


class MltClsEvaluation(Evaluation):

    def detailed_metrics(self):
        pass
