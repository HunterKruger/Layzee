from abc import abstractmethod
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats.stats import pearsonr
from scipy.stats import shapiro
from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score, \
    accuracy_score, hamming_loss, matthews_corrcoef, confusion_matrix, roc_curve, explained_variance_score, \
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, classification_report


class Evaluation:
    """
    Interpret model results in different aspects for regression and classification tasks.
    """

    def __init__(self, y_score, y_true, return_result=False):
        """
        Constructor.
        :param y_score: target prediction, 1-D np.array or Series
        :param y_true: ground truth of target, 1-D np.array or Series
        :param return_result: return result in dict if True
        """
        self.y_score = np.array(y_score)
        self.y_true = np.array(y_true)
        self.return_result = return_result

    @abstractmethod
    def detailed_metrics(self):
        pass

    @staticmethod
    def feature_importance(model, features, top_n=10):
        """
        Plot feature importance of a tree-based model
        :param features: list of features, the order matters
                eg: use X_train.columns.tolist()
        :param model: a tree-based model object such as Random Forest, XGBoost
        :param top_n: plot top n important features
        """
        pack = sorted(zip(features, model.feature_importances_.tolist()), key=lambda tup: tup[1], reverse=True)
        data, idx = zip(*pack)
        ss = pd.Series(data=data, index=idx)
        sns.barplot(y=ss.values[:top_n], x=ss.index[:top_n], orient='h')

    @staticmethod
    def coefficients_intercept(model, features):
        """
        Plot coefficients and intercept of a (quasi) linear model
        :param features: list of features, the order matters
                eg: use X_train.columns.tolist()
        :param model: a linear model object such as Linear Regression, Logistic Regression
        """
        print('intercept=' + str(model.intercept_[0]))
        pack = sorted(zip(features, model.coef_.tolist()[0], [abs(x) for x in model.coef_.tolist()[0]]),
                      key=lambda tup: tup[2], reverse=True)
        idx, data, data_abs = zip(*pack)
        ss = pd.Series(data=data, index=idx)
        sns.barplot(x=ss.values, y=ss.index)


class RegEvaluation(Evaluation):

    def __init__(self, y_score, y_true, return_result=False):
        """
        Constructor.
        :param y_score: target prediction, 1-D np.array or Series
        :param y_true: ground truth of target, 1-D np.array or Series
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
        dummy = [x for x in range(int(minimum) - 1, int(maximum) + 1, 1)]
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
        print('--------------------------------------')
        shapiro_test = shapiro(error_clipped)
        print('W test statistic: ' + str(shapiro_test.statistic))
        print('p-value: ' + str(shapiro_test.pvalue))
        if shapiro_test.pvalue >= 0.05:
            print('<Error is normally distributed> cannot be rejected.')
        else:
            print('Error is not normally distributed.')
        print('--------------------------------------')
        print('Error plot & QQ plot for 2% ~ 98% quantile')
        sns.displot(data=error_clipped)
        sm.qqplot(error_clipped, line='45', fit=True)
        plt.show()

        if self.return_result:
            return to_display


class BinClsEvaluation(Evaluation):

    def __init__(self, y_score, y_true, return_result=False):
        """
        Constructor.
        :param y_score: target prediction, a score or probability from 0.0 to 1.0, in list or Series
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

    def confusion_matrix(self, cutoff=None, simple=False, cost_matrix=False, cost_tp=1, cost_tn=0, cost_fp=-0.3,
                         cost_fn=0):
        """
        Confusion matrix, cost matrix and related metrics.
        :param cutoff: cut-off threshold of y_pred
        :param simple: display simplified result
        :param cost_matrix: calculate & display cost matrix
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

        if cost_matrix:
            to_display['gain_tn'] = to_display['tn'] * cost_tn
            to_display['gain_tp'] = to_display['tp'] * cost_tp
            to_display['gain_fn'] = to_display['fn'] * cost_fn
            to_display['gain_fp'] = to_display['fp'] * cost_fp
            to_display['gain_all'] = to_display['gain_tn'] + to_display['gain_tp'] + to_display['gain_fn'] + to_display[
                'gain_fp']
            to_display['gain_per_record'] = to_display['gain_all'] / len(self.y_true)

        to_display['acc'] = accuracy_score(self.y_true, self.y_score > cutoff)
        to_display['pcs'] = precision_score(self.y_true, self.y_score > cutoff)
        to_display['rec'] = recall_score(self.y_true, self.y_score > cutoff)
        to_display['f1'] = f1_score(self.y_true, self.y_score > cutoff)

        print('Cutoff = ' + str(cutoff))
        print('--------Confusion Matrix-----------')
        cols = ['Predicted_1', 'Predicted_0']
        idx = ['Actual_1', 'Actual_0']
        cm = confusion_matrix(self.y_true, self.y_score > cutoff)
        record_count = pd.DataFrame(data=cm, columns=cols, index=idx)
        print(record_count)
        print('-----------------------------------')

        if not simple:
            cols = ['Predicted_1', 'Predicted_0']
            idx = ['Actual_1%', 'Actual_0%']
            cm2 = confusion_matrix(self.y_true, self.y_score > cutoff, normalize='true')
            pct_actual = pd.DataFrame(data=np.round(cm2, 2), columns=cols, index=idx)
            print(pct_actual)
            print('-----------------------------------')
            cols = ['Predicted_1%', 'Predicted_0%']
            idx = ['Actual_1', 'Actual_0']
            cm3 = confusion_matrix(self.y_true, self.y_score > cutoff, normalize='pred')
            pct_pred = pd.DataFrame(data=np.round(cm3, 2), columns=cols, index=idx)
            print(pct_pred)

        if cost_matrix:
            print('-------------Cost Matrix-----------')
            print('If model predict 1 and value 1, the gain is ' +
                  str(cost_tp) + ' x ' + str(to_display['tp']) + ' = ' + str(to_display['gain_tp']))
            print('If model predict 1 and value 0, the gain is ' +
                  str(cost_fp) + ' x ' + str(to_display['fp']) + ' = ' + str(to_display['gain_fp']))
            print('If model predict 0 and value 1, the gain is ' +
                  str(cost_fn) + ' x ' + str(to_display['fn']) + ' = ' + str(to_display['gain_fn']))
            print('If model predict 0 and value 0, the gain is ' +
                  str(cost_tn) + ' x ' + str(to_display['tn']) + ' = ' + str(to_display['gain_tn']))
            print('Average gain per record ' + str(np.round(to_display['gain_per_record'], 2)) + ' x ' + str(
                len(self.y_true)) + ' = ' + str(to_display['gain_all']))
            print('-----------------------------------')

        ss = pd.Series(data=[to_display['acc'], to_display['pcs'], to_display['rec'], to_display['f1']],
                       index=['Accuracy', 'Precision', 'Recall', 'F1_score'])
        sns.barplot(x=ss.values, y=ss.index, orient='h')
        plt.xlim(0, 1)

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
        model_auc = roc_auc_score(self.y_true, self.y_score)
        # summarize scores
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
        plt.xlim(0, 1)

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

    def __init__(self, y_score, y_true, labels, y_proba=None, return_result=False):
        """
        Constructor.
        :param y_score: target prediction, encoded, list or Series of int
        :param y_true: ground truth of target, encoded, list or Series of int
        :param labels: list of labels, the order matters; or a dict with key as label and value as encoded value.
                eg: labels = ['Red','Green','Yellow'], encoded to [0,1,2] by default
                    labels = {'Red':0, 'Green':1, 'Yellow':2}
        :param y_proba:
                2-d array like, shape = (n_samples, n_classes)
                    eg:    Red | Green | Yellow
                           ---------------------
                           0.1 | 0.4   | 0.5
                           0.2 | 0.5   | 0.3
                           0.3 | 0.1   | 0.6
        :param return_result: return result in dict if True
        """
        super().__init__(y_score, y_true, return_result)

        if isinstance(labels, dict):
            self.label2num = labels
            self.labels = [k for k, v in labels.items()]
        elif isinstance(labels, list):
            self.labels = labels
            self.label2num = dict()
            for idx, lb in enumerate(labels):
                self.label2num[lb] = idx
        else:
            raise ValueError('Wrong input type for y_proba')

        self.num2label = {v: k for k, v in self.label2num.items()}
        self.y_proba = y_proba

    def confusion_matrix(self, simple=True):
        """
        Confusion matrix.
        :param simple: display simplified result
        :return:
        """
        print('--------Confusion Matrix-----------')
        cols = ['Predicted_' + str(x) for x in self.num2label.values()]
        idx = ['Actual_' + str(x) for x in self.num2label.values()]
        cm = confusion_matrix(self.y_true, self.y_score)
        record_count = pd.DataFrame(data=cm, columns=cols, index=idx)
        print(record_count)
        print('------------------------------')

        if not simple:
            cols = ['Predicted_' + str(x) for x in self.num2label.values()]
            idx = ['Actual%_' + str(x) for x in self.num2label.values()]
            cm2 = confusion_matrix(self.y_true, self.y_score, normalize='true')
            pct_actual = pd.DataFrame(data=np.round(cm2, 2), columns=cols, index=idx)
            print(pct_actual)
            print('------------------------------')
            cols = ['Predicted%_' + str(x) for x in self.num2label.values()]
            idx = ['Actual_' + str(x) for x in self.num2label.values()]
            cm3 = confusion_matrix(self.y_true, self.y_score, normalize='pred')
            pct_pred = pd.DataFrame(data=np.round(cm3, 2), columns=cols, index=idx)
            print(pct_pred)
            print('------------------------------')
            print(classification_report(self.y_true, self.y_score, target_names=self.labels))

    def calibration_curve(self, label, bins=10):
        """
        :param label: label name, not encoded
        :param bins: number of bins
        """
        if self.y_proba is not None:
            y_score_temp = self.y_proba[:, self.label2num[label]]  # proba of label
            # 1 is label, 0 is not label
            y_true_temp = [1 if num == self.label2num[label] else 0 for num in self.y_true]
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true_temp, y_score_temp, n_bins=bins)
            dummy = [x / 100 for x in range(0, 101, 1)]
            print('Class: ' + label)
            plt.plot(dummy, dummy, linestyle='--', label='Perfectly calibrated')
            plt.plot(mean_predicted_value, fraction_of_positives, marker='.', label="Model")
            plt.xlabel('Average of Predicted Probability for Positive Class')
            plt.ylabel('Frequency of Positive Class')
            plt.legend()
            plt.show()
        else:
            print('Input predicted probability when initiating this class.')

    def roc_curve(self, label):
        """
        :param label: label name, not encoded
        """
        if self.y_proba is None:
            print('Input predicted y probability when initiating this class.')
        else:
            ns_probs = [0 for _ in range(len(self.y_true))]
            # calculate scores
            y_score_temp = self.y_proba[:, self.label2num[label]]  # proba of label
            y_true_temp = [1 if num == self.label2num[label] else 0 for num in self.y_true]

            model_auc = roc_auc_score(y_true_temp, y_score_temp)
            # summarize scores
            print('Class: ' + label)
            print('Model: ROC AUC=%.3f' % model_auc)
            # calculate roc curves
            ns_fpr, ns_tpr, _ = roc_curve(y_true_temp, ns_probs)
            model_fpr, model_tpr, _ = roc_curve(y_true_temp, y_score_temp)
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

    def density_chart(self, label):
        """
        :param label: label name, not encoded
        """
        # 1 is label, 0 is not label
        y_score_temp = [1 if num == self.label2num[label] else 0 for num in self.y_score]
        y_true_temp = [1 if num == self.label2num[label] else 0 for num in self.y_true]

        df_plot = pd.DataFrame(data=[y_score_temp, y_true_temp]).T
        df_plot.columns = ['proba', 'label']
        print('1 for ' + label + ', 0 for not ' + label)
        sns.displot(data=df_plot, x="proba", hue="label", kind='kde')
        plt.xlim(0, 1)

    def detailed_metrics(self):
        to_display = dict()
        to_display['Accuracy'] = accuracy_score(self.y_true, self.y_score)
        to_display['Precision'] = precision_score(self.y_true, self.y_score, average='macro')
        to_display['Recall'] = recall_score(self.y_true, self.y_score, average='macro')
        to_display['F1_score'] = f1_score(self.y_true, self.y_score, average='macro')
        to_display['Hamming_Loss'] = hamming_loss(self.y_true, self.y_score)

        if self.y_proba is not None:
            to_display['Log_loss'] = log_loss(self.y_true, self.y_proba)
            to_display['Auc_Roc'] = roc_auc_score(self.y_true, self.y_proba, multi_class='ovr')

        for k, v in to_display.items():
            print(str(k) + ': ' + str(v))

        if self.return_result:
            return to_display
