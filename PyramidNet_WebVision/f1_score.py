import numpy as np

def compute_precision_recall_f1_score(confusion_matrix):
    """Compute the average precision, recall and F1-score"""
    sum_precision = 0
    sum_recall = 0
    sum_f1_score = 0
    print(confusion_matrix.shape[0])
    for i in range(confusion_matrix.shape[0]):
        print("i = " + str(i))
        true_pos = confusion_matrix[i, i]
        print("true_pos = " + str(true_pos))
        true_and_false_pos = np.sum(confusion_matrix[i, :])
        print("true_and_false_pos = " + str(true_and_false_pos))
        true_pos_and_false_neg = np.sum(confusion_matrix[:, [i]])
        print("true_pos_and_false_neg = " + str(true_pos_and_false_neg))

        if true_pos == 0:
            if true_and_false_pos == 0 and true_pos_and_false_neg == 0:
                precision = 1
                recall = 1
                f1_score = 1
            else:
                if true_pos_and_false_neg == 0:
                    precision = 1
                    recall = 0
                elif true_and_false_pos == 0:
                    precision = 0
                    recall = 1
                else:
                    precision = 0
                    recall = 0
                    
                f1_score = 0
        else:
            precision = true_pos / true_and_false_pos
            recall = true_pos / true_pos_and_false_neg
            f1_score = (2 * precision * recall) / (precision + recall)

        sum_precision += precision
        sum_recall += recall
        sum_f1_score += f1_score
        print(sum_precision, sum_recall, sum_f1_score)

    average_precision = sum_precision / confusion_matrix.shape[0]
    average_recall = sum_recall / confusion_matrix.shape[0]
    average_f1_score = sum_f1_score / confusion_matrix.shape[0]

    return average_precision, average_recall, average_f1_score

precision, recall, f1_score = compute_precision_recall_f1_score(np.array([[35, 15], [9, 91]]))
print(precision, recall, f1_score)
