def plot_and_save_confusion_matrix(cm, classes,
                                   normalize=False,
                                   title='Confusion matrix',
                                   cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float')
        non_zero_indices = (cm.sum(axis=1) > 0)
        cm[non_zero_indices] = cm[non_zero_indices] / cm[non_zero_indices].sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(0, len(classes), 10)
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)

    # thresh = cm.max() / 2.

    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, '',
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual labels')
    plt.xlabel('Predicted labels')
    plt.tight_layout()
    plt.savefig(FLAGS.eval_dir + "confusion_matrix.png")

