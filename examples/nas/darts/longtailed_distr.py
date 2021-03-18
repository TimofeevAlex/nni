import numpy as np
np.random.seed = 0

def get_class_i(x, y, i, n_reduce=None):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:,0])
    if n_reduce is not None:
        pos_i = pos_i[:n_reduce]
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i


def reduce_classes_dbset_longtailed(db_set, permute=True, lt_factor=None):
    """ Accepts a trainset torch db (which is assumed to have the same 
        number of samples in all classes) and creates a long-tailed 
        distribution with factor reduction factor. """
    db1 = db_set.data
    if not hasattr(db_set, 'targets'):
        db_set.targets = db_set.labels
    lbls = db_set.targets
    n_classes = int(np.max(db_set.targets)) + 1
    n_samples_class = int(db1.shape[0] // n_classes)
    # # create the undersampled lists of data and samples.
    data, classes = [], []
    for cl_id in range(n_classes):
        n_reduce = int(n_samples_class * lt_factor ** cl_id)
        samples = get_class_i(db1, lbls, cl_id, n_reduce=n_reduce)
        data.append(samples)
        classes.extend([cl_id] * n_reduce)
        print(n_reduce, len(data))
    # # convert into numpy arrays.
    data, classes =  np.concatenate(data, axis=0), np.array(classes, dtype=np.int)
    if permute:
        # # optionally permute the data to avoid having them sorted.
        permut1 = np.random.permutation(len(classes))
        data, classes = data[permut1], classes[permut1]
    return data, classes

