def get_se(x):
    """
    Function to calculate the Standart error for an avr/proportion in the array-like data structure
    :param x: an array to calculate the standart error for
    :return: Standart error
    """
    return np.std(x) / np.sqrt(len(x))

def get_ci_95(x):
    ci_upper = np.mean(x) + 1.96*get_se(x)
    ci_lower = np.mean(x) - 1.96*get_se(x)
    return {"ci_lower": ci_lower,
            "ci_upper": ci_upper}