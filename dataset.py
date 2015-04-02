'''generate artificial data sets

SUMMARY OF APIs
bishop_ch1(num_samples)
d1(fun, num_samples, x_low, x_high, error_mean, error_variance)
outliers(num_features, num_unique_samples)
sparse(num_features, num_samples)
'''
from dataset_bishop_ch1 import bishop_ch1
from dataset_d1 import d1
from dataset_outliers import outliers
from dataset_sparse import sparse

if False:
    # use the imports, so as to avoid an error from pyflakes
    bishop_ch1()
    d1()
    outliers()
    sparse()
