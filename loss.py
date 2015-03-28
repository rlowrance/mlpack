'''loss functions

COMMON API
function(optional_args): returns two functions
    derivative(prediction, label) -> number
    loss(prediction, label) -> number

ARGS
prediction: number
label     : number

Loss functions
hinge()
log()
quantile(tau)
squared()
squared_vw()
'''
from loss_hinge import hinge
from loss_log import log
from loss_quantile import quantile
from loss_squared import squared
from loss_squared_vw import squared_vw


# use all imported functions, so as to avoid an error message from pyflakes
if False:
    hinge()
    log()
    quantile()
    squared()
    squared_vw()


# alternative names (mimicing vw)
# BUT note these differences:
# vw's classic is our squared
# vw's squared is our squared_vw
huber = squared
logistic = log


def absolute():
    quantile(.5)
