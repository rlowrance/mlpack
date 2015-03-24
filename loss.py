'''loss functions

COMMOND API
function(optional_args): returns two functions
    output(prediction, expected) -> number
    derivative(prediction, expected) -> number

Loss functions
hinge()
log()
quantile(tau)
squared()
squared_vw()
'''
import loss_hinge
import loss_log
import loss_quantile
import loss_squared
import loss_squared_vw

hinge = loss_hinge.hinge
log = loss_log.log
quantile = loss_quantile.quantile
squared = loss_squared.squared
squared_vw = loss_squared_vw.squared_vw
