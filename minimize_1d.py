'''minimizers for functions of 1 variable

FUNCTIONS
golden_section(fun, low, high, interval_size) -> low*, high*

ARGS
fun(x) -> number
low: number
high: number
interval_size: search stops when high* - low* < interval_size
'''
from minimize_1d_golden_section import golden_section

if False:
    # use imports, to avoid pyflakes warning
    golden_section()
