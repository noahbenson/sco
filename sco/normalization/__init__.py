####################################################################################################
# sco/normalization/__init__.py
# Second-order-contrast normalization and nonlinearity application
# By Noah C. Benson

from .core  import (Kay2013_pRF_sigma_slope, Kay2013_output_nonlinearity, Kay2013_SOC_constant,
                    calc_Kay2013_SOC_normalization, calc_Kay2013_output_nonlinearity)
from ..core import calc_chain

normalization_chain = (('calc_SOC_normalization',   calc_Kay2013_SOC_normalization),
                       ('calc_output_nonlinearity', calc_Kay2013_output_nonlinearity))

calc_normalization = calc_chain(normalization_chain)