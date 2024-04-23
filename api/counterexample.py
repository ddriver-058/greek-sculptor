# returns positive delta.
from optlib.gbs import black_scholes, american

american(
    'p', # contract type
    7.0298, # spot price
    5.0, # strike price
    0.003360421041286149, # TTE,
    0.0549, # risk-free rate
    0, # dividend yield
    1.75 # IV
)

# whereas this doesn't.

black_scholes(
    'p', # contract type
    7.0298, # spot price
    5.0, # strike price
    0.003360421041286149, # TTE,
    0.0549, # risk-free rate
    1.75 # IV
)