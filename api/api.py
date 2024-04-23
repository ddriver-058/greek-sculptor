from flask import Flask, request, jsonify
from yahooquery import Ticker
from datetime import datetime, date, timedelta, timezone
from pandas import unique, DataFrame, concat
import pulp as plp
from math import isnan
from statistics import mean
import QuantLib as ql 

app = Flask(__name__)

# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
  return jsonify('0'), 200

@app.route('/option_chain', methods=['GET'])
def get_data():

  try:
    # Get the value of the 'symbol' query parameter
    symbol = request.args.get('symbol')

    # Check if the parameter is provided
    if symbol:
      t = Ticker(symbol)
      federal_funds_rate_ticker = Ticker('^NYOMFFX')

      # Pull option chain
      df = t.option_chain.reset_index()

      # Get price of underlying
      spot_price = t.quotes[symbol]['regularMarketPrice']

      # Get (estimated) risk-free rate by taking NYOMFFX / 100*100
      risk_free_rate = federal_funds_rate_ticker.quotes['^NYOMFFX']['regularMarketPrice'] / 10000

      # Calculate expiration, convert datetimes to be more legible
      # Assume that options expire at market close (8pm UTC)
      now = datetime.now()

      df = df.assign(
        expirationDttm = df['expiration'], # Use a separate column before strftime
        nBuys = 1,
        nSells = 0,
        nCalls = (df['optionType'] == 'calls').apply(lambda x: int(x)),
        nPuts = (df['optionType'] == 'puts').apply(lambda x: int(x)),
        inTheMoney = (df['inTheMoney']).apply(lambda x: int(x))
        #expiration = df['expiration'].apply(lambda x: x.isoformat()),
        #lastTradeDate = df['lastTradeDate'].apply(lambda x: x.isoformat())
      )
      
      df = df.assign(
        expiration = df['expirationDttm'].apply(lambda x: x.replace(tzinfo=timezone.utc).astimezone(tz=None).strftime('%Y-%m-%d')), # Make date readable
      )[[
        'expirationDttm', 
        'expiration',
        'optionType',
        'nBuys',
        'nSells',
        'nCalls',
        'nPuts', 
        'strike',
        'lastPrice',
        'volume',
        'openInterest',
        'bid',
        'ask',
        'lastTradeDate',
        'impliedVolatility',
        'inTheMoney'
      ]]

      # Calculate greeks
      # First need to filter the df to 2 > IV > 0.005 
      model_filter = df[
        (df['impliedVolatility'] > 0.005) & 
        (df['impliedVolatility'] < 2)
      ]

      def calcAmericanGreeks(row):

        # http://gouthamanbalaraman.com/blog/american-option-pricing-quantlib-python.html

        maturity_date = ql.Date(row['expirationDttm'].day, row['expirationDttm'].month, row['expirationDttm'].year)
        strike_price = row['strike']
        volatility = row['impliedVolatility'] # the historical vols or implied vols
        dividend_rate =  0
        if row['optionType'][0] == 'c':
          option_type = ql.Option.Call
        elif row['optionType'][0] == 'p':
          option_type = ql.Option.Put
        
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

        calculation_date = ql.Date(datetime.today().day, datetime.today().month, datetime.today().year)
        ql.Settings.instance().evaluationDate = calculation_date

        # eu_exercise = ql.EuropeanExercise(maturity_date)
        # european_option = ql.VanillaOption(payoff, eu_exercise)

        # Run the pricing engine
        # Since we will need to compute vega and rho manually, parameterize based on volatility and interest rate
        def get_american_option(vol, rfr):
          payoff = ql.PlainVanillaPayoff(option_type, strike_price)
          settlement = calculation_date

          am_exercise = ql.AmericanExercise(settlement, maturity_date)
          american_option = ql.VanillaOption(payoff, am_exercise)

          spot_handle = ql.QuoteHandle(
            ql.SimpleQuote(spot_price)
          )
          flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, rfr, day_count)
          )
          dividend_yield = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, dividend_rate, day_count)
          )
          flat_vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(calculation_date, calendar, vol, day_count)
          )
          bsm_process = ql.BlackScholesMertonProcess(
            spot_handle, 
            dividend_yield, 
            flat_ts, 
            flat_vol_ts
          )

          steps = 200
          binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
          american_option.setPricingEngine(binomial_engine)
          return american_option

        opt = get_american_option(volatility, risk_free_rate)

        # Need to compute vega and rho manually
        perturbation_amt = 0.01
        opt_vol_perturb = get_american_option(volatility + perturbation_amt, risk_free_rate)
        opt_rfr_perturb = get_american_option(volatility, risk_free_rate + perturbation_amt)

        vega = (opt_vol_perturb.NPV() - opt.NPV()) / perturbation_amt
        rho = (opt_rfr_perturb.NPV() - opt.NPV()) / perturbation_amt

        return [
          opt.NPV(),
          opt.delta(),
          opt.gamma(),
          opt.theta() / 365, # QuantLib returns 1y theta
          vega,
          rho
        ]

      model_out = model_filter.apply(calcAmericanGreeks, axis=1)
      model_df = DataFrame(
        model_out.to_list(),
        columns=[
          'modeledPrice',
          'delta',
          'gamma',
          'theta',
          'vega',
          'rho'
        ]
      )

      chain = concat(
        [
          model_filter.reset_index(drop=True),
          model_df.reset_index(drop=True)
        ],
        axis=1
      )

      # Filter the chain to options modeled in a valid way
      chain = chain[(chain['modeledPrice'] != 0) & 
                      (chain['delta'] != 0) &
                      (chain['gamma'] != 0) &
                      (chain['theta'] != 0) &
                      (chain['vega'] != 0) &
                      (chain['rho'] != 0)]
      
      # if the chain is heavy, filter it down by keeping high OI options, then slicing evenly
      if chain.shape[0] > 500:
        chain = chain[chain['openInterest'] >= chain['openInterest'].quantile(0.8)]
        step_size = int(chain.shape[0] / 500)
        chain = chain.iloc[::step_size]

      # chain contains BTO options. We will create a new df of STO positions
      sto_chain = chain.assign(
        nBuys = 0,
        nSells = 1,
        lastPrice = chain['lastPrice'] * -1,
        modeledPrice = chain['modeledPrice'] * -1,
        bid = chain['bid'] * -1,
        ask = chain['ask'] * -1,
        delta = chain['delta'] * -1,
        gamma = chain['gamma'] * -1,
        theta = chain['theta'] * -1,
        vega = chain['vega'] * -1,
        rho = chain['rho'] * -1
      )

      full_chain = concat(
        [
          chain.reset_index(drop=True),
          sto_chain.reset_index(drop=True)
        ],
        axis=0
      ).reset_index(drop=True)

      return full_chain.to_json(), 200
    else:
      return jsonify({'error': 'No parameter value provided'}), 400
  except Exception as e:
    return jsonify({'error': str(e)}), 500

@app.route('/portfolio', methods=['POST'])
def find_optimal_portfolio():

  try:

    if request.data:
      objective = request.json['objective']
      constraints = request.json['constraints']
      filtered_chain = request.json['filtered-chain']

      # Define the optimization problem in PULP.
      # Each variable x will represent an option (row) in the filtered_chain.
      # Each row, e.g., each constraint, will represent an option feature (column) in the filtered_chain.
      # Hence, each constraint row represents one feature (e.g., delta) in the chosen options
      # (with the choices represented by x).

      # Convert chain to df. we will assign an index, 'i', to preserve the order at this time.
      chain_df = DataFrame.from_dict(filtered_chain)
      chain_df = chain_df.assign(
        i = range(chain_df.shape[0])
      )

      # Generate option choice vars (x), 1 for each row.
      choice_vars = [plp.LpVariable(f'{i}', lowBound=0, cat='Integer') for i in chain_df['i']]

      # Form objective function.
      # initialize based on user input
      opt_prob = {
        'Maximize': plp.LpProblem('Maximize', plp.LpMaximize),
        'Minimize': plp.LpProblem('Minimize', plp.LpMinimize)
      }[objective['verb']]

      # Add the objective variables.
      # We will use c1x1+c2x2+..., where c1,c2,... are the specified feature values for each option x.

      # Multiply each choice var by the value being optimized, then sum to get objective
      opt_prob += chain_df[objective['variable']].mul(choice_vars).sum()
      print('added objective LHS')

      # Now define constraints
      # Each constraint is a particular feature * the choice vars [operator] [operand]
      for key in constraints:
        c = constraints[key]

        # Look up operator
        op = {
          '>=': plp.LpConstraintGE,
          '<=': plp.LpConstraintLE,
          '=': plp.LpConstraintEQ
        }[c['operator']]

        opt_prob += plp.LpConstraint(
          e=chain_df[c['var']].mul(choice_vars).sum(),
          sense=op, 
          rhs=c['operand']
        )
        print('added constraint LHS')

      print('starting solver')
      opt_prob.solve(plp.PULP_CBC_CMD(msg=1, timeLimit=30, threads=2))

      # Get solution status
      prob_status = {
        1: 'Optimal - the solver found an optimal solution.',
        0: 'Not Solved - the solver found a solution, but it isn\'t optimal.',
        -1: 'Infeasible - the problem likely has conflicting constraints that make it impossible to solve.',
        -2: 'Unbounded - the solver thinks the optimal solution goes to infinity. Try setting more constraints, such as on nBuys and nSells, or on the objective variable.',
        -3: 'Undefined - the solver couldn\'t receive the problem correctly -- this is an internal app issue.'
      }[opt_prob.status]

      # Return the choice variable values and objective value
      out = {
        'objective': opt_prob.objective.value(),
        'choices': {},
        'problem_status_code': opt_prob.status,
        'problem_status_desc': prob_status
      }

      # Here, v.name corresponds to the 'i' variable in the returned data
      for v in opt_prob.variables():
        out['choices'][v.name] = v.varValue
      
      if opt_prob.status in [-1, -2, -3]:
        portfolio = {}
      elif opt_prob.status in [1, 0]:
        # We will also generate the portfolio here
        # Ensure chain_df has original order
        chain_df = chain_df.sort_values(by='i')

        non_zero_choices = [
          chain_df.iloc[[int(key)]].assign(qty = val) for key, val in out['choices'].items() if val > 0
        ]

        if len(non_zero_choices) == 0:
          out['problem_status_code'] = 0
          out['problem_status_desc'] = 'Given the CPU resources available, the solver found a null solution of 0 choices. Try again later, or relax the constraints.'
          portfolio = {}
        else:
          portfolio = concat(non_zero_choices, axis=0).to_dict()

      out['portfolio'] = portfolio

      return jsonify(out), 200
  except Exception as e:
    return jsonify({'error': str(e)}), 500


# @app.route('/pl-table', methods=['POST'])
# def form_pl_table():
#   try:
#     if request.json:
#       model_df = DataFrame.from_dict(request.json['model_output'])
#       symbol = request.json['symbol']

#       # get spot price
#       t = Ticker(symbol)
#       spot = t.quotes[symbol]['regularMarketPrice']

#       strat = []
#       for i in range(model_df.shape[0]):
#         row = model_df.iloc[i]
#         if row['nCalls'] == 1:
#           op_type = 'call'
#         elif row['nPuts'] == 1:
#           op_type = 'put'
        
#         if row['nBuys'] == 1:
#           tr_type = 'buy'
#         elif row['nSells'] == 1:
#           tr_type = 'sell'
#           row['lastPrice'] *= -1
        
#         dt = row['expiration']

#         strat.append({
#           'type': op_type,
#           'strike': row['strike'],
#           'premium': row['lastPrice'],
#           'n': row['qty']*100,
#           'action': tr_type,
#           'expiration': dt
#         })

#       # The final date will be the date of the soonest expiry
#       final_dt = datetime.strptime(min(model_df['expiration']), '%Y-%m-%d').date()
#       start_dt = date.today()
#       n_days = (final_dt - start_dt).days

#       pl_out = []

#       for i in range(0, n_days):
#         target_dt = start_dt + timedelta(days=i+1)

#         inputs = Inputs(
#           stock_price=spot,
#           start_date=start_dt,
#           target_date=target_dt,
#           volatility=mean(model_df['impliedVolatility']) / 100,
#           interest_rate=0.0009,
#           min_stock=spot - round(spot * 0.5, 2),
#           max_stock=spot + round(spot * 0.5, 2),
#           strategy=strat,
#           compute_expectation=True
#         )

#         st = StrategyEngine(inputs)
#         out = st.run()

#         pl_out.append({
#           'spot_price': out.data.stock_price_array.tolist(),
#           'profit': out.data.profit.tolist()[0],
#           'dt': target_dt,
#           'strategy_cost': out.strategy_cost,
#           'max_profit_in_domain': out.maximum_return_in_the_domain,
#           'min_profit_in_domain': out.minimum_return_in_the_domain,
#           'average_profit_mc': out.average_profit_from_mc,
#           'average_loss_mc': out.average_loss_from_mc
#         })
      
#       return jsonify(pl_out), 200
#   except:
#     return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')
