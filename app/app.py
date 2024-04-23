from dash import Dash, dcc, html, Input, Output, State, callback, dash_table, no_update, ALL, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
from requests import get, post
from io import StringIO
from time import time
from datetime import datetime
from statistics import mean
from flask import request

# Global variables
API_ROOT='http://my-greek-sculptor-service.my-greek-sculptor-ns.svc.cluster.local:5000'
# API_ROOT='http://api:5000'


app = Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR])
app.title = 'GreekSculptor'
app._favicon = 'favicon.svg'

app.layout = html.Div([
  dcc.Location(id='url', refresh=False),
  dcc.Store(id = 'app-store'),
  dcc.Store(id = 'model-output'), # Seems to be necessary due to callback mechanics?
  dbc.Row(
    [
      dbc.Col(
        dbc.Card(
          [
            dbc.NavbarSimple(
              children=[
                dbc.NavItem(dbc.NavLink('About This App', href = '/about')),
                dbc.NavItem(dbc.NavLink('Query & Filter Chain', href='/query')),
                dbc.NavItem(dbc.NavLink('Optimize', href='/optimize')),
                dbc.NavItem(dbc.NavLink('Portfolio Metrics', href='/metrics'))
              ],
              brand='GreekSculptor',
              brand_href='/about',
              color='primary',
              dark=True,
            ),
            html.Div(
              id = 'card-body'
            )
          ]
        ),
        width = 12
      )
    ]
  )
  
])

### CALLBACKS ###

# Routing callback
@app.callback(
    Output('card-body', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    # Get the User-Agent header from the HTTP request
    user_agent = request.user_agent.string
    
    # Check if the User-Agent header indicates a mobile device
    is_mobile = user_agent and any(agent in user_agent.lower() for agent in ['iphone', 'android', 'mobile'])

    if pathname == '/' or pathname == '/about':
      return dbc.CardBody([
        html.P(html.B('Options Optimization Through Linear Programming')),
        html.P('If you have ever wondered about questions like: '),
        html.P('  - What is the most theta I can get for a delta-neutral position on T?'),
        html.P('  - What delta-positive, theta-positive positions for available for AAPL?'),
        html.P('  - What is the maximum number of AMD call options can I buy while remaining under 500 delta?'),
        html.P('Then this app will help you answer those questions by making optimal selections from the option chain of a chosen US stock symbol.'),
        html.P([
          'It works by using ', 
          html.A('Linear Programming ', href='https://en.wikipedia.org/wiki/Linear_programming', target = '_blank'),
          'via ',
          html.A('PULP ', href = 'https://github.com/coin-or/pulp', target = '_blank'),
          'to optimize for a selection vector of length equal to the number of options in the chain, where each element represents the number of positions ',
          'opened for the corresponding option.'
        ]),
        html.P([
          'It also makes use of ',
          html.A('yahooquery ', href = 'https://pypi.org/project/yahooquery', target = '_blank'),
          'and ',
          html.A('QuantLib ', href = 'https://pypi.org/project/QuantLib/', target = '_blank'),
          'to pull option chains from Yahoo Finance and to model the price and Greeks of American options, respectively.'
        ]),
        html.P([
          'This tool is provided for entertainment and educational purposes only. It is not intended for profit-making activities or as financial ',
          'advice. Options trading involves significant risk, including the potential for unlimited loss. Users of this tool should be ', 
          'aware of the complexity and inherent risks associated with options trading and should exercise caution and diligence in their decision-making.'
        ]),
        html.P([
          'This tool also has no affiliation with Yahoo Finance.'
        ]),
        html.Br(),
        html.P([
          html.B('How to Use')
        ]),
        html.P([
          'You should start in the Query tab by querying data for a US stock symbol. Then, you can apply column filters as desired to change the solver\'s input dataset.'
        ]),
        html.P([
          'Then, in the Optimize tab, select an objective by choosing a verb (maximize or minimize) and a variable from the chain. ',
          'You can then define constraints, which is always recommended to avoid unbounded problems, by clicking the + button, then ',
          'choosing a variable, an operator, and a desired value. E.g., `delta = 0`.  Then click optimize.'
        ]),
        html.P([
          'Once the solver is done, review the model output status, then move on to the Portfolio Metrics tab. You can review the choices ',
          'and the greeks of the solution, then access a link to view the strategy in OptionStrat.'
        ]),
        html.P([
          'Please note that this tool has no affiliation with OptionStrat.'
        ])
      ])
    if pathname == '/query':
        return dbc.CardBody(
          [
            html.H4('Choose Symbol', className='card-title'),
            html.P('Pull the option chain for your chosen symbol.', className='card-subtitle'),
            dbc.Row(
              dbc.Col(
                [
                  dbc.Input(id='query-symbol-input', placeholder='SPY, IWM, etc.', type='text'),
                  dbc.Button('Query', id='query-btn',  color='primary', className='me-1', n_clicks=0)
                ],
                width = 6
              )
            ),
            html.Br(),
            html.Div(id='query-chain-table-container', style={'width': '80%', 'height': '500px', 'overflow': 'auto'})
          ]
        )
    elif pathname == '/optimize':
      return dbc.CardBody([
        html.H4('Define optimization objective & restraints'),
        'Objective',
        html.Div(
          id='optimize-objective-container'
        ),
        html.Br(),
        'Constraints',
        dbc.Row([
          dbc.Col(
            'Such that...',
            width = 4,
          ),
          dbc.Col(
            dbc.Button(
              '+',
              id='optimize-contraints-add-btn',
              n_clicks=0
            ),
            width = 2
          )
        ]),
        html.Div(id='optimize-constraints-container'),
        html.Br(),
        dbc.Row([
          dbc.Col(width=8),
          dbc.Col(
            html.Div(
              dbc.Button(
                'Optimize',
                id='optimize-find-btn',
                n_clicks=0
              ),
              id = 'optimize-find-btn-container' 
            ),
            width = 2
          )
        ]),
        html.Br(),
        html.H4('Model output status'),
        html.Div(
          "Note: the optimizer will run for at most 30 seconds, but formulating the problem may take longer, especially for >1000 input rows."
        ),
        html.Div(
          id = 'optimize-model-status'
        ),
        html.Br(),
        html.H4('Filtered chain (input data)'),
        html.Div(
          id='optimize-chain-table-container',
          style={'width': '80%', 'height': '500px', 'overflow': 'auto'}
        ),
      ])
    elif pathname=='/metrics':

      if is_mobile:
        value_box_width = 6
      else:
        value_box_width = 2
      
      value_boxes = dbc.Row([
        dbc.Col(
          html.Div(
            id = 'metrics-delta-box',
            className='value-box'
          ),
          width=value_box_width
        ),
        dbc.Col(
          html.Div(
            id = 'metrics-gamma-box',
            className='value-box'
          ),
          width=value_box_width
        ),
        dbc.Col(
          html.Div(
            id = 'metrics-theta-box',
            className='value-box'
          ),
          width=value_box_width
        ),
        dbc.Col(
          html.Div(
            id = 'metrics-vega-box',
            className='value-box'
          ),
          width=value_box_width
        ),
        dbc.Col(
          html.Div(
            id = 'metrics-rho-box',
            className='value-box'
          ),
          width=value_box_width
        )
      ])

      return dbc.CardBody([
        dbc.Row([
          dbc.Col([
            "Selected options: ",
            html.Div(
              id='metrics-selected'
            )
          ])
        ]),
        html.Br(),
        value_boxes,
        html.Br(),
        # dbc.Row([
        #   dbc.Col(
        #     html.Div(
        #       id = 'metrics-cost-box',
        #       className='value-box'
        #     ),
        #     width=2
        #   ),
        #   dbc.Col(
        #     html.Div(
        #       id = 'metrics-max-profit-box',
        #       className='value-box'
        #     ),
        #     width=2
        #   ),
        #   dbc.Col(
        #     html.Div(
        #       id = 'metrics-max-loss-box',
        #       className='value-box'
        #     ),
        #     width=2
        #   ),
        #   dbc.Col(
        #     html.Div(
        #       id = 'metrics-avg-profit-box',
        #       className='value-box'
        #     ),
        #     width=2
        #   ),
        #   dbc.Col(
        #     html.Div(
        #       id = 'metrics-avg-loss-box',
        #       className='value-box'
        #     ),
        #     width=2
        #   )
        # ]),
        # html.Br(),
        html.Div(
          id = 'metrics-optionstrat-link-container'
        ),
        # "Profit & Loss Table",
        # html.Div(
        #   id='metrics-pl-table-container',
        #   style={'width': '80%', 'height': '400px', 'overflow': 'auto'}
        # )
      ])

    else:
      return html.Div([
        html.H1('404 - Page not found'),
        # Add custom error message or redirection logic
      ])

# Query - Query the API for the options chain of a symbol
@app.callback(
  Output('query-chain-table-container', 'children'),
  [Input('query-btn', 'n_clicks')],
  [State('query-symbol-input', 'value'),
   State('app-store', 'filtered-chain')]
)
def display_chain_table(n_clicks, symbol_input, filtered_chain):
  if n_clicks > 0:
    if not symbol_input:
      return "Please supply a symbol."
    
    resp = get(url=f'{API_ROOT}/option_chain',
             params={'symbol': symbol_input})

    if(resp.status_code == 500):
      return ['API server returned an error: ' + resp.json()['error']]
    
    df = pd.read_json(StringIO(resp.text))

    return [
      html.P([
        'Use expressions like `>10`, `datestartswith 2024-04`, etc. to filter. For more, see ',
        html.A('here', 
               href='https://dash.plotly.com/datatable/filtering#operators',
               target = '_blank'),
        '. The optimizer will work with the filtered dataset. Note that, if the chain is large, it is filtered to high-OI ',
        'options, then evenly sliced to ~1000 rows.'
      ]),
      dash_table.DataTable(
        id='query-chain-table',
        columns=[{'name': col, 'id': col} for col in df.columns],
        data=df.to_dict('records'),
        filter_action='native'
      )
    ]
  elif filtered_chain:
    df = pd.DataFrame.from_dict(filtered_chain)
    return [
      html.P([
        'Use expressions like `>10`, `datestartswith 2024-04`, etc. to filter. For more, see ',
        html.A('here', 
               href='https://dash.plotly.com/datatable/filtering#operators',
               target = '_blank'),
        '. The optimizer will work with the filtered dataset.'
      ]),
      dash_table.DataTable(
        id='query-chain-table',
        columns=[{'name': col, 'id': col} for col in df.columns],
        data=df.to_dict('records'),
        filter_action='native'
      )
    ]

# Query - store the symbol in the store
@app.callback(
    [Output('app-store', 'symbol')],
    [[Input('query-symbol-input', 'value')]]
)
def store_symbol(value):
  if value:
    return [value]
  else:
    return no_update

# Query - Store the filtered option chain data as JSON
@app.callback(
  [Output('app-store', 'filtered-chain')],
  [Input('query-chain-table', 'derived_virtual_data')]
)
def store_chain(filtered_chain):
  if(filtered_chain):
    return [filtered_chain]
  else:
    return no_update

# Optimize - display objective UI
@app.callback(
    [Output('optimize-objective-container', 'children')],
    [Input('app-store', 'filtered-chain')],
    [State('app-store', 'objective')]
)
def show_objective_ui(filtered_chain, objective):
  if filtered_chain:
    columnDropOptions = get_model_var_options(filtered_chain)

    if not objective:
      return [ dbc.Row([
        dbc.Col(
          dcc.Dropdown(
            id="optimize-objective-verb",
            options=[
              "Maximize",
              "Minimize"
            ]
          ),
          width = 6
        ),
        dbc.Col(
          dcc.Dropdown(
            id="optimize-objective-variable",
            options=columnDropOptions,
          ),
          width = 6
        )
      ]) ]
    elif objective:
      return [ dbc.Row([
        dbc.Col(
          dcc.Dropdown(
            id="optimize-objective-verb",
            options=[
              "Maximize",
              "Minimize"
            ],
            value = objective['verb']
          ),
          width = 6
        ),
        dbc.Col(
          dcc.Dropdown(
            id="optimize-objective-variable",
            options=columnDropOptions,
            value = objective['variable']
          ),
          width = 6
        )
      ]) ]
  else:
    return ["Need to query data first"]

# Optimize - update objective verb
@app.callback(
    [Output('app-store', 'objective', allow_duplicate=True)],
    [Input('optimize-objective-verb', 'value')],
    [State('app-store', 'objective')],
    prevent_initial_call=True
)
def update_objective_verb(value, objective):
  # If objective undefined, initialize it
  if not objective:
    return [
      {
        'verb': '',
        'variable': ''
      }
    ]

  if value:
    objective['verb'] = value
  
  return [objective]

# Optimize - update objective variable
@app.callback(
    [Output('app-store', 'objective', allow_duplicate=True)],
    [Input('optimize-objective-variable', 'value')],
    [State('app-store', 'objective')],
    prevent_initial_call=True
)
def update_objective_variable(value, objective):
  # If objective undefined, initialize it
  if not objective:
    return [
      {
        'verb': '',
        'variable': ''
      }
    ]
  
  if value:
    objective['variable'] = value
  
  return [objective]


# Optimize - add constraints button
@app.callback(
    [Output('app-store', 'constraints', allow_duplicate=True)],
    [Input('optimize-contraints-add-btn', 'n_clicks')],
    [State('app-store', 'constraints'),
     State('app-store', 'filtered-chain')],
    prevent_initial_call=True
)
def add_constraint(n_clicks, constraints, filtered_chain):
  if n_clicks > 0:
    if not filtered_chain:
      return no_update
    
    key = time()

    if not constraints:
      return [{key: {'var': '', 'operator': '', 'operand': 0}}]
    else:
      constraints[key] = {'var': '', 'operator': '', 'operand': 0}
      return [constraints]
  else:
    return no_update

# Optimize - delete the constraint on press
@app.callback(
    [Output('app-store', 'constraints', allow_duplicate=True)],
    [Input({"type": "optimize-constraints-delete", "index": ALL}, "n_clicks")],
    [State('app-store', 'constraints')],
    prevent_initial_call=True
)
def delete_constraint(n_clicks, constraints):
  if sum(n_clicks) > 0:
    clicked_index = -1
    for i in range(0, len(n_clicks)):
      if list(constraints.keys())[i] == ctx.triggered_id['index']:
        clicked_index = i
    
    if n_clicks[clicked_index] > 0:
      del constraints[ctx.triggered_id['index']]
      return [constraints]
  else:
    return no_update

# Optimize - helper function to update a constraints store value
def set_constraints_store_val(value, constraints, triggered_id, updated_constraints_key):
  clicked_index = -1
  for i in range(0, len(value)):
    if list(constraints.keys())[i] == triggered_id['index']:
      clicked_index = i
  
  if value[clicked_index]:
    constraints[triggered_id['index']][updated_constraints_key] = value[clicked_index]
    return [constraints]
  else:
    return no_update

# Optimize - update the stored input variables
@app.callback(
    [Output('app-store', 'constraints', allow_duplicate=True)],
    [Input({"type": "optimize-constraints-variable", "index": ALL}, "value")],
    [State('app-store', 'constraints')],
    prevent_initial_call=True
)
def update_variable(value, constraints):
  return set_constraints_store_val(
    value,
    constraints,
    ctx.triggered_id,
    'var'
  )

# Optimize - update the stored input operators
@app.callback(
    [Output('app-store', 'constraints', allow_duplicate=True)],
    [Input({"type": "optimize-constraints-operator", "index": ALL}, "value")],
    [State('app-store', 'constraints')],
    prevent_initial_call=True
)
def update_operator(value, constraints):
  return set_constraints_store_val(
    value,
    constraints,
    ctx.triggered_id,
    'operator'
  )

# Optimize - update the stored input operands
@app.callback(
    [Output('app-store', 'constraints', allow_duplicate=True)],
    [Input({"type": "optimize-constraints-operand", "index": ALL}, "value")],
    [State('app-store', 'constraints')],
    prevent_initial_call=True
)
def update_operand(value, constraints):
  return set_constraints_store_val(
    value,
    constraints,
    ctx.triggered_id,
    'operand'
  )

def get_model_var_options(filtered_chain):
  return list(
    pd.DataFrame
    .from_dict(filtered_chain[0], orient='index')
    .transpose()
    .infer_objects()
    .select_dtypes(include=['int', 'float'])
    .drop([
      'strike',
      'lastTradeDate'
    ], axis=1)
  )

# Optimize - update in-process store value
@app.callback(
  [Output('app-store', 'solver-in-process', allow_duplicate=True)],
  [Input('optimize-find-btn', 'n_clicks')],
  prevent_initial_call=True
)
def set_solver_in_process_on_click(n_clicks):
  if n_clicks > 0:
    return [ True ]
  else:
    return no_update

# Optimize - show optimize button
@app.callback(
  [Output('optimize-find-btn-container', 'children')],
  [Input('app-store', 'solver-in-process')],
  [State('optimize-find-btn', 'n_clicks')]
)
def show_opt_btn(solver_in_process, n_clicks):
  if solver_in_process:
    return [ dbc.Button(
      'Optimize',
      id='optimize-find-btn',
      n_clicks=n_clicks,
      disabled=True
    ) ]
  else:
    return no_update

# Optimize - display the constraints from the store
@app.callback(
    [Output('optimize-constraints-container', 'children')],
    [Input('app-store', 'constraints')],
    [State('app-store', 'filtered-chain')]
)
def render_constraints_ui(constraints, filtered_chain):
  if constraints:
    columnDropOptions = get_model_var_options(filtered_chain)

    out = []
    for id in constraints:
      out.append(
        dbc.Row([
          dbc.Col(
            dcc.Dropdown(
              columnDropOptions,
              id={
                'type': 'optimize-constraints-variable', 
                'index': id
              },
              value=constraints[id]['var']
            ),
            width=3
          ),
          dbc.Col(
            dcc.Dropdown(
              ['>=', '=', '<='],
              id={
                'type': 'optimize-constraints-operator', 
                'index': id
              },
              value=constraints[id]['operator']
            ),
            width=3
          ),
          dbc.Col(
            dcc.Input(
              type='number',
              id={
                'type': 'optimize-constraints-operand', 
                'index': id
              },
              value=constraints[id]['operand'],
              style={'width': '100%'}
            ),
            width=3
          ),
          dbc.Col(
            dbc.Button(
              'X',
              n_clicks=0,
              id={
                'type': 'optimize-constraints-delete', 
                'index': id
              }
            ),
            width=3
          )
        ])
      )
    
    return [out]
  else:
    return no_update

# Optimize -- start solver on button click
@app.callback(
  [Output('model-output', 'data')],
  [Input('app-store', 'solver-in-process')],
  [State('app-store', 'constraints'),
   State('app-store', 'objective'),
   State('app-store', 'filtered-chain')]
)
def display_chain_table(solver_in_process, constraints, objective, filtered_chain):
  if solver_in_process: # ensure 2nd instance of callback runs, so after solver_in_proc=True
    if not constraints:
      return [{
        'portfolio': {},
        'problem_status_code': -4,
        'problem_status_desc': 'Constraints not initialized -- try changing tabs.'
      }]
    
    if not objective:
      return [{
        'portfolio': {},
        'problem_status_code': -4,
        'problem_status_desc': 'Objective not initialized -- try changing tabs.'
      }]

    if objective['verb'] == '' or objective['variable'] == '':
      return [{
        'portfolio': {},
        'problem_status_code': -4,
        'problem_status_desc': 'Please select an objective'
      }]

    # Call the solver endpoint
    print('calling solver')
    resp = post(
      f'{API_ROOT}/portfolio',
      json={
        'objective': objective,
        'constraints': constraints,
        'filtered-chain': filtered_chain
      }
    )

    if resp.status_code == 500:
      return ['API server returned an error: ' + resp.json()['error']]

    solution = resp.json()

    prob_status = solution['problem_status_code']
    if prob_status in [1, 0]:
      out = {
        'portfolio': solution['portfolio'],
        'problem_status_code': prob_status,
        'problem_status_desc': solution['problem_status_desc']
      }
      print('returning solution')
      return [out]
    elif prob_status in [-1, -2, -3]:
      out = {
        'portfolio': {},
        'problem_status_code': prob_status,
        'problem_status_desc': solution['problem_status_desc']
      }
      return [out]
  else:
    return no_update
  
# Optimize - update in process value when output changes
@app.callback(
    [Output('app-store', 'solver-in-process', allow_duplicate=True)],
    [Input('model-output', 'data')],
    prevent_initial_call=True
)
def show_model_status(model_output):
  if model_output:
    return [
      False
    ]
  else:
    return no_update


# Optimize - Display the model output status when model-output changes
@app.callback(
    [Output('optimize-model-status', 'children', allow_duplicate=True)],
    [Input('model-output', 'data')],
    prevent_initial_call=True
)
def show_model_status(model_output):
  if model_output:
    print('show model status')
    return [
      model_output['problem_status_desc']
    ]
  else:
    return [
      "Not run yet."
    ]


# Optimize - reset optimizer status on button click
@app.callback(
    [Output('optimize-model-status', 'children', allow_duplicate=True)],
    [Input('optimize-find-btn', 'n_clicks')],
    prevent_initial_call=True
)
def show_model_status(n_clicks):
  if n_clicks > 0:
    return [
      "Running, do not change to about/query/metrics tabs..."
    ]
  else:
    return no_update


# Optimize - Display the filtered chain
@app.callback(
  [Output('optimize-chain-table-container', 'children')],
  [Input('app-store', 'filtered-chain')]
)
def show_filtered_chain(filtered_chain):
  df = pd.DataFrame.from_dict(filtered_chain)
  return [
     dash_table.DataTable(
        id='table',
        columns=[{'name': col, 'id': col} for col in df.columns],
        data=df.to_dict('records')
      )
  ]

# Metrics - Display selected options
@app.callback(
  [Output('metrics-selected', 'children')],
  [Input('model-output', 'data')]
)
def show_selected(model_output):
  if model_output:
    model_df = pd.DataFrame.from_dict(model_output['portfolio'])

    # Go through each row and form a string
    smry = []
    for i in range(model_df.shape[0]):
      row = model_df.iloc[i]
      if row['nBuys'] == 1:
        posType = 'BTO'
      elif row['nSells'] == 1:
        posType = 'STO'

      contractType = {
        'puts': 'Put',
        'calls': 'Call'
      }[row['optionType']]

      expiry = row['expiration']

      smry.append(f'{row['qty']}x{posType}: {contractType} {row['strike']} exp. {expiry}')
  
    return [[
      html.Div(x) for x in smry
    ]]
  else:
    return no_update
  
# Metrics -- helper to form greek value boxes
def form_greek_value_box(model_output, greek):
  model_df = pd.DataFrame.from_dict(model_output)

  grk = model_df[greek].mul(model_df['qty']).sum()
  return [[
    html.Div(greek.title(), className="value-box-label"),
    html.Div(round(grk*100, 2), className="value-box-value"),
  ]]

# Metrics - Delta box
@app.callback(
  [Output('metrics-delta-box', 'children')],
  [Input('model-output', 'data')]
)
def show_delta_box(model_output):
  if model_output:
    return form_greek_value_box(model_output['portfolio'], 'delta')
  else:
    return no_update

# Metrics - Vega box
@app.callback(
  [Output('metrics-vega-box', 'children')],
  [Input('model-output', 'data')]
)
def show_vega_box(model_output):
  if model_output:
    return form_greek_value_box(model_output['portfolio'], 'vega')
  else:
    return no_update

# Metrics - Theta box
@app.callback(
  [Output('metrics-theta-box', 'children')],
  [Input('model-output', 'data')]
)
def show_theta_box(model_output):
  if model_output:
    return form_greek_value_box(model_output['portfolio'], 'theta')
  else:
    return no_update
  
# Metrics - Gamma box
@app.callback(
  [Output('metrics-gamma-box', 'children')],
  [Input('model-output', 'data')]
)
def show_gamma_box(model_output):
  if model_output:
    return form_greek_value_box(model_output['portfolio'], 'gamma')
  else:
    return no_update

# Metrics - Rho box
@app.callback(
  [Output('metrics-rho-box', 'children')],
  [Input('model-output', 'data')]
)
def show_rho_box(model_output):
  if model_output:
    return form_greek_value_box(model_output['portfolio'], 'rho')
  else:
    return no_update

# # Metrics - helper to get a value from the portfolio metrics
# def get_metric_value_box(
#     model_metrics,
#     column_name,
#     summary_function,
#     display_name,
#     multiplier
# ):
#   metrics_df = pd.DataFrame.from_dict(model_metrics)

#   metric = summary_function(metrics_df[column_name])
#   return [[
#     html.Div(display_name, className="value-box-label"),
#     html.Div(round(metric*multiplier, 2), className="value-box-value"),
#   ]]

# # Metrics - Probability of Profit box
# @app.callback(
#   [Output('metrics-cost-box', 'children')],
#   [Input('app-store', 'model-metrics')]
# )
# def show_cost_box(model_metrics):
#   return get_metric_value_box(
#     model_metrics,
#     'strategy_cost',
#     mean,
#     'Strategy Cost',
#     100
#   )

# # Metrics - Max profit box
# @app.callback(
#   [Output('metrics-max-profit-box', 'children')],
#   [Input('app-store', 'model-metrics')]
# )
# def show_max_profit_box(model_metrics):
#   return get_metric_value_box(
#     model_metrics,
#     'profit',
#     max,
#     'Max Profit (in table)',
#     1
#   )

# # Metrics - Min profit box
# @app.callback(
#   [Output('metrics-max-loss-box', 'children')],
#   [Input('app-store', 'model-metrics')]
# )
# def show_max_loss_box(model_metrics):
#   return get_metric_value_box(
#     model_metrics,
#     'profit',
#     min,
#     'Max Loss (in table)',
#     1
#   )

# # Metrics - Avg profit box
# @app.callback(
#   [Output('metrics-avg-profit-box', 'children')],
#   [Input('app-store', 'model-metrics')]
# )
# def show_avg_profit_box(model_metrics):
#   return get_metric_value_box(
#     model_metrics,
#     'average_profit_mc',
#     mean,
#     'Average Profit in Profitable Scenarios',
#     1
#   )

# # Metrics - Min profit box
# @app.callback(
#   [Output('metrics-avg-loss-box', 'children')],
#   [Input('app-store', 'model-metrics')]
# )
# def show_avg_loss_box(model_metrics):
#   return get_metric_value_box(
#     model_metrics,
#     'average_loss_mc',
#     mean,
#     'Average Loss in Loss Scenarios',
#     1
#   )

# Metrics - Get model metrics and output to store
@app.callback(
  [Output('app-store', 'model-metrics')],
  [Input('model-output', 'data')],
  [State('app-store', 'symbol')]
)
def get_model_metrics(model_output, symbol):
  if model_output:
    resp = post(
      f'{API_ROOT}/pl-table',
      json={
        'model_output': model_output['portfolio'],
        'symbol': symbol
      }
    )

    if(resp.status_code == 500):
      return ['API server returned an error: ' + resp.json()['error']]

    if resp.json:
      pl_dict = resp.json()
      df = pd.DataFrame.from_dict(pl_dict)
      df = df.explode(['profit', 'spot_price']).reset_index(drop=True)
      return [df.to_dict()]
  else:
    return no_update

# Metrics - show P&L table
# @app.callback(
#   [Output('metrics-pl-table-container', 'children')],
#   [Input('app-store', 'model-metrics')]
# )
# def show_pl_table(metrics):
#   if metrics:
#     metrics_df = pd.DataFrame.from_dict(metrics)
#     to_spread = metrics_df.assign(
#       dt = metrics_df['dt'].apply(lambda x: datetime.strptime(x, '%a, %d %b %Y %H:%M:%S %Z').strftime('%Y-%m-%d'))
#     ).drop_duplicates(subset=['spot_price', 'dt'])
#     pl_df = to_spread.pivot(index='spot_price', columns='dt', values='profit').reset_index()

#     # Remove rows to reach 100
#     target_rows = 100
#     step_size = int(pl_df.shape[0] / target_rows)
#     pl_df = pl_df.iloc[::step_size].sort_values(by='spot_price', ascending=False)

#     # TODO: Look at color_grid example and implement red heatmap for negative values
#     # and green heatmap for positive.

#     return [ dash_table.DataTable(
#         id='metrics-pl-table',
#         columns=[{'name': col, 'id': col} for col in pl_df.columns],
#         data=pl_df.to_dict('records')
#       ) ]
    
# Metrics -- show OptionStrat link
@app.callback(
  Output('metrics-optionstrat-link-container', 'children'),
  [Input('model-output', 'data')],
  [State('app-store', 'symbol')]
)
def show_optionstrat_link(model_output, symbol):
  if model_output:
    portfolio_df = pd.DataFrame.from_dict(model_output['portfolio'])

    # Will need a custom round function to send round(3, 1) -> 3 while round(3.5, 1) -> 3.5
    def my_round(num, digits=0):
      rounded = round(num, digits)
      if rounded == int(rounded):
          return int(rounded)
      return rounded

    opt_strat_strings = []
    for i in range(portfolio_df.shape[0]):
      row = portfolio_df.iloc[i]
      if row['nBuys'] == 1:
        open_or_close = ''
      elif row['nSells'] == 1:
        open_or_close = '-'
      
      dt_str = pd.to_datetime(row['expiration']).strftime('%y%m%d')

      if row['nCalls'] == 1:
        put_or_call = 'C'
      if row['nPuts'] == 1:
        put_or_call = 'P'
      
      opt_strat_str = open_or_close + '.' + symbol + dt_str + put_or_call + str(my_round(row['strike'], 1)) + 'x' + str(row['qty'])
      opt_strat_strings.append(opt_strat_str)
    
    url = 'https://optionstrat.com/build/custom/' + symbol + '/' + ','.join(opt_strat_strings)

    return [
      html.A(
        'OptionStrat Link',
        id='metrics-optionstrat-link',
        href=url,
        target='_blank'
      ),
      " - View P&L chart, max / min return, debit / credit amount, etc. (GreekSculptor has no affiliation with OptionStrat)"
    ]
  else:
    return no_update


### END CALLBACKS ###

if __name__ == '__main__':
  app.run(debug=False, 
          host='0.0.0.0',
          use_reloader=False)
