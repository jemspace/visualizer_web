import base64
import json
import requests
import uuid 

import flask
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State,  MATCH, ALL
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

import data_utils as du


FLAG=0
TRACE=1
ALGR=2
SIZE=3
BACKEND_URL = 'http://127.0.0.1:5000'

ERR_TAG = 'Error'
OVER_TIME_LBL = 'time'

graph_types = du.get_graph_key()



server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX], server=server)


"""
    Provides the basic layout of the webpage,
    renders all main html/dcc components of the interface
"""
def serve_layout():
    layout = html.Div(
        [
            html.H3("build / pick existing config"),
            dcc.Tabs(id='config-tabs', value='build', children=[
                dcc.Tab(label='Build config', value='build', 
                    children=[
                        dbc.Form( children=get_config_form(), id='config-form' )
                ]),
                dcc.Tab(label='Pick existing', value='pick', 
                    children=[
                        html.Div(   id='radio', children=[
                        # component to show radio buttons list of available configs
                        dcc.RadioItems(id = 'config-rad', options = fill_config_list() )  
                        ]       ),
                ]),
            ]),
            dbc.Popover([   # component to display currently chosen config
                            dbc.PopoverHeader(''),
                            dbc.PopoverBody(''),
                            html.Div(id='current-config', style={'display': 'none'})                       
                        ], id='config-po', is_open=False,
                        target='config-tabs', placement='right'),
            #   Checkbox list of options for graph types
            html.Br(), html.H3("graph options"),
            html.Div(id='options-container', 
                children =[
                    dcc.Checklist(
                        id='graph-options',
                        options= get_graph_types(),
                            labelStyle={'display': 'block', 'column-count': 3}
                        )
                    ], 
                style={ 'max-width' : '600px', 'max-width': '200px' }
            ),
            html.Div( id='traces', children=[
                html.Div(id='trace-options')
            ] ),
            html.Button(id='submit', type='submit', children='generate graphs'),
            html.Button(id='map-submit', type='submit', children='heat map'),
            html.Button(id='overlay-submit', type='submit', children='overlay hit rate'),
            # ------
            # 
            html.Div( 
                id='graph-info',
                children=[
                    html.Div(id='config')
                ],
                style={'display': 'none'} 
            ),
            html.Div(id="graph-status", children=[]),
            html.Div(
                id='graph-target',   # graphs from checkbox list are rendered in this div
                children=[html.Div(id='graph-list', children=[])]
            ),
            html.Div( id='map-target'),   # heatmap is rendered in this div
            html.Div( id='overlay-hr-target') # comparative hit rate is rendered in this div
            
        ], style={ 'max-width' : 1000, "padding": "40px"}
    )
    return layout

# ===========================================================================
"""    INTERFACE OPTION FILTERS / LAYOUTS    """
"""
    gets available options for building a custom config
    api call to backend returns options for algorithm, cache_size and dataset
    categories as a dict
    dict is expected to have these exact keys (algorithm, cache_size, dataset)
    because callback updating it depends on the dcc components (checklists)
    with these exact ids
"""
def get_config_form():
    all_options = requests.get(BACKEND_URL + '/get_conf_options').json()
    columns = []
    # expects categories: algorithm, cache_size, dataset
    # category dataset replaced by pages listing trace files
    for category in all_options:
        cat_options = [    {'label':op, 'value':op} for op in all_options[category]  ]
        columns.append(
            dbc.Col(    dbc.FormGroup([
                dbc.Label(category),
                dbc.Checklist( 
                    options=cat_options, id=category )
                ]), width=4     )
        )
    columns.append(     dbc.Col(    dbc.FormGroup([
        dbc.Label('trace file'),
        dbc.Checklist( id='trace-file' )
        ]), width=4     )
    )
    return dbc.Row( columns )

"""
    filters out trace options
    trace files available for generating graphs are determined
    by the chosen dataset ('dataset' checklist component in 'build' config menu tab)
"""
@app.callback(
    Output('trace-file', 'options'),
    [Input('dataset', 'value')]
)
def filter_trace_opts(dataset_opt):
    if dataset_opt is None or dataset_opt == []: return []
    # dataset filter; to disabple: pld = {'param': 'none'}
    # to include filter: pld = {'param': 'dataset,'+ dataset_opt} 'param':'dataset,FIU'
    trace_opts = []
    for opt in dataset_opt:
        pld = {'param': 'dataset,'+opt}
        filtered_opts = requests.post(BACKEND_URL + '/get_trace_options', data=pld).json()
        trace_opts.extend(filtered_opts)

    return [ {'label':trace, 'value':trace} for trace in trace_opts]


"""
    Decodes a string of utf-8 characters
    originating from an uploaded file
"""
def decode_file(content):
    data = content.encode("utf8").split(b";base64,")[1]
    return base64.decodebytes(data)

"""
    Gets a list of existing config files from Mongo
"""
def fill_config_list():
    conf_list = []
    for c in du.get_all_configs():
        conf_list.append(        {'label': c['name'], 'value': str(c['_id'])}         )
    return conf_list

"""
    Gets a list of all available graph types
    based on graph keys from Mongo
"""
def get_graph_types():
    types = graph_types
    return [ {'label': graph['minio_flag'].replace('_', ' '), 'value': flag } 
        for flag, graph in graph_types.items()  ]



"""
    Updates list of config option radio buttons;    
    If there is a new config in the upload component, it's uploaded
    to Mongo, added to the list of radio buttons and set as the 
    chosen value in the RadioItems component; otherwise returned
    RadioItems component only contains configs from Mongo and
    no value is set as chosen.
"""
@app.callback(
    Output('radio', 'children'), #Input('upload-data', 'contents'), Input('upload-data', 'filename')
    [Input('config-rad', 'value')],
    [State('config-rad', 'options') ]
)
def upd_config(option, existing_confs):
    current_conf = None
    current_value = None
    if option is not None:
        current_value = option
        current_conf_body = du.find_config(option)
        current_conf = current_conf_body
    # 1-new list of config options 2-currently chosen config
    #return radio_buttons, html.Div(id='current-config', children=current_conf, style={'display': 'none'})
    radio_buttons = dcc.RadioItems(id = 'config-rad', 
                        options = existing_confs, value=current_value,
                        labelStyle = {'display': 'block'} )
    return radio_buttons


"""
    Checks for syntax and parameter errors in a config file;
    If a config has a syntax error, or provides an invalid 
    algorithm/cache size, an error tag and error message is returned;
    otherwise the name and contents of the config are returned
"""
def conf_error_check(name, conf):
    err_msg = ERR_TAG + " "
    checked = du.catch_json_errors(conf)  
    if checked:
        checked2 = du.catch_config_errors(checked)
        if checked2 == ([], []):
            return [name, conf]
        else: err_msg = err_msg + "Invalid algorithms and/or cache sizes given: " + str(checked2)
    else: err_msg = err_msg + "\nJSON syntax error"
    return [ERR_TAG, err_msg]



"""
    Updates a Popover component config-po that displays currently chosen or
    built config - popover displays the contents of the current config;
    returns array of 2 outputs: currently chosen/built config
    and boolean value that opens/closes the popover
"""
@app.callback(
    [Output('config-po', 'children'), Output('config-po', 'is_open')],
    [Input('config-rad', 'value'), Input('config-tabs', 'value'), 
    Input('algorithm', 'value'), Input('cache_size', 'value'), Input('dataset', 'value'), Input('trace-file', 'value')] 
        #add extra param for algorithm options
)
def upd_config_display(conf_id, tab, builder_algos, builder_sizes, builder_dataset, builder_traces):
    r=[]
    conf={}
    if tab == 'pick' and conf_id is not None:
        conf = du.find_config(conf_id)
    elif tab == 'build':
        conf={  'cache_sizes': builder_sizes, 'algorithms' : builder_algos, 
                'dataset': builder_dataset, 'traces': builder_traces  }
    r.extend((   dbc.PopoverHeader('Current config'), 
            dbc.PopoverBody(str(conf)) ,    
            html.Div(id='current-config', children = str(conf), style={'display': 'none'})   ) )
    return r, (tab=='build' or conf_id)
    


# ====================================================================================

"""    STATUS UPDATES FOR GRAPHS    """
"""
    Provides periodical live updates on the current graphs
    - status info for current graphs is sent as a response to this post request:
    requests.post(BACKEND_URL + '/status',data = pl)
    status corresponds to the id from status-id div, created when graphs are requested
"""
@app.callback(
    Output({'type':'status', 'index': MATCH}, 'children'),
    [Input({'type':'status-interv', 'index': MATCH}, 'n_intervals')],
    [State({'type':'status-interv', 'index': MATCH}, 'id')]
)
def status_update(interv, request_id):
    pl = {'id': str(request_id['index'])}
    stat = requests.post(BACKEND_URL + '/status',data = pl)
    print("#### " + stat.text + " ####")
    return stat.text



def get_request_id():
    r_id = uuid.uuid4()
    return r_id

# ==============================================================================


"""    GRAPH DATA / GRAPH RENDERING    """
"""
    Prepare divs in which graphs would be rendered; 
    Create a list of parameters for each graph (graph-options * current-config)
    based on data from current config and chosed graph types in graph-options;
    Add a status-id div where the id of current request is stored
"""
@app.callback(
    Output('graph-target', 'children'),
    [Input('submit', 'n_clicks')],
    [State('graph-options', 'value'), State('current-config', 'children')]  
)
def prep_graph_divs(clicks, graph_opts, config):
    if clicks is None or clicks == 0:
        return html.Div( id='graph-list', style={'display': 'none'}, children=[
            html.Div(id='status-id')
        ])
    all_divs = []
    pl = {'config':str(config), 'graph flag': str(graph_opts)}
    resp = requests.post(BACKEND_URL + '/get_permutations',data = pl)
    print(resp)
    graph_params = json.loads(resp.text)
    all_divs.append( html.Div( 
        id='graph-list',
        children=graph_params,
        style={'display': 'none'}
        ) )    
    for i in range( len(graph_params)): #starting from last graph
        r_id = get_request_id()
        all_divs.append( html.Div(
            id={'type':'graph-div', 'index': i }, children=[str(r_id)]
        ))
        all_divs.append( html.Div( id={'type':'status', 'index': str(r_id) }  ))
        all_divs.append(    dcc.Interval(
            id={'type':'status-interv', 'index': str(r_id)},
            interval=1*1000, # in milliseconds
            n_intervals=0,
            )    )
    return all_divs



"""
    Add the graphs to the newly created divs;
    Div index corresponds to the graph that will populate it - 
    index points to the graph parameters in graph-list component
"""
@app.callback(
    Output({'type':'graph-div', 'index': MATCH}, 'children'),
    [Input({'type':'graph-div', 'index': MATCH}, 'id')],
    [State('graph-list', 'children'), State('current-config', 'children'), State({'type':'graph-div', 'index': MATCH}, 'children')]
)
def add_graphs_to_div(current_div, graph_list, conf, r_id):
    if current_div is None: return ''
    # check for errors in graphs parameters
    print("render graph checked div >>>")
    if graph_list[current_div['index']].startswith(ERR_TAG):
        return html.Div( 
            id={'type':'figure', 'index':current_div['index']},
            children=[  "Could not render graph: ", 
                graph_list[current_div['index']]  ]
        )
    request_id = r_id[0]
    print(request_id)
    current_params = graph_list[current_div['index']].split(',')
    print(current_params)
    return render_graph(current_div['index'], conf, current_params, request_id)



"""
    Determines the type of graph requested (scatter, histogram or line)
    gets data for all x and y values of the graph from s_generate_graph_data(config, params)
    on the backend (BACKEND_URL), and calls corresponding graphing function
    (line, scatter or bar, depending on graph type)
"""
def render_graph(g_id, conf, params, request_id):
    pload = {'config':str(conf), 'params': str(params), 'id':str(request_id)}
    e_resp = requests.post(BACKEND_URL + '/get_graph', data = pload)
    d_resp = json.loads(e_resp.text)
    xs = list(map(int, d_resp['xaxis'][1:-1].split(',')))
    ys = list(map(float, d_resp['yaxis'][1:-1].split(',')))
    title = d_resp['res_title']
    graph=None
    flag = params[FLAG]
    print("graphing >>>>")
    if graph_types[flag]['graph_type'] == "scatter":
        graph = get_scatter(
        g_id, graph_types[flag]['title'] + " " + title , xs, ys, 
        OVER_TIME_LBL, graph_types[flag]['y_label']
        )
    if graph_types[flag]['graph_type'] == "histogram":
        xs = list(xs)
        graph = get_bar(
        g_id, graph_types[flag]['title'] + title, xs, ys, 
        graph_types[flag]['x_label'], graph_types[flag]['y_label']
        )
    elif graph_types[flag]['graph_type'] == "line":
        graph = get_line(
        g_id, title, xs, ys, 
        OVER_TIME_LBL, graph_types[flag]['y_label']
        )
    return graph



"""
    Gets data an annotated heatmap graph
"""
@app.callback(
    Output('map-target', 'children'),
    [Input('map-submit', 'n_clicks')],
    [State('current-config', 'children')]
)
def render_heatmap(clicks, conf):
    if clicks is None: return '0' 
    title = 'ranking heatmap'
    index = 1
    xlbl = 'cache size'
    ylbl = 'algorithm'
    conf = json.loads(conf.replace('\'', '\"'))
    pld={'dataset': ",".join(conf['dataset']), 'algos': ",".join(conf['algorithms']),
        'cache size': str(conf['cache_sizes'])[1:-1]} 

    r = requests.post(BACKEND_URL + '/get_heat', data=pld)
    xyzs = r.json()
    xs = xyzs['x_cache']
    print(xs)
    print('==============')
    ys = xyzs['y_algs']
    print(ys)
    print('==============')
    zs = xyzs['data']
    print(zs)
    print('==============')
    return get_an_heatmap(index, title, xs, ys, zs, xlbl, ylbl)



"""
    Gets data for a comparative graph of hit rates of two algorithms
"""
@app.callback(
    Output('overlay-hr-target', 'children'),
    [Input('overlay-submit', 'n_clicks')],
    [State('current-config', 'children')]
)
def render_overlay(clicks, conf):
    if clicks is None: return '0'
    cf = json.loads(conf)
    params = du.config_permutation_gen(cf, ['-H']) # -H for hit rate
    title = 'comparative hit rate '
    allxys = []
    names=[]
    for p in params:
        # flag, trace, algorithm, size
        pload = {'config':str(cf), 'params': p.split(',')}
        e_resp = requests.post(BACKEND_URL ,data = pload)
        d_resp = json.loads(e_resp.text)
        xs = list(map(int, d_resp['xaxis'][1:-1].split(',')))
        ys = list(map(int, d_resp['yaxis'][1:-1].split(',')))
        t = d_resp['title']
        names.append(p.split(',')[2]) #algorithm name
        allxys.append({'x' : xs, 'y' : ys})
        t=p.split(',')[1]
        if len(allxys) >= 2: break

    return get_line_overlay2(0, title+t, allxys, names, OVER_TIME_LBL, graph_types['-H'][Y_LBL])
    


"""
    generates a line plot (plot over time)
    given a list of x values, a list of y values
"""
def get_line(idx, title, xs, ys, x_label, y_label):
    print('graphing line >>>')
    fig = go.Figure()
    counter =0
    #for y in ys:
    fig.add_trace(
        go.Scattergl(
            x = xs,
            y = ys,
            mode = 'lines',
            line = dict( color="#3d2652" ),
            name=title+str(counter)
        )
    )
    fig.update_layout(
        title = title,
        xaxis_title = x_label,
        yaxis_title = y_label
    )
    counter += 1
    return dcc.Graph(
        id={'type':'figure', 'index':idx},
        figure=fig
    )


"""
    generates a scatter plot (access pattern plot)
    given a list of x values, a list of y values
"""
def get_scatter(idx, title, xs, ys, x_label, y_label):
    print('graphing scatter >>>')
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x = xs,
            y = ys,
            mode = 'markers',
            marker = dict( size=5, color="#224f6b" ),
            name=title
        )
    )
    fig.update_layout(
        title = title,
        xaxis_title = x_label,
        yaxis_title = y_label
    )
    return dcc.Graph(
        id={'type':'figure', 'index':idx},
        figure=fig
    )



"""
    generates a scatter plot (access pattern plot)
    given a list of x values, a list of y values 
"""
def get_bar(idx, title, xs, ys, x_label, y_label):
    print('graphing bar >>>')
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x = xs,
            y = ys,
            name=title
        )
    )
    fig.update_layout(
        title = title,
        xaxis_title = x_label,
        yaxis_title = y_label
    )
    return dcc.Graph(
        id={'type':'figure', 'index':idx},
        figure=fig
    )


"""
    generates a comparative line plot of hit rates for 2 algorithms
    given a list of x values, a list of y values;
    shades the difference between the 2 plots based on which plot
    has a higher hit rate at that point
"""
def get_line_overlay2(idx, title, all_xys, names, x_label, y_label):
    bottom_y=[]
    color1 = 'rgba(3, 128, 166, 0.5)' 
    color2 = 'rgba(217, 44, 22, 0.5)'
    for i in range(len(all_xys[0]['x'])):
        if all_xys[0]['y'][i] > all_xys[1]['y'][i]:
            bottom_y.append(all_xys[1]['y'][i])
        else:
            bottom_y.append(all_xys[0]['y'][i])
    fig = go.Figure()
    fig.add_trace( go.Scattergl( 
        x=all_xys[0]['x'], y= all_xys[0]['y'], mode= 'lines', fill='tozeroy', fillcolor=color1, name=names[0]  ) )
    fig.add_trace( go.Scattergl( 
        x=all_xys[1]['x'], y= all_xys[1]['y'], mode='lines', fill='tozeroy', fillcolor=color2, name=names[1]   ) )
    fig.add_trace( go.Scattergl( 
        x=all_xys[0]['x'], y=bottom_y , mode= 'none', fill='tozeroy', fillcolor='rgb(255, 255, 255)', showlegend=False  ) )

    fig.add_trace( go.Scattergl(
        x=all_xys[0]['x'], y=all_xys[0]['y'], mode='lines', line=dict(color=color1), showlegend=False   )
    )
    fig.add_trace( go.Scattergl(
        x=all_xys[1]['x'], y=all_xys[1]['y'], mode='lines', line=dict(color=color2), showlegend=False   )
    )
    fig.update_layout(
        title = title,
        xaxis_title = x_label,
        yaxis_title = y_label,
        plot_bgcolor = 'rgb(255, 255, 255)',
        paper_bgcolor = 'rgb(245, 245, 245)'
    )
    return dcc.Graph(
        id='overlay' + str(idx),
        figure=fig
    )



"""
    generates an annotated heat map
    given a list of x values, a list of y values,
    and 2d list of z values
"""
def get_an_heatmap(idx, title, xs, ys, zs, x_label, y_label):
    fig = ff.create_annotated_heatmap(
        zs, y=ys,
        colorscale='Viridis'
    )
    fig.update_layout(
        title = title,  # xaxis_title = x_label,
        yaxis_title = y_label,
        # xaxis_type='category',
        xaxis = dict(
            tickmode= 'array', 
            ticktext= xs,
            tickvals= [ *range(0, len(xs))],
            side='bottom'     ) 
    )
    return dcc.Graph(
        id= 'heatmap' + str(idx),
        figure=fig
        )
# ===========================================================================

"""
    Call the layout function to render main html and dcc components on the page
"""

app.layout = serve_layout()

#if __name__ == "__main__":
#    app.run_server(debug=True, port=5055)


if __name__ == "__main__":
    app.run_server(port=8050, debug=True)
    #app.run_server(debug=True)   host='0.0.0.0',

