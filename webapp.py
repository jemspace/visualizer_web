import base64
import json
import requests

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
PLOT_TYPE = 1 #position of plot type in {plots} - scatter, histo, line
X_LBL = 3
Y_LBL = 4
PLOT_NAME = 5
OVER_TIME_LBL = 'time'

graph_types = du.get_graph_key()



application = flask.Flask(__name__)
app = dash.Dash(server=application, external_stylesheets=[dbc.themes.SIMPLEX])


"""
    Provides the basic layout of the webpage,
    renders all main html/dcc components of the interface
"""
def serve_layout():
    layout = html.Div(
        [
            html.H3("upload or chose an existing config"),
            dcc.Upload( #component that handles config file upload
                id="upload-data",
                children=html.Div(
                    ["[upload a new config]"]
                ),
                style={
                    "width": "45%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                }
                #multiple=True      # --multiple uploads
            ),
            dcc.Upload( #component that handles trace data upload
                id="upload-trace",
                children=html.Div(
                    ["[upload a trace]"]
                ),
                style={
                    "width": "45%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                }
            ),
            html.Div(id='config-container',
                children= [
                    html.Div(   id='radio', children=[
                        # component to show radio buttons list of available configs
                        dcc.RadioItems(id = 'config-rad', options = fill_config_list() )  
                        ]       ),
                    dbc.Popover([   # component to display currently chosen config
                            dbc.PopoverHeader(''),
                            dbc.PopoverBody(''),
                            html.Div(id='current-config', style={'display': 'none'})                       
                        ], id='config-po',
                        is_open=False,
                        target='config-container',
                        placement='right')
                ],
                style={'padding': '10px', 'max-width': '400px', 'margin': 'auto'},
            ),
           
            #   Checkbox list of options for graph types
            html.Div(id='options-container', 
                children =[
                    dcc.Checklist(
                        id='graph-options',
                        options=[
                            {'label': 'Access pattern', 'value': '-p'},
                            {'label': 'Reuse distance', 'value': '-r'},
                            {'label': 'Request frequency', 'value': '-g'},
                            {'label': 'Hit rate', 'value': '-H'},
                            {'label': 'Pollution', 'value': '-P'},
                            {'label': 'Miss rate', 'value': '-m'},
                            {'label': 'Lecar - LRU weight', 'value': '-l'},
                            {'label': 'Value of Arc P', 'value': '-w'},
                            {'label': 'Dlirs Stack Size', 'value': '-d'},
                            {'label': 'Cacheus - LRU weight', 'value': '-y'},
                            {'label': 'Cacheus Learning Rate', 'value': '-x'},
                            {'label': 'Cacheus SR Stack Size', 'value': '-z'}
                            ],
                            labelStyle={'display': 'block', 'column-count': 3},
                            value=[]
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


#upload-trace
@app.callback(
    Output('traces', 'children'),
    [Input('upload-trace', 'contents'), Input('upload-trace', 'filename')]
)
def upload_trace(contents, name):
    #{'label': 'Access pattern', 'value': '-p'}
    options = []
    return dcc.Checklist(
        id='trace-options', options=options,
        labelStyle={'display': 'block', 'column-count': 3},
        value=[]
    )


"""
    Decodes a string of utf-8 characters
    originating from an uploaded file
"""
def decode_file(content):
    data = content.encode("utf8").split(b";base64,")[1]
    return base64.decodebytes(data)

"""
    Retrieves a list of existing config files from Mongo
    (currently not in use)
"""
def fill_config_list():
    conf_list = []
    for c in du.get_all_configs():
        conf_list.append(        {'label': c['name'], 'value': str(c['_id'])}         )
    return conf_list



"""
    Updates list of config option radio buttons;    
    If there is a new config in the upload component, it's uploaded
    to Mongo, added to the list of radio buttons and set as the 
    chosen value in the RadioItems component; otherwise returned
    RadioItems component only contains configs from Mongo and
    no value is set as chosen.
"""
@app.callback(
    Output('radio', 'children'),
    [Input('upload-data', 'contents'), Input('upload-data', 'filename'), Input('config-rad', 'value')],
    [State('config-rad', 'options') ]
)
def upd_config(upload_content_enc, upload_name, option, existing_confs):
    current_conf = None
    current_value = None
    if upload_content_enc is not None and upload_name is not None: 
        upload_content = decode_file(upload_content_enc)
        current_conf = conf_error_check(upload_name, upload_content)
        if current_conf[0] != ERR_TAG: #    current_conf = [name, config]
            ins_id = du.insert_config(json.loads(current_conf[1]), current_conf[0])
            existing_confs.append(  { 'label': current_conf[0], 'value': ins_id }  )
            current_value = ins_id
    elif option is not None:
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
    Updates a Popover component config-po that displays currently chosen config
    - popover displays the contents of the currently chosen config;
    Clears the Upload component (replaces the uploaded file that is no
    longer in use with None);
"""
@app.callback(
    [Output('config-po', 'children'), Output('config-po', 'is_open'), 
    Output('upload-data', 'contents'), Output('upload-data', 'filename')],
    [Input('config-rad', 'value')]
)
def upd_config_display(conf_id):
    if conf_id is not None:
        conf = du.find_config(conf_id)
        r = [
            dbc.PopoverHeader('config name'), # name of config at [0]
            dbc.PopoverBody(conf) ,     # config contents at [1] 
            html.Div(id='current-config', children = conf, style={'display': 'none'})
        ]
        return r, True, None, None
    return [], False, None, None
    


# ====================================================================================
"""
    Prepare divs in which graphs would be rendered; 
    Create a list of parameters for each graph (graph-options * current-config)
    based on data from current config and chosed graph types in graph-options
"""
@app.callback(
    Output('graph-target', 'children'),
    [Input('submit', 'n_clicks')],
    [State('graph-options', 'value'), State('current-config', 'children'), 
    State('graph-list', 'children'), State('graph-target', 'children')]  
)
def prep_graph_divs(clicks, graph_opts, config, prev_params, prev_graphs):
    if clicks is None or clicks == 0:
        return html.Div( id='graph-list', style={'display': 'none'})
    all_divs = []
    # /get_permutations
    pl = {'config':str(config), 'flag': str(graph_opts)}
    resp = requests.post(BACKEND_URL + '/get_permutations',data = pl)
    graph_params = json.loads(resp.text)
    print(graph_params)
    if prev_params is not None:
        graph_params = [i for i in graph_params if i not in prev_params] 
    all_divs.append( html.Div( 
        id='graph-list',
        children=graph_params,
        style={'display': 'none'}
        ) )
    start=0
    try: 
        all_divs.extend(prev_graphs[1:])
        start = len(prev_graphs)
    except: pass 

    for i in range(start, ( start + len(graph_params) )): #starting from last graph
        all_divs.append( html.Div(
            id={'type':'graph-div', 'index': i }
        ))
    return all_divs



"""
    Add the graphs to the newly created divs;
    Div index corresponds to the graph that will populate it - 
    index points to the graph parameters in graph-list component
"""
@app.callback(
    Output({'type':'graph-div', 'index': MATCH}, 'children'),
    [Input({'type':'graph-div', 'index': MATCH}, 'id')],
    [State('graph-list', 'children'), State('current-config', 'children')]
)
def render_graph(current_div, graph_list, conf):
    if current_div is None: return ''
    # check for errors in graphs parameters
    if graph_list[current_div['index']].startswith(ERR_TAG):
        return html.Div( 
            id={'type':'figure', 'index':current_div['index']},
            children=[  "Could not render graph: ", 
                graph_list[current_div['index']]  ]
        )
    current_params = graph_list[current_div['index']].split(',')
    print(current_params)
    return add_graph(current_div['index'], conf, current_params)



"""
    Determines the type of graph requested (scatter, histogram or line)
    gets data for all x and y values of the graph from s_generate_graph_data(config, params)
    on the backend (BACKEND_URL), and calls corresponding graphing function
    (line, scatter or bar, depending on graph type)
"""
def add_graph(g_id, conf, params):
    pload = {'config':str(conf), 'params': str(params)}
    e_resp = requests.post(BACKEND_URL + '/get_graph', data = pload)
    d_resp = json.loads(e_resp.text)
    print(d_resp)
    xs = list(map(int, d_resp['xaxis'][1:-1].split(',')))
    ys = list(map(int, d_resp['yaxis'][1:-1].split(',')))
    title = d_resp['title']
    graph=None
    flag = params[FLAG]
    print("calls graphing >>>>")
    if graph_types[flag][PLOT_TYPE] == "scatter":
        graph = get_scatter(
        g_id, graph_types[flag][PLOT_NAME] + " " + title , xs, ys, 
        OVER_TIME_LBL, graph_types[flag][Y_LBL]
        )
    if graph_types[flag][PLOT_TYPE] == "histogram":
        xs = list(xs)
        graph = get_bar(
        g_id, graph_types[flag][PLOT_NAME] + title, xs, ys, 
        graph_types[flag][X_LBL], graph_types[flag][Y_LBL]
        )
    elif graph_types[flag][PLOT_TYPE] == "line":
        graph = get_line(
        g_id, title, xs, ys, 
        OVER_TIME_LBL, graph_types[flag][Y_LBL]
        )
    return graph



"""
    Creates an annotated heatmap graph
"""
@app.callback(
    Output('map-target', 'children'),
    [Input('map-submit', 'n_clicks')],
    [State('current-config', 'children')]
)
def render_heatmap(clicks, conf):
    if clicks is None or clicks: return '0' # DISABLED *********************************************************
    cf = json.load(open('rank.config'))
    title = 'ranking heatmap'
    index = 1
    xlbl = 'cache size'
    ylbl = 'algorithm'
    # index = y, column = x, value = z
    xs, ys, zs = 0,0,0 # csvp.dash_rank_heat_map(cf)
    #get_heatmap
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

'''
if __name__ == "__main__":
    app.run_server(debug=True, port=5055)'''

if __name__ == "__main__":
    app.run_server(host='0.0.0.0')
