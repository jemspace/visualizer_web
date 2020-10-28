## Cache visualizer web app

**Front-end Dash application for cache visualizer**
*Description*
Contains the graphical layout (html and other components) of Dash web app,
callbacks to validate/filter user input, request data from the backend, 
and functions that render graphs.



#### to run locally:

requires python 3.6
and everything in requirements.txt, mainly:
flask, plotly, dash, dash bootstrap components, requests
 - all can be installed with pip

recommended to create a virtual environment and run the app in it
ex:
	python -m venv some_env
	source some_env/bin/activate

to run the Dash app on localhost:
	python webapp.py

if running on a public IP, install gunicorn and run with:
	gunicorn webapp:app.server -b :8050



#### INTERFACE OPTION FILTERS / LAYOUTS

Functions in this section fetch the options for config builder; Config builder interface allows the user to pick
what data to graph (type of graph, type of cache algorithm to simulate, cache size, and trace file)
 - these options are fetched from the backend server



#### STATUS UPDATES

Each graph requested has an associated request id; Backend server stores status information corresponding to each id - 
status shows the state of completion for any given request (ex.: "trace file downloaded", "processing data for ...", 
"rendering graph ...", etc.)
*request format for status updates:*
	requests.post(BACKEND_URL + '/status',json = {"id" : unique_graph_id })

additional callbacks included to terminate status updates once a graph is completed



#### GRAPH DATA / GRAPH RENDERING
Callbacks and functions that request the data from backend and render graphs
Graph types:
+ regular graphs from the checklist
+ heatmap - graphs for the entire dataset, ranks algorithms by hit rate
+ pairwise overlay - compares hit rates for 2 or more algorithms in a line graph

Each graph is stored in its own Div, identified by graph's unique id;
Any graph is created with following general steps:
1. For each type, there is a callback that creates the Div component for the graph
   and initiates the interval for graph's status updates (function name typically has 'init' at the end);
2. Each graph Div triggers a callback that requests data for the graph from backend
3. Once graph data for each axis is retrieved from the backend, a plotly graph with that data
   is generated in a separate function

**Regular graphs**
graph_divs_init(clicks, graph_opts, config)
add_graphs_to_div(current_div, graph_list, conf)
render_graph(g_id, conf, params, request_id)

get_line(idx, title, xs, ys, x_label, y_label)
get_scatter(idx, title, xs, ys, x_label, y_label)
get_bar(idx, title, xs, ys, x_label, y_label)

**Heatmap**
heatmap_init(clicks):
render_heatmap(hmap_id, conf)
get_an_heatmap(idx, title, xs, ys, zs, x_label, y_label)

**Overlay**
overlay_graph_init(clicks)
gen_all_pairs(array)
render_overlay(r_id, conf)
get_line_overlay2(idx, title, xs, ys0, ys1, names, x_label, y_label)
