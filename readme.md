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


























