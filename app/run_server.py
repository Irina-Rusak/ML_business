# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py

# import the necessary packages
import pandas as pd
import os
import dill
dill._dill._reverse_typemap['ClassType'] = type
#import cloudpickle
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_model(model_path):
	# load the pre-trained model
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)
	print(model)

# modelpath = "/app/app/models/catboost_pipeline.dill"
modelpath = "../models/catboost_pipeline.dill"
load_model(modelpath)

@app.route("/", methods=["GET"])
def general():
	return """Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST"""

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
	dt = strftime("[%Y-%b-%d %H:%M:%S]")
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":

		job, marital, education, contact = "", "", "", ""
		age, balance, duration, campaign, pdays, previous, default, housing, loan, poutcome = None, None, None, None, \
																							  None, None, None, None, \
																							  None, None

		request_json = flask.request.get_json()

		if request_json["age"]:
			age = float(request_json['age'])

		if request_json["job"]:
			job = request_json['job']

		if request_json["marital"]:
			marital = request_json['marital']

		if request_json["education"]:
			education = request_json['education']

		if request_json["default"]:
			default = float(request_json['default'])

		if request_json["balance"]:
			balance = float(request_json['balance'])

		if request_json["housing"]:
			housing = float(request_json['housing'])

		if request_json["loan"]:
			loan = float(request_json['loan'])

		if request_json["contact"]:
			contact = request_json['contact']

		if request_json["duration"]:
			duration = float(request_json['duration'])

		if request_json["campaign"]:
			campaign = float(request_json['campaign'])

		if request_json["pdays"]:
			pdays = float(request_json['pdays'])

		if request_json["previous"]:
			previous = float(request_json['previous'])

		if request_json["poutcome"]:
			poutcome = float(request_json['poutcome'])

		try:
			preds = model.predict_proba(pd.DataFrame({"age": [age],
													  "job": [job],
													  "marital": [marital],
													  "education": [education],
													  "default": [default],
													  "balance": [balance],
													  "housing": [housing],
													  "loan": [loan],
													  "contact": [contact],
													  "duration": [duration],
													  "campaign": [campaign],
													  "pdays": [pdays],
													  "previous": [previous],
													  "poutcome": [poutcome]}))
		except AttributeError as e:
			# logger.warning(f'Exception: {str(e)}')
			data['predictions'] = str(e)
			data['success'] = False
			return flask.jsonify(data)

		data["predictions"] = preds[:, 1][0]
		# indicate that the request was a success
		data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	port = int(os.environ.get('PORT', 8180))
	app.run(host='0.0.0.0', debug=True, port=port)
