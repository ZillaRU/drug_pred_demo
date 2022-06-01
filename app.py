import flask
from flask_cors import *
import pandas as pd
# from testing import wavelength_prediction, wavelength_prediction_batch
# from visualization import viz
from testing import wavelength_prediction, wavelength_prediction_batch, generate_download_headers

app = flask.Flask(__name__, template_folder='templates')
CORS(app, supports_credentials=True)  # 设置跨域


# wavelength_pred_single
@app.route('/', methods=['GET', 'POST'])
def pred_single():
    if flask.request.method == 'GET':
        return flask.render_template('wavelength_pred.html')
    if flask.request.method == 'POST':
        mol_smiles = flask.request.form['mol_str']
        sol_smiles = flask.request.form['sol_str']
        res = wavelength_prediction(mol_smiles, sol_smiles)
        prediction = res if isinstance(res, str) else round(float(res), 4)
        return flask.render_template('wavelength_pred.html',
                                     original_input={'Molecule': mol_smiles,
                                                     'Solvent': sol_smiles},
                                     result=prediction
                                     )


@app.route('/wavelength_pred_batch', methods=['GET', 'POST'])
def pred_batch():
    if flask.request.method == 'GET':
        return flask.render_template('wavelength_pred_batch.html')
    if flask.request.method == 'POST':
        mol_sol_csv = flask.request.files['molsolcsv']
        temp = wavelength_prediction_batch(mol_sol_csv)
        if isinstance(temp, str):
            return flask.render_template("Error.html")
        else:
            mols, sols, preds = temp
        df = pd.DataFrame({
            'Molecule': mols,
            'Solvent': sols,
            'Prediction': preds
        })
        csv_data = df.to_csv(index=False, encoding="utf-8")
        return flask.Response(
            csv_data,
            status=200,
            headers=generate_download_headers("csv"),
            mimetype="application/csv",
        )


@app.route('/ee_pred')
def pred_EE():
    return 'pred_EE'


@app.route('/chembiotip')
def pred_multi_DDI_page():
    return "pred_multi_DDI_page"


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5604)
