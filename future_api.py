from flask import Blueprint, request, jsonify
from analytics_service import AnalyticsService
import io
import pandas as pd

future_bp = Blueprint('future_bp', __name__)
service = AnalyticsService()

@future_bp.route('/api/future_analysis', methods=['POST'])
def future_analysis():
    """
    Expect multipart/form-data:
      - 'historical_gee_json' : file (JSON) or form field (string JSON)
      - 'cmip6_csv' : file upload (CSV)
      - 'area_id' : integer
      - 'year' : integer (future year)
    """

    # ---- Read historical GEE JSON ----
    hist_list = None
    if 'historical_gee_json' in request.files:
        hist_file = request.files['historical_gee_json']
        content = hist_file.read().decode('utf-8')
        hist_list = pd.read_json(content).to_dict(orient='records')
    else:
        hist_text = request.form.get('historical_gee_json')
        if hist_text:
            hist_list = pd.read_json(hist_text).to_dict(orient='records')

    if hist_list is None:
        return jsonify({"error":"historical_gee_json is required"}), 400

    # ---- Read CMIP6 CSV ---- 
    cmip6_buffer = "data/cmip6.csv"


    # ---- Read area_id & year from FORM ----
    area_id = request.form.get("area_id")
    target_year = request.form.get("year")

    if not area_id or not target_year:
        return jsonify({"error": "area_id and year are required"}), 400

    area_id = int(area_id)
    target_year = int(target_year)

    # ---- Run Analysis ----
    try:
        result = service.perform_full_analysis_future(
            historical_gee_data_list=hist_list,
            cmip6_csv_buffer_or_path=cmip6_buffer,
            area_id=area_id,
            target_year=target_year
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
