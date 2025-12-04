from flask import Blueprint, request, jsonify
from analytics_service import AnalyticsService

past_bp = Blueprint('past_bp', __name__)
service = AnalyticsService()

@past_bp.route('/api/past_analysis', methods=['POST'])
def past_analysis():
    """
    Expects JSON:
    {
      "gee_data": [ { "system:time_start": ..., "NDVI": ..., "Temperature": ..., "Pressure": ... }, ... ],
      "year": 2022,
      "season": "summer"
    }
    """
    payload = request.get_json(force=True)
    gee_data = payload.get('gee_data')
    year = payload.get('year')
    season = payload.get('season', 'summer')

    if gee_data is None or year is None:
        return jsonify({"error":"gee_data and year are required"}), 400

    try:
        year = int(year)
    except:
        return jsonify({"error":"year must be an integer"}), 400

    try:
        result = service.perform_full_analysis_past(gee_data, year, season)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
