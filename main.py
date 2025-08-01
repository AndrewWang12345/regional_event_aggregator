from flask import Flask, jsonify, request
from utils import fetch_instagram_events

app = Flask(__name__)

@app.route("/api/events", methods=["GET"])
def get_events():
    # Optional query params: club name, date filters etc.
    club = request.args.get("club")
    start_date = request.args.get("start_date")  # ISO format e.g. 2025-06-27
    end_date = request.args.get("end_date")

    events = fetch_instagram_events(club=club, start_date=start_date, end_date=end_date)
    return jsonify(events)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
