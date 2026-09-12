from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def dashboard():
    # Load detected plates
    detected = pd.read_csv("../test.csv")
    # Load allowed plates
    whitelist = pd.read_csv("../whitelist.csv")
    
    # Merge to find allowed entries
    allowed_entries = pd.merge(detected, whitelist, left_on='license_number', right_on='plate_number')
    allowed_entries = allowed_entries[allowed_entries['status'] == 'allowed']
    
    # Send table to HTML template
    return render_template("dashboard.html", tables=[allowed_entries.to_html(classes='table table-striped', index=False)])

if __name__ == "__main__":
    app.run(debug=True)
