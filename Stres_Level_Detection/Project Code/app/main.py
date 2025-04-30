from flask import Flask, render_template, request, redirect, flash
import model.model as m
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # needed for flashing messages!

features = ['MEAN_RR', 'RMSSD', 'pNN25', 'pNN50', 'LF', 'HF', 'LF_HF']

@app.route("/", methods=["GET", "POST"])
def hello():
    st = ""
    if request.method == "POST":
        l = []
        for i in features:
            value = request.form.get(i)
            if value is None or value.strip() == "":
                flash(f"Error: Please fill the '{i}' field!")
                return redirect("/")
            try:
                l.append(float(value))
            except ValueError:
                flash(f"Error: '{i}' must be a number!")
                return redirect("/")

        # Predict using model
        p = m.predict_pipe(l)
        p = np.argmax(p[0])

        # Set stress level based on prediction
        if p == 0:
            st = "No Stress"
        elif p == 1:
            st = "Low Stress"
        else:
            st = "High Stress"

    return render_template('index.html', cond=st)

if __name__ == "__main__":
    app.run(debug=True)
