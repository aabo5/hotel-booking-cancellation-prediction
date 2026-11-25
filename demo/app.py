"""Live demo: predict the cancellation probability of a single booking.

Run from the project folder:   python demo/app.py
Uses models/preprocessor.joblib (notebook 02) and the tuned models of notebooks 03-08,
so the input is a raw booking exactly as it is stored in hotel_bookings.csv.
"""
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
MODEL_FILES = {
    "LightGBM (tuned, best)": "lightgbm_tuned.joblib",
    "Random Forest (tuned)": "random_forest_tuned.joblib",
    "XGBoost (tuned)": "xgboost_tuned.joblib",
    "MLP Neural Net (tuned)": "mlp_tuned.joblib",
    "Decision Tree (tuned)": "decision_tree_tuned.joblib",
    "KNN (tuned)": "knn_tuned.joblib",
}
PREPROCESSOR = joblib.load(ROOT / "models" / "preprocessor.joblib")

# Fields shown in the form: (column, label, widget type, options/default)
FIELDS = [
    ("hotel", "Hotel", "combo", ["City Hotel", "Resort Hotel"]),
    ("lead_time", "Lead time (days before arrival)", "int", 120),
    ("arrival_date_month", "Arrival month", "combo", MONTHS),
    ("stays_in_weekend_nights", "Weekend nights", "int", 1),
    ("stays_in_week_nights", "Week nights", "int", 2),
    ("adults", "Adults", "int", 2),
    ("children", "Children", "int", 0),
    ("country", "Country (ISO-3)", "combo", ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL",
                                             "BEL", "BRA", "NLD", "USA", "TUR", "Other"]),
    ("market_segment", "Market segment", "combo", ["Online TA", "Offline TA/TO", "Direct",
                                                   "Groups", "Corporate", "Complementary", "Aviation"]),
    ("deposit_type", "Deposit type", "combo", ["No Deposit", "Non Refund", "Refundable"]),
    ("customer_type", "Customer type", "combo", ["Transient", "Transient-Party", "Contract", "Group"]),
    ("adr", "Average daily rate (EUR)", "float", 100.0),
    ("previous_cancellations", "Previous cancellations", "int", 0),
    ("booking_changes", "Booking changes", "int", 0),
    ("required_car_parking_spaces", "Parking spaces", "int", 0),
    ("total_of_special_requests", "Special requests", "int", 0),
]

PRESETS = {
    "High-risk example": {"hotel": "City Hotel", "lead_time": 250, "market_segment": "Groups",
                          "deposit_type": "Non Refund", "total_of_special_requests": 0,
                          "country": "PRT", "previous_cancellations": 1, "customer_type": "Transient"},
    "Typical online booking": {"hotel": "City Hotel", "lead_time": 90, "market_segment": "Online TA",
                               "deposit_type": "No Deposit", "total_of_special_requests": 1,
                               "country": "FRA", "adr": 120.0},
    "Low-risk example": {"hotel": "Resort Hotel", "lead_time": 5, "market_segment": "Direct",
                         "deposit_type": "No Deposit", "total_of_special_requests": 2,
                         "required_car_parking_spaces": 1, "country": "GBR",
                         "previous_cancellations": 0, "customer_type": "Transient"},
}

# Columns not in the form get a typical value
DEFAULTS = {
    "arrival_date_year": 2017, "arrival_date_week_number": 27, "arrival_date_day_of_month": 15,
    "babies": 0, "meal": "BB", "distribution_channel": "TA/TO", "is_repeated_guest": 0,
    "previous_bookings_not_canceled": 0, "reserved_room_type": "A", "assigned_room_type": "A",
    "agent": 9.0, "company": 0.0, "days_in_waiting_list": 0,
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hotel Booking Cancellation Predictor")
        self.geometry("760x640")
        self.models = {}
        self.vars = {}

        left = ttk.Frame(self, padding=12)
        left.grid(row=0, column=0, sticky="nsew")
        right = ttk.Frame(self, padding=12)
        right.grid(row=0, column=1, sticky="nsew")
        self.columnconfigure(1, weight=1)

        ttk.Label(left, text="Booking details", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        for i, (col, label, kind, opt) in enumerate(FIELDS, start=1):
            ttk.Label(left, text=label).grid(row=i, column=0, sticky="w", pady=2)
            if kind == "combo":
                var = tk.StringVar(value=opt[0] if col != "arrival_date_month" else "July")
                ttk.Combobox(left, textvariable=var, values=opt, width=18, state="readonly").grid(row=i, column=1, pady=2)
            else:
                var = tk.StringVar(value=str(opt))
                ttk.Entry(left, textvariable=var, width=21).grid(row=i, column=1, pady=2)
            self.vars[col] = var

        ttk.Label(right, text="Model", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.model_var = tk.StringVar(value=list(MODEL_FILES)[0])
        ttk.Combobox(right, textvariable=self.model_var, values=list(MODEL_FILES),
                     state="readonly", width=26).pack(anchor="w", pady=(2, 12))
        ttk.Button(right, text="Predict", command=self.predict).pack(fill="x", pady=2)
        for name in PRESETS:
            ttk.Button(right, text=f"Load {name}", command=lambda n=name: self.load_preset(n)).pack(fill="x", pady=2)

        self.result = tk.Label(right, text="-", font=("Segoe UI", 34, "bold"))
        self.result.pack(pady=(30, 4))
        self.verdict = tk.Label(right, text="Fill the form and press Predict", font=("Segoe UI", 12))
        self.verdict.pack()
        self.bar = ttk.Progressbar(right, length=260, maximum=100)
        self.bar.pack(pady=12)
        self.info = tk.Label(right, text="", justify="left", fg="#555", wraplength=300)
        self.info.pack(anchor="w")

    def load_preset(self, name):
        for col, _, kind, opt in FIELDS:  # reset the form first
            self.vars[col].set(str(opt) if kind != "combo" else ("July" if col == "arrival_date_month" else opt[0]))
        for col, val in PRESETS[name].items():
            self.vars[col].set(str(val))
        self.predict()

    def get_model(self, name):
        if name not in self.models:
            self.models[name] = joblib.load(ROOT / "models" / MODEL_FILES[name])
        return self.models[name]

    def booking(self):
        row = dict(DEFAULTS)
        for col, _, kind, _ in FIELDS:
            v = self.vars[col].get()
            row[col] = int(v) if kind == "int" else float(v) if kind == "float" else v
        if row["country"] == "Other":
            row["country"] = "Unknown"
        if row["adults"] + row["children"] + row["babies"] == 0 or row["adr"] < 0:
            raise ValueError("At least one guest and a non-negative rate are required.")
        df = pd.DataFrame([row])
        # same feature engineering as notebook 02
        df["total_stay_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        df["total_guests"] = df["adults"] + df["children"] + df["babies"]
        df["has_special_requests"] = (df["total_of_special_requests"] > 0).astype(int)
        df["has_booking_changes"] = (df["booking_changes"] > 0).astype(int)
        df["room_type_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
        cols = list(PREPROCESSOR.feature_names_in_)
        X = PREPROCESSOR.transform(df[cols])
        return pd.DataFrame(X, columns=PREPROCESSOR.get_feature_names_out())

    def predict(self):
        try:
            X = self.booking()
            p = self.get_model(self.model_var.get()).predict_proba(X)[0, 1] * 100
        except Exception as e:  # show input errors in the window
            self.result.config(text="!", fg="#b00")
            self.verdict.config(text=str(e), fg="#b00")
            return
        color = "#c0392b" if p >= 50 else "#e67e22" if p >= 30 else "#27ae60"
        self.result.config(text=f"{p:.1f}%", fg=color)
        self.verdict.config(text="Likely to CANCEL" if p >= 50 else "Likely to STAY", fg=color)
        self.bar["value"] = p
        self.info.config(text=f"Model: {self.model_var.get()}\nProbability of cancellation predicted by the "
                              "trained pipeline (feature engineering, scaling, one-hot encoding + model).")


if __name__ == "__main__":
    App().mainloop()
