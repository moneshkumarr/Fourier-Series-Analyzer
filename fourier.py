import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import csv
import json
import os
from datetime import datetime
import unittest
import logging
from pathlib import Path
import re

# Configure logging
logging.basicConfig(filename='fourier_series.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize SymPy symbols
x = sp.symbols('x')
n = sp.symbols('n', integer=True)
f = sp.symbols('f', cls=sp.Function)

class FourierSeriesAnalyzer:
    """Main class for Fourier Series computations and GUI."""
    def __init__(self, root):
        self.root = root
        self.root.title("Fourier Series Analyzer")
        self.root.geometry("1200x800")
        self.results = []
        self.config = self.load_config()
        self.setup_gui()
        self.setup_logging()

    def setup_logging(self):
        """Initialize logging for the application."""
        logging.info("Fourier Series Analyzer started.")

    def load_config(self):
        """Load configuration from a JSON file."""
        default_config = {
            "default_harmonics": 10,
            "plot_resolution": 1000,
            "default_interval": [-sp.pi, sp.pi],
            "export_dir": "./fourier_results"
        }
        config_path = "config.json"
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    config = json.load(file)
                    logging.info("Configuration loaded from config.json")
                    return config
            else:
                with open(config_path, 'w') as file:
                    json.dump(default_config, file, indent=4)
                logging.info("Default configuration created.")
                return default_config
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return default_config

    def save_config(self):
        """Save configuration to a JSON file."""
        try:
            with open("config.json", 'w') as file:
                json.dump(self.config, file, indent=4)
            logging.info("Configuration saved.")
        except Exception as e:
            logging.error(f"Error saving config: {e}")

    def setup_gui(self):
        """Set up the Tkinter GUI."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, expand=True, fill='both')

        # Create tabs
        self.full_range_frame = ttk.Frame(self.notebook)
        self.half_range_frame = ttk.Frame(self.notebook)
        self.complex_frame = ttk.Frame(self.notebook)
        self.results_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.full_range_frame, text="Full Range Series")
        self.notebook.add(self.half_range_frame, text="Half Range Series")
        self.notebook.add(self.complex_frame, text="Complex Fourier Series")
        self.notebook.add(self.results_frame, text="Results")

        self.setup_full_range_tab()
        self.setup_half_range_tab()
        self.setup_complex_tab()
        self.setup_results_tab()

    def setup_full_range_tab(self):
        """Set up the Full Range Series tab."""
        frame = self.full_range_frame
        ttk.Label(frame, text="Full Range Fourier Series", font=("Arial", 16)).pack(pady=10)

        # Piecewise function selection
        ttk.Label(frame, text="Number of Piecewise Functions:").pack()
        self.full_piecewise_var = tk.StringVar(value="1")
        ttk.Radiobutton(frame, text="1", variable=self.full_piecewise_var, value="1").pack()
        ttk.Radiobutton(frame, text="2", variable=self.full_piecewise_var, value="2").pack()

        # Interval inputs
        self.full_inputs = {}
        self.full_inputs['lower'] = self.create_input_field(frame, "Lower Limit:", "0")
        self.full_inputs['upper'] = self.create_input_field(frame, "Upper Limit:", "2*pi")
        self.full_inputs['function1'] = self.create_input_field(frame, "Function 1:", "x")
        self.full_inputs['function2'] = self.create_input_field(frame, "Function 2:", "0")
        self.full_inputs['harmonics'] = self.create_input_field(frame, "Harmonics:", str(self.config['default_harmonics']))

        # Buttons
        ttk.Button(frame, text="Calculate", command=self.calculate_full_range).pack(pady=5)
        ttk.Button(frame, text="Plot", command=self.plot_full_range).pack(pady=5)
        ttk.Button(frame, text="Export to LaTeX", command=self.export_latex_full).pack(pady=5)
        ttk.Button(frame, text="Export to CSV", command=self.export_csv_full).pack(pady=5)

        # Plot area
        self.full_fig, self.full_ax = plt.subplots(figsize=(6, 4))
        self.full_canvas = FigureCanvasTkAgg(self.full_fig, master=frame)
        self.full_canvas.get_tk_widget().pack(pady=10)

    def setup_half_range_tab(self):
        """Set up the Half Range Series tab."""
        frame = self.half_range_frame
        ttk.Label(frame, text="Half Range Fourier Series", font=("Arial", 16)).pack(pady=10)

        # Piecewise function selection
        ttk.Label(frame, text="Number of Piecewise Functions:").pack()
        self.half_piecewise_var = tk.StringVar(value="1")
        ttk.Radiobutton(frame, text="1", variable=self.half_piecewise_var, value="1").pack()
        ttk.Radiobutton(frame, text="2", variable=self.half_piecewise_var, value="2").pack()

        # Interval inputs
        self.half_inputs = {}
        self.half_inputs['upper'] = self.create_input_field(frame, "Upper Limit:", "pi")
        self.half_inputs['function1'] = self.create_input_field(frame, "Function 1:", "x")
        self.half_inputs['lower2'] = self.create_input_field(frame, "Lower Limit 2:", "pi")
        self.half_inputs['upper2'] = self.create_input_field(frame, "Upper Limit 2:", "2*pi")
        self.half_inputs['function2'] = self.create_input_field(frame, "Function 2:", "0")
        self.half_inputs['harmonics'] = self.create_input_field(frame, "Harmonics:", str(self.config['default_harmonics']))

        # Buttons
        ttk.Button(frame, text="Calculate Sine Series", command=self.calculate_half_range_sine).pack(pady=5)
        ttk.Button(frame, text="Calculate Cosine Series", command=self.calculate_half_range_cosine).pack(pady=5)
        ttk.Button(frame, text="Plot", command=self.plot_half_range).pack(pady=5)
        ttk.Button(frame, text="Export to LaTeX", command=self.export_latex_half).pack(pady=5)
        ttk.Button(frame, text="Export to CSV", command=self.export_csv_half).pack(pady=5)

        # Plot area
        self.half_fig, self.half_ax = plt.subplots(figsize=(6, 4))
        self.half_canvas = FigureCanvasTkAgg(self.half_fig, master=frame)
        self.half_canvas.get_tk_widget().pack(pady=10)

    def setup_complex_tab(self):
        """Set up the Complex Fourier Series tab."""
        frame = self.complex_frame
        ttk.Label(frame, text="Complex Fourier Series", font=("Arial", 16)).pack(pady=10)

        # Interval inputs
        self.complex_inputs = {}
        self.complex_inputs['lower'] = self.create_input_field(frame, "Lower Limit:", "-pi")
        self.complex_inputs['upper'] = self.create_input_field(frame, "Upper Limit:", "pi")
        self.complex_inputs['function'] = self.create_input_field(frame, "Function:", "x")
        self.complex_inputs['harmonics'] = self.create_input_field(frame, "Harmonics:", str(self.config['default_harmonics']))

        # Buttons
        ttk.Button(frame, text="Calculate", command=self.calculate_complex).pack(pady=5)
        ttk.Button(frame, text="Plot", command=self.plot_complex).pack(pady=5)
        ttk.Button(frame, text="Export to LaTeX", command=self.export_latex_complex).pack(pady=5)
        ttk.Button(frame, text="Export to CSV", command=self.export_csv_complex).pack(pady=5)

        # Plot area
        self.complex_fig, self.complex_ax = plt.subplots(figsize=(6, 4))
        self.complex_canvas = FigureCanvasTkAgg(self.complex_fig, master=frame)
        self.complex_canvas.get_tk_widget().pack(pady=10)

    def setup_results_tab(self):
        """Set up the Results tab."""
        frame = self.results_frame
        ttk.Label(frame, text="Calculation Results", font=("Arial", 16)).pack(pady=10)
        self.results_text = tk.Text(frame, height=20, width=80)
        self.results_text.pack(pady=10)
        ttk.Button(frame, text="Clear Results", command=self.clear_results).pack(pady=5)
        ttk.Button(frame, text="Save Results", command=self.save_results).pack(pady=5)

    def create_input_field(self, frame, label, default):
        """Create a labeled input field."""
        container = ttk.Frame(frame)
        container.pack(fill='x', padx=5, pady=2)
        ttk.Label(container, text=label).pack(side='left')
        entry = ttk.Entry(container)
        entry.insert(0, default)
        entry.pack(side='left', fill='x', expand=True)
        return entry

    def validate_input(self, expr):
        """Validate mathematical expression."""
        try:
            return sp.sympify(expr)
        except Exception as e:
            logging.error(f"Invalid expression: {expr}, Error: {e}")
            messagebox.showerror("Error", f"Invalid expression: {expr}")
            return None

    def calculate_full_range(self):
        """Calculate Full Range Fourier Series."""
        try:
            num_pieces = int(self.full_piecewise_var.get())
            lower = self.validate_input(self.full_inputs['lower'].get())
            upper = self.validate_input(self.full_inputs['upper'].get())
            harmonics = int(self.full_inputs['harmonics'].get())

            if None in (lower, upper):
                return

            c = lower
            l = (upper - c) / 2
            period = 2 * l

            if num_pieces == 1:
                f1 = self.validate_input(self.full_inputs['function1'].get())
                if f1 is None:
                    return
                an, bn, a0 = self.compute_full_range_coefficients(f1, c, l)
                series = self.compute_series(a0, an, bn, l, harmonics)
                result = f"Full Range Series in ({lower}, {upper}):\n{series}"
                self.results.append(result)
                self.results_text.insert(tk.END, result + "\n\n")
                logging.info(f"Calculated full range series: {series}")

            elif num_pieces == 2:
                f1 = self.validate_input(self.full_inputs['function1'].get())
                f2 = self.validate_input(self.full_inputs['function2'].get())
                if None in (f1, f2):
                    return
                an, bn, a0 = self.compute_full_range_piecewise(f1, f2, c, l, lower, upper)
                series = self.compute_series(a0, an, bn, l, harmonics)
                result = f"Full Range Series in ({lower}, {upper}):\n{series}"
                self.results.append(result)
                self.results_text.insert(tk.END, result + "\n\n")
                logging.info(f"Calculated piecewise full range series: {series}")

        except Exception as e:
            logging.error(f"Error in full range calculation: {e}")
            messagebox.showerror("Error", f"Calculation error: {e}")

    def compute_full_range_coefficients(self, f, c, l):
        """Compute coefficients for full range series."""
        funca = f * sp.cos((n * sp.pi * x) / l)
        an = sp.sympify(sp.integrate(funca / l, (x, c, c + 2 * l)))
        funcb = f * sp.sin((n * sp.pi * x) / l)
        bn = sp.sympify(sp.integrate(funcb / l, (x, c, c + 2 * l)))
        a0 = sp.sympify(sp.integrate(f / l, (x, c, c + 2 * l)))
        return an, bn, a0

    def compute_full_range_piecewise(self, f1, f2, c, l, lower, upper):
        """Compute coefficients for piecewise full range series."""
        funca1 = f1 * sp.cos((n * sp.pi * x) / l)
        funca2 = f2 * sp.cos((n * sp.pi * x) / l)
        an = sp.sympify(sp.integrate(funca1 / l, (x, lower, upper)) +
                         sp.integrate(funca2 / l, (x, lower, upper)))
        funcb1 = f1 * sp.sin((n * sp.pi * x) / l)
        funcb2 = f2 * sp.sin((n * sp.pi * x) / l)
        bn = sp.sympify(sp.integrate(funcb1 / l, (x, lower, upper)) +
                         sp.integrate(funcb2 / l, (x, lower, upper)))
        a0 = sp.sympify(sp.integrate(f1 / l, (x, lower, upper)) +
                         sp.integrate(f2 / l, (x, lower, upper)))
        return an, bn, a0

    def compute_series(self, a0, an, bn, l, harmonics):
        """Compute the Fourier series."""
        series = a0 / 2 + sp.Sum(an * sp.cos((n * sp.pi * x) / l) +
                                 bn * sp.sin((n * sp.pi * x) / l), (n, 1, harmonics)).doit()
        return series

    def calculate_half_range_sine(self):
        """Calculate Half Range Sine Series."""
        try:
            num_pieces = int(self.half_piecewise_var.get())
            upper = self.validate_input(self.half_inputs['upper'].get())
            harmonics = int(self.half_inputs['harmonics'].get())

            if upper is None:
                return
            l = upper

            if num_pieces == 1:
                f1 = self.validate_input(self.half_inputs['function1'].get())
                if f1 is None:
                    return
                bn = self.compute_half_range_sine_coefficients(f1, l)
                series = sp.Sum(bn * sp.sin((n * sp.pi * x) / l), (n, 1, harmonics)).doit()
                result = f"Half Range Sine Series in (0, {l}):\n{series}"
                self.results.append(result)
                self.results_text.insert(tk.END, result + "\n\n")
                logging.info(f"Calculated half range sine series: {series}")

            elif num_pieces == 2:
                f1 = self.validate_input(self.half_inputs['function1'].get())
                f2 = self.validate_input(self.half_inputs['function2'].get())
                lower2 = self.validate_input(self.half_inputs['lower2'].get())
                if None in (f1, f2, lower2):
                    return
                bn = self.compute_half_range_sine_piecewise(f1, f2, l, lower2)
                series = sp.Sum(bn * sp.sin((n * sp.pi * x) / l), (n, 1, harmonics)).doit()
                result = f"Half Range Sine Series in (0, {l}):\n{series}"
                self.results.append(result)
                self.results_text.insert(tk.END, result + "\n\n")
                logging.info(f"Calculated piecewise half range sine series: {series}")

        except Exception as e:
            logging.error(f"Error in half range sine calculation: {e}")
            messagebox.showerror("Error", f"Calculation error: {e}")

    def compute_half_range_sine_coefficients(self, f, l):
        """Compute coefficients for half range sine series."""
        funcb = f * sp.sin((n * sp.pi * x) / l)
        bn = sp.integrate((2 * funcb) / l, (x, 0, l))
        return bn

    def compute_half_range_sine_piecewise(self, f1, f2, l, lower2):
        """Compute coefficients for piecewise half range sine series."""
        funcb1 = f1 * sp.sin((n * sp.pi * x) / l)
        funcb2 = f2 * sp.sin((n * sp.pi * x) / l)
        bn = (2 / l) * (sp.integrate(funcb1, (x, 0, lower2)) +
                        sp.integrate(funcb2, (x, lower2, l)))
        return bn

    def calculate_half_range_cosine(self):
        """Calculate Half Range Cosine Series."""
        try:
            num_pieces = int(self.half_piecewise_var.get())
            upper = self.validate_input(self.half_inputs['upper'].get())
            harmonics = int(self.half_inputs['harmonics'].get())

            if upper is None:
                return
            l = upper

            if num_pieces == 1:
                f1 = self.validate_input(self.half_inputs['function1'].get())
                if f1 is None:
                    return
                an, a0 = self.compute_half_range_cosine_coefficients(f1, l)
                series = (a0 / 2) + sp.Sum(an * sp.cos((n * sp.pi * x) / l), (n, 1, harmonics)).doit()
                result = f"Half Range Cosine Series in (0, {l}):\n{series}"
                self.results.append(result)
                self.results_text.insert(tk.END, result + "\n\n")
                logging.info(f"Calculated half range cosine series: {series}")

            elif num_pieces == 2:
                f1 = self.validate_input(self.half_inputs['function1'].get())
                f2 = self.validate_input(self.half_inputs['function2'].get())
                lower2 = self.validate_input(self.half_inputs['lower2'].get())
                if None in (f1, f2, lower2):
                    return
                an, a0 = self.compute_half_range_cosine_piecewise(f1, f2, l, lower2)
                series = (a0 / 2) + sp.Sum(an * sp.cos((n * sp.pi * x) / l), (n, 1, harmonics)).doit()
                result = f"Half Range Cosine Series in (0, {l}):\n{series}"
                self.results.append(result)
                self.results_text.insert(tk.END, result + "\n\n")
                logging.info(f"Calculated piecewise half range cosine series: {series}")

        except Exception as e:
            logging.error(f"Error in half range cosine calculation: {e}")
            messagebox.showerror("Error", f"Calculation error: {e}")

    def compute_half_range_cosine_coefficients(self, f, l):
        """Compute coefficients for half range cosine series."""
        funca = f * sp.cos((n * sp.pi * x) / l)
        an = sp.integrate((2 * funca) / l, (x, 0, l))
        a0 = sp.integrate((2 / l) * f, (x, 0, l))
        return an, a0

    def compute_half_range_cosine_piecewise(self, f1, f2, l, lower2):
        """Compute coefficients for piecewise half range cosine series."""
        funca1 = f1 * sp.cos((n * sp.pi * x) / l)
        funca2 = f2 * sp.cos((n * sp.pi * x) / l)
        an = (2 / l) * (sp.integrate(funca1, (x, 0, lower2)) +
                        sp.integrate(funca2, (x, lower2, l)))
        a0 = (2 / l) * (sp.integrate(f1, (x, 0, lower2)) +
                        sp.integrate(f2, (x, lower2, l)))
        return an, a0

    def calculate_complex(self):
        """Calculate Complex Fourier Series."""
        try:
            lower = self.validate_input(self.complex_inputs['lower'].get())
            upper = self.validate_input(self.complex_inputs['upper'].get())
            f = self.validate_input(self.complex_inputs['function'].get())
            harmonics = int(self.complex_inputs['harmonics'].get())

            if None in (lower, upper, f):
                return

            l = (upper - lower) / 2
            cn = self.compute_complex_coefficients(f, l, lower)
            series = sp.Sum(cn * sp.exp(sp.I * n * sp.pi * x / l), (n, -harmonics, harmonics)).doit()
            result = f"Complex Fourier Series in ({lower}, {upper}):\n{series}"
            self.results.append(result)
            self.results_text.insert(tk.END, result + "\n\n")
            logging.info(f"Calculated complex Fourier series: {series}")

        except Exception as e:
            logging.error(f"Error in complex series calculation: {e}")
            messagebox.showerror("Error", f"Calculation error: {e}")

    def compute_complex_coefficients(self, f, l, lower):
        """Compute coefficients for complex Fourier series."""
        cn = sp.integrate(f * sp.exp(-sp.I * n * sp.pi * x / l) / (2 * l), (x, lower, lower + 2 * l))
        return cn

    def plot_full_range(self):
        """Plot Full Range Fourier Series."""
        try:
            num_pieces = int(self.full_piecewise_var.get())
            lower = self.validate_input(self.full_inputs['lower'].get())
            upper = self.validate_input(self.full_inputs['upper'].get())
            harmonics = int(self.full_inputs['harmonics'].get())

            if None in (lower, upper):
                return

            c = lower
            l = (upper - c) / 2
            period = 2 * l

            self.full_ax.clear()
            if num_pieces == 1:
                f1 = self.validate_input(self.full_inputs['function1'].get())
                if f1 is None:
                    return
                an, bn, a0 = self.compute_full_range_coefficients(f1, c, l)
                series = self.compute_series(a0, an, bn, l, harmonics)
                self.plot_function(series, 0, 4 * period, self.full_ax, "Full Range Series")
            elif num_pieces == 2:
                f1 = self.validate_input(self.full_inputs['function1'].get())
                f2 = self.validate_input(self.full_inputs['function2'].get())
                if None in (f1, f2):
                    return
                an, bn, a0 = self.compute_full_range_piecewise(f1, f2, c, l, lower, upper)
                series = self.compute_series(a0, an, bn, l, harmonics)
                self.plot_function(series, 0, 4 * period, self.full_ax, "Full Range Series (Piecewise)")
            self.full_canvas.draw()

        except Exception as e:
            logging.error(f"Error in plotting full range: {e}")
            messagebox.showerror("Error", f"Plotting error: {e}")

    def plot_half_range(self):
        """Plot Half Range Fourier Series."""
        try:
            num_pieces = int(self.half_piecewise_var.get())
            upper = self.validate_input(self.half_inputs['upper'].get())
            harmonics = int(self.half_inputs['harmonics'].get())

            if upper is None:
                return
            l = upper

            self.half_ax.clear()
            if num_pieces == 1:
                f1 = self.validate_input(self.half_inputs['function1'].get())
                if f1 is None:
                    return
                bn = self.compute_half_range_sine_coefficients(f1, l)
                sin_series = sp.Sum(bn * sp.sin((n * sp.pi * x) / l), (n, 1, harmonics)).doit()
                an, a0 = self.compute_half_range_cosine_coefficients(f1, l)
                cos_series = (a0 / 2) + sp.Sum(an * sp.cos((n * sp.pi * x) / l), (n, 1, harmonics)).doit()
                self.plot_function(sin_series, 0, 2 * l, self.half_ax, "Sine Series")
                self.plot_function(cos_series, 0, 2 * l, self.half_ax, "Cosine Series")
            elif num_pieces == 2:
                f1 = self.validate_input(self.half_inputs['function1'].get())
                f2 = self.validate_input(self.half_inputs['function2'].get())
                lower2 = self.validate_input(self.half_inputs['lower2'].get())
                if None in (f1, f2, lower2):
                    return
                bn = self.compute_half_range_sine_piecewise(f1, f2, l, lower2)
                sin_series = sp.Sum(bn * sp.sin((n * sp.pi * x) / l), (n, 1, harmonics)).doit()
                an, a0 = self.compute_half_range_cosine_piecewise(f1, f2, l, lower2)
                cos_series = (a0 / 2) + sp.Sum(an * sp.cos((n * sp.pi * x) / l), (n, 1, harmonics)).doit()
                self.plot_function(sin_series, 0, 2 * l, self.half_ax, "Sine Series (Piecewise)")
                self.plot_function(cos_series, 0, 2 * l, self.half_ax, "Cosine Series (Piecewise)")
            self.half_canvas.draw()

        except Exception as e:
            logging.error(f"Error in plotting half range: {e}")
            messagebox.showerror("Error", f"Plotting error: {e}")

    def plot_complex(self):
        """Plot Complex Fourier Series."""
        try:
            lower = self.validate_input(self.complex_inputs['lower'].get())
            upper = self.validate_input(self.complex_inputs['upper'].get())
            f = self.validate_input(self.complex_inputs['function'].get())
            harmonics = int(self.complex_inputs['harmonics'].get())

            if None in (lower, upper, f):
                return

            l = (upper - lower) / 2
            cn = self.compute_complex_coefficients(f, l, lower)
            series = sp.Sum(cn * sp.exp(sp.I * n * sp.pi * x / l), (n, -harmonics, harmonics)).doit()

            self.complex_ax.clear()
            self.plot_function(series, lower, lower + 4 * l, self.complex_ax, "Complex Fourier Series")
            self.complex_canvas.draw()

        except Exception as e:
            logging.error(f"Error in plotting complex series: {e}")
            messagebox.showerror("Error", f"Plotting error: {e}")

    def plot_function(self, expr, start, end, ax, label):
        """Plot a SymPy expression."""
        try:
            lambdified = sp.lambdify(x, expr, modules=['numpy'])
            x_vals = np.linspace(float(start), float(end), self.config['plot_resolution'])
            y_vals = lambdified(x_vals)
            if np.iscomplexobj(y_vals):
                y_vals = np.real(y_vals)  # Plot real part for complex series
            ax.plot(x_vals, y_vals, label=label)
            ax.legend()
            ax.grid(True)
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title(label)
        except Exception as e:
            logging.error(f"Error in plotting function: {e}")
            messagebox.showerror("Error", f"Plotting error: {e}")

    def export_latex_full(self):
        """Export Full Range Series to LaTeX."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export.")
            return
        try:
            latex_content = self.generate_latex(self.results[-1])
            file_path = filedialog.asksaveasfilename(defaultextension=".tex", filetypes=[("LaTeX files", "*.tex")])
            if file_path:
                with open(file_path, 'w') as file:
                    file.write(latex_content)
                logging.info(f"Exported LaTeX to {file_path}")
                messagebox.showinfo("Success", "LaTeX file exported successfully.")
        except Exception as e:
            logging.error(f"Error exporting LaTeX: {e}")
            messagebox.showerror("Error", f"Export error: {e}")

    def export_latex_half(self):
        """Export Half Range Series to LaTeX."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export.")
            return
        try:
            latex_content = self.generate_latex(self.results[-1])
            file_path = filedialog.asksaveasfilename(defaultextension=".tex", filetypes=[("LaTeX files", "*.tex")])
            if file_path:
                with open(file_path, 'w') as file:
                    file.write(latex_content)
                logging.info(f"Exported LaTeX to {file_path}")
                messagebox.showinfo("Success", "LaTeX file exported successfully.")
        except Exception as e:
            logging.error(f"Error exporting LaTeX: {e}")
            messagebox.showerror("Error", f"Export error: {e}")

    def export_latex_complex(self):
        """Export Complex Fourier Series to LaTeX."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export.")
            return
        try:
            latex_content = self.generate_latex(self.results[-1])
            file_path = filedialog.asksaveasfilename(defaultextension=".tex", filetypes=[("LaTeX files", "*.tex")])
            if file_path:
                with open(file_path, 'w') as file:
                    file.write(latex_content)
                logging.info(f"Exported LaTeX to {file_path}")
                messagebox.showinfo("Success", "LaTeX file exported successfully.")
        except Exception as e:
            logging.error(f"Error exporting LaTeX: {e}")
            messagebox.showerror("Error", f"Export error: {e}")

    def generate_latex(self, series):
        """Generate LaTeX code for the series."""
        latex_preamble = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\begin{document}
\section{Fourier Series Result}
The Fourier series is given by:
\begin{equation}
"""
        latex_content = sp.latex(series)
        latex_end = r"""
\end{equation}
\end{document}
"""
        return latex_preamble + latex_content + latex_end

    def export_csv_full(self):
        """Export Full Range Series data to CSV."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export.")
            return
        try:
            series = self.results[-1]
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.export_series_to_csv(series, file_path)
                logging.info(f"Exported CSV to {file_path}")
                messagebox.showinfo("Success", "CSV file exported successfully.")
        except Exception as e:
            logging.error(f"Error exporting CSV: {e}")
            messagebox.showerror("Error", f"Export error: {e}")

    def export_csv_half(self):
        """Export Half Range Series data to CSV."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export.")
            return
        try:
            series = self.results[-1]
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.export_series_to_csv(series, file_path)
                logging.info(f"Exported CSV to {file_path}")
                messagebox.showinfo("Success", "CSV file exported successfully.")
        except Exception as e:
            logging.error(f"Error exporting CSV: {e}")
            messagebox.showerror("Error", f"Export error: {e}")

    def export_csv_complex(self):
        """Export Complex Fourier Series data to CSV."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export.")
            return
        try:
            series = self.results[-1]
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.export_series_to_csv(series, file_path)
                logging.info(f"Exported CSV to {file_path}")
                messagebox.showinfo("Success", "CSV file exported successfully.")
        except Exception as e:
            logging.error(f"Error exporting CSV: {e}")
            messagebox.showerror("Error", f"Export error: {e}")

    def export_series_to_csv(self, series, file_path):
        """Export series data to CSV."""
        try:
            lambdified = sp.lambdify(x, series, modules=['numpy'])
            x_vals = np.linspace(-10, 10, self.config['plot_resolution'])
            y_vals = lambdified(x_vals)
            if np.iscomplexobj(y_vals):
                y_vals = np.real(y_vals)
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['x', 'f(x)'])
                for x_val, y_val in zip(x_vals, y_vals):
                    writer.writerow([x_val, y_val])
        except Exception as e:
            logging.error(f"Error in CSV export: {e}")
            raise

    def clear_results(self):
        """Clear the results text area."""
        self.results_text.delete(1.0, tk.END)
        self.results.clear()
        logging.info("Results cleared.")

    def save_results(self):
        """Save results to a text file."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to save.")
            return
        try:
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if file_path:
                with open(file_path, 'w') as file:
                    file.write("\n".join(self.results))
                logging.info(f"Results saved to {file_path}")
                messagebox.showinfo("Success", "Results saved successfully.")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            messagebox.showerror("Error", f"Save error: {e}")

class TestFourierSeriesAnalyzer(unittest.TestCase):
    """Unit tests for FourierSeriesAnalyzer."""
    def setUp(self):
        self.root = tk.Tk()
        self.analyzer = FourierSeriesAnalyzer(self.root)
        self.root.withdraw()  # Hide the main window during tests

    def tearDown(self):
        self.root.destroy()

    def test_validate_input_valid(self):
        """Test valid input parsing."""
        result = self.analyzer.validate_input("x**2")
        self.assertEqual(str(result), "x**2")

    def test_validate_input_invalid(self):
        """Test invalid input parsing."""
        result = self.analyzer.validate_input("invalid++")
        self.assertIsNone(result)

    def test_compute_full_range_coefficients(self):
        """Test full range coefficient computation."""
        f = x**2
        c = -sp.pi
        l = sp.pi
        an, bn, a0 = self.analyzer.compute_full_range_coefficients(f, c, l)
        self.assertIsNotNone(an)
        self.assertIsNotNone(bn)
        self.assertIsNotNone(a0)

    def test_compute_complex_coefficients(self):
        """Test complex coefficient computation."""
        f = x
        l = sp.pi
        lower = -sp.pi
        cn = self.analyzer.compute_complex_coefficients(f, l, lower)
        self.assertIsNotNone(cn)

    def test_generate_latex(self):
        """Test LaTeX generation."""
        series = x**2
        latex = self.analyzer.generate_latex(series)
        self.assertIn(r"\begin{equation}", latex)
        self.assertIn(r"x^{2}", latex)

if __name__ == "__main__":
    root = tk.Tk()
    app = FourierSeriesAnalyzer(root)
    root.mainloop()

    # Run unit tests
    unittest.main(argv=[''], exit=False)