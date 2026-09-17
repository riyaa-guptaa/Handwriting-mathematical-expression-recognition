import os

# Reduce TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import re
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
import sympy as sp


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "expression_model.keras"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "expression_classes.json"
)

IMAGE_SIZE = 128


# ============================================================
# GLOBAL VARIABLES
# ============================================================

model = None
classes = None

selected_image_path = None
display_image = None


# ============================================================
# LOAD AI MODEL
# ============================================================

def load_ai_model():
    global model, classes

    try:
        # Check model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found:\n{MODEL_PATH}"
            )

        # Check class file
        if not os.path.exists(CLASS_PATH):
            raise FileNotFoundError(
                f"Class file not found:\n{CLASS_PATH}"
            )

        # Load trained AI model
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        # Load expression classes
        with open(CLASS_PATH, "r", encoding="utf-8") as file:
            classes = json.load(file)

        print("AI model loaded successfully.")

        # Update status
        status_label.config(
            text="● AI Model Ready",
            fg="green"
        )

    except Exception as error:

        model = None
        classes = None

        status_label.config(
            text="● AI Model Not Loaded",
            fg="red"
        )

        messagebox.showerror(
            "Model Error",
            f"Could not load AI model.\n\n{error}"
        )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "The selected image could not be read."
        )

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Remove noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Convert handwriting to white
    # pixels on black background
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Resize image
    binary = cv2.resize(
        binary,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    binary = binary.astype(
        np.float32
    ) / 255.0

    # Add channel
    binary = np.expand_dims(
        binary,
        axis=-1
    )

    # Add batch
    binary = np.expand_dims(
        binary,
        axis=0
    )

    return binary


# ============================================================
# GET CLASS NAME
# ============================================================

def get_class_name(index):

    if isinstance(classes, list):

        if index < len(classes):
            return str(classes[index])

    elif isinstance(classes, dict):

        # Integer key
        if index in classes:
            return str(classes[index])

        # String key
        if str(index) in classes:
            return str(classes[str(index)])

        # Sometimes dictionary may contain
        # another nested structure
        if "classes" in classes:

            class_list = classes["classes"]

            if index < len(class_list):
                return str(class_list[index])

    return "Unknown"


# ============================================================
# PREPARE EQUATION FOR SYMPY
# ============================================================

def prepare_equation(equation):

    equation = equation.strip()

    # Remove spaces
    equation = equation.replace(" ", "")

    # Replace Unicode minus
    equation = equation.replace("−", "-")

    # Replace Unicode powers
    equation = equation.replace("²", "^2")
    equation = equation.replace("³", "^3")

    # Replace multiplication symbols
    equation = equation.replace("×", "*")
    equation = equation.replace("·", "*")

    # Convert implicit multiplication:
    # 2x -> 2*x
    equation = re.sub(
        r"(\d)([a-zA-Z])",
        r"\1*\2",
        equation
    )

    # x2 -> x*2
    equation = re.sub(
        r"([a-zA-Z])(\d)(?!\d)",
        r"\1*\2",
        equation
    )

    # Convert ^ to **
    equation = equation.replace("^", "**")

    return equation


# ============================================================
# FORMAT SYMPY EXPRESSION
# ============================================================

def pretty_expression(expression):

    try:
        return str(
            sp.sstr(
                sp.expand(expression)
            )
        )
    except Exception:
        return str(expression)


# ============================================================
# STEP-BY-STEP EQUATION SOLVER
# ============================================================

def solve_equation_steps(equation):

    try:

        # Prepare equation
        clean_equation = prepare_equation(
            equation
        )

        # Check "="
        if "=" not in clean_equation:
            return [
                "The recognized expression is not an equation.",
                "",
                "Please upload an equation containing '='."
            ], "Could not solve"

        # Split equation
        left_text, right_text = clean_equation.split(
            "=",
            1
        )

        if not left_text or not right_text:
            return [
                "Invalid equation.",
                "",
                "Both sides of the equation are required."
            ], "Could not solve"

        # Create variable
        x = sp.symbols("x")

        # Convert strings into SymPy expressions
        left_expr = sp.sympify(left_text)
        right_expr = sp.sympify(right_text)

        # Original readable equation
        original_left = sp.expand(left_expr)
        original_right = sp.expand(right_expr)

        # Standard form
        expression = sp.expand(
            left_expr - right_expr
        )

        # Find solutions
        solutions = sp.solve(
            expression,
            x
        )

        # ====================================================
        # STEP LIST
        # ====================================================

        steps = []

        # Step 1
        steps.append(
            "Step 1: Write the recognized equation"
        )

        steps.append(
            f"    {pretty_expression(original_left)} = "
            f"{pretty_expression(original_right)}"
        )

        # Step 2
        steps.append(
            ""
        )

        steps.append(
            "Step 2: Bring everything to one side"
        )

        steps.append(
            f"    {pretty_expression(expression)} = 0"
        )

        # ====================================================
        # FACTORISATION
        # ====================================================

        factored = sp.factor(expression)

        if factored != expression:

            steps.append("")

            steps.append(
                "Step 3: Factorize the equation"
            )

            steps.append(
                f"    {pretty_expression(factored)} = 0"
            )

            # Get factors
            factor_data = sp.factor_list(
                expression
            )

            factors = factor_data[1]

            if len(factors) > 0:

                steps.append("")

                steps.append(
                    "Step 4: Set each factor equal to zero"
                )

                factor_number = 1

                for factor, power in factors:

                    factor_solution = sp.solve(
                        factor,
                        x
                    )

                    # Only show linear factors
                    if factor_solution:

                        for solution in factor_solution:

                            steps.append(
                                f"    Factor {factor_number}: "
                                f"{pretty_expression(factor)} = 0"
                            )

                            steps.append(
                                f"        → x = "
                                f"{sp.sstr(solution)}"
                            )

                            factor_number += 1

        else:

            # No factorization
            steps.append("")

            steps.append(
                "Step 3: Solve the equation"
            )

            if solutions:

                for solution in solutions:

                    steps.append(
                        f"    → x = "
                        f"{sp.sstr(solution)}"
                    )

        # ====================================================
        # FINAL ANSWER
        # ====================================================

        steps.append("")
        steps.append(
            "========================================"
        )
        steps.append(
            "FINAL ANSWER"
        )
        steps.append(
            "========================================"
        )

        if solutions:

            answer_parts = []

            for solution in solutions:

                answer_parts.append(
                    f"x = {sp.sstr(solution)}"
                )

            final_answer = "   OR   ".join(
                answer_parts
            )

            steps.append(
                f"    {final_answer}"
            )

        else:

            final_answer = "No solution"

            steps.append(
                "    No solution"
            )

        return steps, final_answer

    except Exception as error:

        return [
            "Unable to solve the recognized equation.",
            "",
            f"Error: {error}"
        ], "Could not solve"


# ============================================================
# RECOGNIZE EQUATION
# ============================================================

def recognize():

    if model is None:

        messagebox.showerror(
            "Error",
            "AI model is not loaded."
        )

        return

    if selected_image_path is None:

        messagebox.showwarning(
            "No Image",
            "Please upload a handwritten equation first."
        )

        return

    try:

        # Clear old solution
        solution_text.delete(
            "1.0",
            tk.END
        )

        final_answer_label.config(
            text="---"
        )

        # Preprocess image
        processed = preprocess_image(
            selected_image_path
        )

        # AI prediction
        prediction = model.predict(
            processed,
            verbose=0
        )

        # Get predicted class
        index = int(
            np.argmax(
                prediction[0]
            )
        )

        # Confidence
        confidence = float(
            np.max(
                prediction[0]
            ) * 100
        )

        # Get expression
        expression = get_class_name(
            index
        )

        # Display expression
        expression_label.config(
            text=expression
        )

        # Display confidence
        confidence_label.config(
            text=f"Confidence: {confidence:.2f}%"
        )

        # ====================================================
        # SOLVE STEP BY STEP
        # ====================================================

        steps, final_answer = solve_equation_steps(
            expression
        )

        # Display steps
        solution_text.delete(
            "1.0",
            tk.END
        )

        for step in steps:

            solution_text.insert(
                tk.END,
                step + "\n"
            )

        # Display final answer separately
        final_answer_label.config(
            text=final_answer
        )

        # Scroll to beginning
        solution_text.see(
            "1.0"
        )

    except Exception as error:

        messagebox.showerror(
            "Recognition Error",
            f"Recognition failed:\n\n{error}"
        )


# ============================================================
# UPLOAD IMAGE
# ============================================================

def upload_image():

    global selected_image_path
    global display_image

    file_path = filedialog.askopenfilename(
        title="Select Handwritten Equation",
        filetypes=[
            (
                "Image Files",
                "*.png *.jpg *.jpeg *.bmp"
            ),
            (
                "All Files",
                "*.*"
            )
        ]
    )

    if not file_path:
        return

    selected_image_path = file_path

    try:

        # Open image
        image = Image.open(
            file_path
        )

        # Make copy
        image = image.copy()

        # Resize for display
        image.thumbnail(
            (700, 300)
        )

        # Convert to Tkinter image
        display_image = ImageTk.PhotoImage(
            image
        )

        # Display image
        image_label.config(
            image=display_image,
            text=""
        )

        # Reset results
        expression_label.config(
            text="---"
        )

        confidence_label.config(
            text="Confidence: ---"
        )

        solution_text.delete(
            "1.0",
            tk.END
        )

        solution_text.insert(
            tk.END,
            "Image uploaded successfully.\n\n"
            "Click the RECOGNIZE button."
        )

        final_answer_label.config(
            text="---"
        )

        file_name_label.config(
            text=os.path.basename(
                file_path
            )
        )

    except Exception as error:

        messagebox.showerror(
            "Image Error",
            str(error)
        )


# ============================================================
# CLEAR EVERYTHING
# ============================================================

def clear_all():

    global selected_image_path
    global display_image

    selected_image_path = None
    display_image = None

    # Clear image
    image_label.config(
        image="",
        text="No image selected\n\n"
             "Click 'UPLOAD EQUATION'"
    )

    # Clear filename
    file_name_label.config(
        text="No file selected"
    )

    # Clear expression
    expression_label.config(
        text="---"
    )

    # Clear confidence
    confidence_label.config(
        text="Confidence: ---"
    )

    # Clear solution
    solution_text.delete(
        "1.0",
        tk.END
    )

    solution_text.insert(
        tk.END,
        "Upload a handwritten equation\n"
        "and click RECOGNIZE."
    )

    # Clear final answer
    final_answer_label.config(
        text="---"
    )

# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "HMER - Handwritten Mathematical Expression Recognition"
)

root.geometry("1100x750")

root.minsize(900, 600)

root.configure(
    bg="#f4f6f8"
)


# ============================================================
# SCROLLABLE MAIN WINDOW
# ============================================================

# Main canvas
main_canvas = tk.Canvas(
    root,
    bg="#f4f6f8",
    highlightthickness=0
)

main_canvas.pack(
    side="left",
    fill="both",
    expand=True
)


# Vertical scrollbar
main_scrollbar = tk.Scrollbar(
    root,
    orient="vertical",
    command=main_canvas.yview
)

main_scrollbar.pack(
    side="right",
    fill="y"
)


# Connect scrollbar to canvas
main_canvas.configure(
    yscrollcommand=main_scrollbar.set
)


# Frame inside canvas
main_frame = tk.Frame(
    main_canvas,
    bg="#f4f6f8"
)


# Create window inside canvas
canvas_window = main_canvas.create_window(
    (0, 0),
    window=main_frame,
    anchor="nw"
)


# ============================================================
# UPDATE SCROLL REGION
# ============================================================

def update_scroll_region(event=None):

    main_canvas.configure(
        scrollregion=main_canvas.bbox("all")
    )


main_frame.bind(
    "<Configure>",
    update_scroll_region
)


# Make inner frame same width as canvas
def resize_main_frame(event):

    main_canvas.itemconfig(
        canvas_window,
        width=event.width
    )


main_canvas.bind(
    "<Configure>",
    resize_main_frame
)


# ============================================================
# MOUSE WHEEL SCROLLING
# ============================================================

def mouse_wheel(event):

    main_canvas.yview_scroll(
        int(-1 * (event.delta / 120)),
        "units"
    )


root.bind_all(
    "<MouseWheel>",
    mouse_wheel
)


# ============================================================
# TITLE
# ============================================================

title = tk.Label(
    main_frame,
    text="HANDWRITTEN MATH SOLVER",
    font=("Arial", 28, "bold"),
    bg="#f4f6f8"
)

title.pack(
    pady=(25, 3)
)


subtitle = tk.Label(
    main_frame,
    text="AI-Based Handwritten Mathematical Expression Recognition",
    font=("Arial", 13),
    bg="#f4f6f8"
)

subtitle.pack(
    pady=(0, 3)
)


# ============================================================
# MODEL STATUS
# ============================================================

status_label = tk.Label(
    main_frame,
    text="● Loading AI Model...",
    font=("Arial", 11, "bold"),
    bg="#f4f6f8",
    fg="orange"
)

status_label.pack(
    pady=(0, 15)
)


# ============================================================
# IMAGE AREA
# ============================================================

image_frame = tk.Frame(
    main_frame,
    bg="white",
    bd=2,
    relief="groove",
    width=800,
    height=300
)

image_frame.pack(
    padx=30,
    pady=5
)

image_frame.pack_propagate(False)


image_label = tk.Label(
    image_frame,
    text="No image selected\n\n"
         "Click 'UPLOAD EQUATION'",
    font=("Arial", 17),
    bg="white"
)

image_label.pack(
    expand=True
)


# ============================================================
# FILE NAME
# ============================================================

file_name_label = tk.Label(
    main_frame,
    text="No file selected",
    font=("Arial", 10),
    bg="#f4f6f8",
    fg="#555555"
)

file_name_label.pack(
    pady=(4, 4)
)


# ============================================================
# BUTTONS
# ============================================================

button_frame = tk.Frame(
    main_frame,
    bg="#f4f6f8"
)

button_frame.pack(
    pady=10
)


# Upload
upload_button = tk.Button(
    button_frame,
    text="UPLOAD EQUATION",
    command=upload_image,
    font=("Arial", 11, "bold"),
    width=18,
    height=2,
    cursor="hand2"
)

upload_button.grid(
    row=0,
    column=0,
    padx=7
)


# Recognize
recognize_button = tk.Button(
    button_frame,
    text="RECOGNIZE",
    command=recognize,
    font=("Arial", 11, "bold"),
    width=18,
    height=2,
    cursor="hand2"
)

recognize_button.grid(
    row=0,
    column=1,
    padx=7
)


# Clear
clear_button = tk.Button(
    button_frame,
    text="CLEAR",
    command=clear_all,
    font=("Arial", 11, "bold"),
    width=12,
    height=2,
    cursor="hand2"
)

clear_button.grid(
    row=0,
    column=2,
    padx=7
)


# Exit
exit_button = tk.Button(
    button_frame,
    text="EXIT",
    command=root.destroy,
    font=("Arial", 11, "bold"),
    width=12,
    height=2,
    cursor="hand2"
)

exit_button.grid(
    row=0,
    column=3,
    padx=7
)


# ============================================================
# RECOGNIZED EXPRESSION
# ============================================================

result_title = tk.Label(
    main_frame,
    text="RECOGNIZED EXPRESSION",
    font=("Arial", 16, "bold"),
    bg="#f4f6f8"
)

result_title.pack(
    pady=(5, 2)
)


expression_label = tk.Label(
    main_frame,
    text="---",
    font=("Arial", 23, "bold"),
    bg="#f4f6f8"
)

expression_label.pack()


# ============================================================
# CONFIDENCE
# ============================================================

confidence_label = tk.Label(
    main_frame,
    text="Confidence: ---",
    font=("Arial", 12),
    bg="#f4f6f8"
)

confidence_label.pack(
    pady=(2, 8)
)


# ============================================================
# STEP-BY-STEP SOLUTION TITLE
# ============================================================

solution_title = tk.Label(
    main_frame,
    text="STEP-BY-STEP SOLUTION",
    font=("Arial", 16, "bold"),
    bg="#f4f6f8"
)

solution_title.pack(
    pady=(5, 5)
)


# ============================================================
# SOLUTION BOX
# ============================================================

solution_frame = tk.Frame(
    main_frame,
    bg="white",
    bd=2,
    relief="groove",
    width=850,
    height=300
)

solution_frame.pack(
    padx=50,
    pady=5,
    fill="both"
)

solution_frame.pack_propagate(False)


# Solution scrollbar
solution_scrollbar = tk.Scrollbar(
    solution_frame,
    orient="vertical"
)

solution_scrollbar.pack(
    side="right",
    fill="y"
)


# Solution text box
solution_text = tk.Text(
    solution_frame,
    font=("Consolas", 12),
    bg="white",
    fg="#222222",
    wrap="word",
    yscrollcommand=solution_scrollbar.set,
    padx=18,
    pady=15
)

solution_text.pack(
    side="left",
    fill="both",
    expand=True
)


solution_scrollbar.config(
    command=solution_text.yview
)


solution_text.insert(
    tk.END,
    "Upload a handwritten equation\n"
    "and click RECOGNIZE."
)


# ============================================================
# FINAL ANSWER
# ============================================================

final_title = tk.Label(
    main_frame,
    text="FINAL ANSWER",
    font=("Arial", 16, "bold"),
    bg="#f4f6f8"
)

final_title.pack(
    pady=(12, 3)
)


final_answer_label = tk.Label(
    main_frame,
    text="---",
    font=("Arial", 20, "bold"),
    bg="#f4f6f8"
)

final_answer_label.pack(
    pady=(0, 25)
)


# ============================================================
# START APPLICATION
# ============================================================

root.after(
    100,
    load_ai_model
)

root.mainloop()