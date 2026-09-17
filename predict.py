import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import re
import numpy as np
import tensorflow as tf
import sympy as sp

from image_processor import preprocess_image


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_PATH = "models/expression_model.keras"
CLASS_FILE = "models/expression_classes.json"


# ============================================================
# LOAD AI MODEL
# ============================================================

print("Loading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


with open(CLASS_FILE, "r", encoding="utf-8") as file:
    id_to_expression = json.load(file)


# ============================================================
# CLEAN EQUATION
# ============================================================

def clean_equation(equation):

    equation = equation.strip()

    equation = equation.replace(" ", "")
    equation = equation.replace("^", "**")
    equation = equation.replace("²", "**2")
    equation = equation.replace("³", "**3")
    equation = equation.replace("−", "-")

    # Convert 2x, 3x, 4x, 5x etc.
    equation = re.sub(r"(\d+)x", r"\1*x", equation)

    return equation


# ============================================================
# STEP-BY-STEP SOLVER
# ============================================================

def solve_equation_steps(equation):

    try:

        original_equation = equation

        equation = clean_equation(equation)

        # Separate left and right side
        if "=" not in equation:
            return [], "Not an equation"

        left_text, right_text = equation.split("=", 1)

        x = sp.symbols("x")

        left = sp.sympify(left_text)
        right = sp.sympify(right_text)

        expression = sp.expand(left - right)

        solutions = sp.solve(expression, x)

        steps = []

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        steps.append(
            "Step 1: Write the equation"
        )

        steps.append(
            f"    {sp.latex(left)} = {sp.latex(right)}"
        )

        # ----------------------------------------------------
        # CHECK TYPE
        # ----------------------------------------------------

        degree = sp.degree(expression, x)

        # ====================================================
        # LINEAR EQUATION
        # ====================================================

        if degree == 1:

            steps.append(
                "Step 2: Move the constant term to the other side"
            )

            expanded = sp.expand(expression)

            # ax + b = 0
            a = expression.coeff(x, 1)
            b = expression.coeff(x, 0)

            if b != 0:

                steps.append(
                    f"    {sp.latex(a*x)} = {sp.latex(-b)}"
                )

            steps.append(
                "Step 3: Divide by the coefficient of x"
            )

            value = sp.simplify(-b / a)

            steps.append(
                f"    x = {sp.latex(value)}"
            )

        # ====================================================
        # QUADRATIC EQUATION
        # ====================================================

        elif degree == 2:

            a = expression.coeff(x, 2)
            b = expression.coeff(x, 1)
            c = expression.coeff(x, 0)

            steps.append(
                "Step 2: Write the quadratic equation in standard form"
            )

            steps.append(
                f"    {sp.latex(expression)} = 0"
            )

            # ------------------------------------------------
            # FACTORIZATION
            # ------------------------------------------------

            factorized = sp.factor(expression)

            if factorized != expression:

                steps.append(
                    "Step 3: Factorize the equation"
                )

                steps.append(
                    f"    {sp.latex(factorized)} = 0"
                )

                factors = sp.factor_list(expression)[1]

                steps.append(
                    "Step 4: Set each factor equal to zero"
                )

                for factor, power in factors:

                    factor_solutions = sp.solve(factor, x)

                    for value in factor_solutions:

                        steps.append(
                            f"    {sp.latex(factor)} = 0"
                            f"  →  x = {sp.latex(value)}"
                        )

            # ------------------------------------------------
            # QUADRATIC FORMULA
            # ------------------------------------------------

            else:

                steps.append(
                    "Step 3: Use the quadratic formula"
                )

                steps.append(
                    "    x = (-b ± √(b² - 4ac)) / 2a"
                )

                discriminant = sp.simplify(
                    b**2 - 4*a*c
                )

                steps.append(
                    f"    a = {a},  b = {b},  c = {c}"
                )

                steps.append(
                    f"    Discriminant = {sp.latex(discriminant)}"
                )

                steps.append(
                    "Step 4: Calculate the roots"
                )

                for value in solutions:

                    steps.append(
                        f"    x = {sp.latex(value)}"
                    )

        # ====================================================
        # OTHER EQUATIONS
        # ====================================================

        else:

            steps.append(
                "Step 2: Solve the equation"
            )

            for value in solutions:

                steps.append(
                    f"    x = {sp.latex(value)}"
                )

        # ====================================================
        # FINAL ANSWER
        # ====================================================

        if len(solutions) == 0:

            final_answer = "No solution"

        else:

            final_answer = "  or  ".join(
                [
                    f"x = {sp.latex(value)}"
                    for value in solutions
                ]
            )

        return steps, final_answer

    except Exception as error:

        return [], f"Could not solve equation: {error}"


# ============================================================
# IMAGE INPUT
# ============================================================

image_path = input(
    "\nEnter image path: "
).strip()


# ============================================================
# PREPROCESS IMAGE
# ============================================================

try:

    _, _, _, processed_image = preprocess_image(
        image_path
    )

except Exception as error:

    print("\nError reading image:")
    print(error)

    exit()


input_image = np.expand_dims(
    processed_image,
    axis=0
)


# ============================================================
# AI RECOGNITION
# ============================================================

print("\nRecognizing equation...")

prediction = model.predict(
    input_image,
    verbose=0
)


predicted_id = int(
    np.argmax(prediction[0])
)

confidence = float(
    np.max(prediction[0]) * 100
)


expression = id_to_expression[
    str(predicted_id)
]


# ============================================================
# SOLVE
# ============================================================

steps, final_answer = solve_equation_steps(
    expression
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 50)
print("       HMER RECOGNITION RESULT")
print("=" * 50)

print("\nRecognized Expression:")
print(expression)

print("\nConfidence:")
print(f"{confidence:.2f}%")

print("\n")
print("=" * 50)
print("          SOLUTION STEPS")
print("=" * 50)

for step in steps:
    print(step)

print("\n")
print("=" * 50)
print("           FINAL ANSWER")
print("=" * 50)

print(final_answer)

print("\n")
print("=" * 50)