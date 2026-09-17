import sympy as sp


def solve_equation_steps(equation):
    x = sp.symbols("x")

    # Clean equation
    equation = equation.replace("^", "**")
    equation = equation.replace("²", "**2")
    equation = equation.replace("−", "-")
    equation = equation.replace(" ", "")

    # Convert 2x, 3x, 4x, 5x etc. into SymPy format
    import re
    equation = re.sub(r'(\d+)x', r'\1*x', equation)

    # Split equation
    left, right = equation.split("=")

    left_expr = sp.sympify(left)
    right_expr = sp.sympify(right)

    expression = left_expr - right_expr

    steps = []

    # Step 1
    steps.append(
        f"Step 1: {sp.latex(left_expr)} = {sp.latex(right_expr)}"
    )

    # Move everything to one side
    steps.append(
        f"Step 2: {sp.latex(sp.expand(expression))} = 0"
    )

    # Factor
    factored = sp.factor(expression)

    if factored != expression:
        steps.append(
            f"Step 3: Factorize → {sp.latex(factored)} = 0"
        )

    # Solve
    solutions = sp.solve(expression, x)

    if len(solutions) == 2 and sp.degree(expression, x) == 2:

        steps.append(
            f"Step 4: Set each factor equal to zero"
        )

        factors = sp.factor_list(expression)[1]

        for factor, power in factors:
            if power == 1:
                factor_solution = sp.solve(factor, x)

                if factor_solution:
                    steps.append(
                        f"        {sp.latex(factor)} = 0  →  "
                        f"x = {sp.latex(factor_solution[0])}"
                    )

    else:
        steps.append(
            f"Step 4: Solve → {', '.join([sp.latex(s) for s in solutions])}"
        )

    # Final answer
    final_answer = ", ".join(
        [f"x = {sp.latex(solution)}" for solution in solutions]
    )

    steps.append(f"Final Answer: {final_answer}")

    return steps


if __name__ == "__main__":

    equation = input("Enter equation: ")

    steps = solve_equation_steps(equation)

    print("\n==============================")
    print("       SOLUTION STEPS")
    print("==============================")

    for step in steps:
        print(step)

    print("==============================")
