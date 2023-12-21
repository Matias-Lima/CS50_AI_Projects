import sys
from crossword import *

class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):

        for variable in self.domains:
            variable_length = variable.length
            invalid_values = {val for val in self.domains[variable] if len(val) != variable_length}

            # Remove all invalid values from the variable domain
            self.domains[variable] -= invalid_values

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revise = False
        overlap = self.crossword.overlaps[x, y]

        if overlap:
            i, j = overlap
            words_to_remove = set()

            for word_x in self.domains[x]:
                if not any(word_x[i] == word_y[j] for word_y in self.domains[y]):
                    words_to_remove.add(word_x)
                    revise = True

            self.domains[x] -= words_to_remove

        return revise
    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            arcs = [(var_1, var_2) for var_1 in self.domains for var_2 in self.domains if var_1 != var_2]

        visited_arcs = set()  # Track visited arcs
        while arcs:
            var_x, var_y = arcs.pop(0)  # Pop the first arc
            if (var_x, var_y) in visited_arcs:
                continue  # Skip if the arc has already been visited
            visited_arcs.add((var_x, var_y))  # Mark the arc as visited
            if self.revise(var_x, var_y):
                if not self.domains[var_x]:
                    return False
                for var_z in self.crossword.neighbors(var_x) - {var_y}:
                    if (var_z, var_x) not in visited_arcs:
                        arcs.append((var_z, var_x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        return all(var in assignment for var in self.domains)

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        for var, word in assignment.items():
            if var.length != len(word) or len(set(assignment.values())) != len(assignment):
                return False
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    i, j = self.crossword.overlaps[var, neighbor]
                    if word[i] != assignment[neighbor][j]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        values = {}
        vizinhos = self.crossword.neighbors(var)

        for word in self.domains[var]:
            if word not in assignment:
                ruled_out = sum(1 for neighbor in vizinhos if word in self.domains[neighbor])
                values[word] = ruled_out

        return sorted(values, key=lambda key: values[key])
    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # Get set of unassigned variables
        unassigned = set(self.domains) - set(assignment)

        # Select a variable with the smallest domain (MRV) and the largest number of neighbors (degree)
        return min(
            unassigned,
            key=lambda var: (len(self.domains[var]), -len(self.crossword.neighbors(var))),
            default=None  # Return None if unassigned is empty
        )

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        stack = [assignment.copy()]  # Inicializa a pilha com o estado inicial da atribuição

        while stack:
            current_assignment = stack[-1]  # Obtém o último estado da atribuição na pilha
            if self.assignment_complete(current_assignment):
                return current_assignment

            unassigned_vars = self.select_unassigned_variable(current_assignment)
            value_assigned = False

            for value in self.order_domain_values(unassigned_vars, current_assignment):
                current_assignment[unassigned_vars] = value
                if self.consistent(current_assignment):
                    stack.append(current_assignment.copy())  # Adiciona o novo estado à pilha
                    value_assigned = True
                    break

            if not value_assigned:
                del stack[-1]  # Remove o último estado da pilha se nenhum valor foi atribuído

        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
