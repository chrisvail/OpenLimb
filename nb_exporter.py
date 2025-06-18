import json
from abc import ABC


def export_notebook(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    out_file_path = file_path[:-5] + "py"

    directives = [ExportAll(), ExportObject(), ExportSection()]

    with open(out_file_path, "w") as f:
        for cell in filter(lambda x: x["cell_type"] == "code", data["cells"]):
            iterator = iter(cell["source"])

            for i, line in enumerate(iterator):
                test_line = line
                while test_line is not None:
                    for directive in directives:
                        if directive.match_line(test_line):
                            test_line = directive.process_lines(
                                test_line, iterator, f)
                            f.write("\n\n")
                            break
                    else:
                        test_line = None


class ExportDirective(ABC):

    def match_line(self, line):
        raise NotImplementedError()

    def process_lines(self, line, iterator, file):
        raise NotImplementedError


class ExportAll(ExportDirective):
    def match_line(self, line):
        return line.strip() == "#| export_all"

    def process_lines(self, line, iterator, file):
        for line in iterator:
            file.write(line)

        return None


class ExportObject(ExportDirective):
    def match_line(self, line):
        return line.strip() == "#| export"

    def process_lines(self, line, iterator, file):
        line = next(iterator)
        file.write(line)
        indentation = len(line) - len(line.lstrip(' '))
        line = next(iterator)
        new_indent = len(line) - len(line.lstrip(' '))
        while new_indent > indentation or line == "\n":
            file.write(line)
            try:
                line = next(iterator)
            except StopIteration:
                return None
            new_indent = len(line) - len(line.lstrip(' '))

        return line


class ExportSection(ExportDirective):
    def match_line(self, line):
        return line.strip() == "#| export_section"

    def process_lines(self, line, iterator, file):
        line = next(iterator)
        while not line.startswith("#| end_section"):
            file.write(line)
            line = next(iterator)

        return line


if __name__ == "__main__":
    export_notebook("discrete_operators.ipynb")
