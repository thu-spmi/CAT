"""Parse arguments with python style
use 'null' to indicat python None in JSON

NOTE:
    1. You should acknownledge the usage of argparse module of python first.
    2. Shell var '$' symbol can be used in 'default', if the order of arguments are right. e.g.
        in the following example: 
        in the default of --output, '$input' will be replaced by <input>;
        but note that you can't refer to '$output' in input parser, where
        the '$output' is not initialized yet.
    3. To flag the start of parser, following statements are all allowed:
        4.1 <<"PARSER"  4.2 <<'PARSER'  4.3 <<PARSER  4.4 << "PARSER" ...

Usage: in shell script
example.sh:
<<"PARSER"
("input", type=str, help="Input file.")
("-o", "--output", type=str, default="${input}_out",
    help="Output file. Default: <input>_out")
PARSER
eval $(python utils/parseopt.py $0 $*)
"""

import re
import sys
import argparse


class WrappedArgParser(argparse.ArgumentParser):
    # re-locate the help information to error
    def print_help(self, file=None) -> None:
        if file is None:
            return super().print_help(sys.stderr)
        elif file == sys.stdout:
            return
        else:
            return super().print_help(file)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write(
            "This script is used to parse options for shell script.\n"
            f"... read header of {sys.argv[0]} for the usage.\n")
        sys.exit(1)
    script = sys.argv[1]
    argsin = sys.argv[2:]

    # match lines in '<<PARSER' in 'PARSER'
    parser_pattern = re.compile(
        r"^<<\s*(?:\"PARSER\"|'PARSER'|PARSER)\s*$((?:.|\n)*?)^\s*PARSER\s*$",
        re.MULTILINE)
    # split lines via brackets
    argument_pattern = re.compile(r"^[(]((?:.|\n)*?)[)]$", re.MULTILINE)
    with open(script, 'r') as fi:
        s = fi.read()
    parserinfo = parser_pattern.findall(s)
    match = argument_pattern.findall(parserinfo[0])

    parser = WrappedArgParser(prog=script)
    for arg in match:
        # NOTE:
        # This is potential dangerous. It's your duty to ensure the safety.
        eval(f"parser.add_argument({arg})")

    try:
        for arg, value in vars(parser.parse_args(argsin)).items():
            if isinstance(value, list):
                # deal with nargs='+' and nargs='*'
                value = f"\"{' '.join([str(x) for x in value])}\""
            sys.stdout.write(f"export {arg}={value}; ")
    except SystemExit as se:
        sys.stdout.write("exit 1;")
        sys.exit(1)
    else:
        sys.exit(0)
