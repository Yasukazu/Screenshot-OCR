import re
import ast

def convert_logger_calls(source_code):
	tree = ast.parse(source_code)
	for node in ast.walk(tree):
		if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ['info', 'debug', 'warning', 'error', 'critical']:
			# Extract f-string msg arg (first positional)
			if node.args and isinstance(node.args[0], ast.JoinedStr):
				f_parts = [f.value.s for f in node.args[0].values if isinstance(f, ast.FormattedValue) and isinstance(f.value, ast.Constant)]
				# Reconstruct the format string by replacing all parts
				format_parts = []
				for value in node.args[0].values:
					if isinstance(value, ast.Constant):
						format_parts.append(value.value)
					elif isinstance(value, ast.FormattedValue):
						format_parts.append('%s')
				percent_msg = re.sub(r'\{[^}]+\}', '%s', ''.join(format_parts))
				# Replace first arg with literal, move expressions to logger args
				node.args[0] = ast.Constant(value=percent_msg)
				node.args = node.args[1:] + list(ast.Name(id='placeholder') for _ in f_parts)
	return ast.unparse(tree)  # Python 3.9+

if __name__ == "__main__":
	import sys
	source_code = open(sys.argv[1], 'r').read()
	print(convert_logger_calls(source_code))

