import libcst as cst
import libcst.matchers as m

class FStringToPercentTransformer(m.MatcherDecoratableTransformer):
	"""
	Transforms f-strings in logger calls to percent-formatted strings.
	E.g., logger.info(f"User {name} logged in") -> logger.info("User %s logged in", name)
	"""

	@m.leave(m.Call(
		func=m.OneOf(
			m.Attribute(attr=m.Name(m.Regex("debug|info|warning|error|critical"))),
			m.Name(m.Regex("debug|info|warning|error|critical")) # For 'from logging import info' style calls
		),
		args=[m.Arg(value=m.FormattedString())]
	))
	def leave_logger_call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
		f_string = updated_node.args[0].value
		
		# This is a simplified conversion logic:
		# 1. Reconstruct the format string with %s placeholders.
		# 2. Extract the expressions to be placed in subsequent arguments.
		format_string_parts = []
		arguments = []

		for part in f_string.parts:
			if isinstance(part, cst.FormattedStringText):
				format_string_parts.append(part.value)
			elif isinstance(part, cst.FormattedStringExpression):
				# We use %s for simplicity, as specific format specifiers need complex handling
				format_string_parts.append("%s")
				# Add the expression as a new argument in the call
				arguments.append(cst.Arg(value=part.expression))
		
		new_string_value = "".join(format_string_parts)
		
		# Create a new SimpleString node for the new format string
		new_msg_arg = updated_node.args[0].with_changes(
			value=cst.SimpleString(value=f'"{new_string_value}"')
		)

		# Combine the new message argument with the extracted arguments
		new_args = [new_msg_arg] + arguments
		
		return updated_node.with_changes(args=new_args)

if __name__ == "__main__":
	# Example usage:
	source_code = """
	import logging

	name = "Alice"
	count = 5

	logging.info(f"User {name} has {count} items.")
	logger = logging.getLogger(__name__)
	logger.debug(f"A debug message for {name!r}.")
	"""
	from sys import argv
	source_code = open(argv[1], 'r').read() if len(argv) > 1 else source_code
	# Parse the code into a CST
	module = cst.parse_module(source_code)

	# Apply the transformer
	transformed_module = module.visit(FStringToPercentTransformer())

	# Output the transformed code
	print(transformed_module.code)
