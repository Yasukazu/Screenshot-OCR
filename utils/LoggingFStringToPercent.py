import libcst as cst
from libcst import codemod
from libcst.matchers import matches, Call, SimpleString, Name

class LoggingFStringToPercent(codemod.Codemod):
    DESCRIPTION: str = "Convert f-string logger calls to %-formatting."

    def transform_module_impl(self, tree: cst.Module) -> cst.Module:
        return tree.visit(LoggingFStringTransformer(codemod.Context(self.context.wrapper)))

class LoggingFStringTransformer(codemod.MatchTransformer):
    def leave_Call(
        self,
        original_node: Call,
        updated_node: Call,
    ) -> Call | None:
        matcher = matches(
            Call(
                func=Name(nm="info") | Name(nm="error") | ... ,  # Add methods
                args=[
                    SimpleString(value=cst.FStringAtom(...)),  # Detect f-strings
                ],
            )
        )
        if matches(original_node, matcher):
            # Extract f-exprs, build "%s" msg + args tuple
            fstring = original_node.args[0].value  # Simplified
            percent_fmt = self._fstring_to_percent(fstring)
            # Return Call(func=..., args=[percent_fmt, Arg(value=Tuple(...))])
            pass  # Implement extraction/logic
        return updated_node

    def _fstring_to_percent(self, fnode: cst.FString) -> cst.SimpleString:
        # Parse parts/atoms to "%s" * len(atoms), collect exprs as args
        placeholders = "%s" * len(fnode.parts)
        return cst.SimpleString(value=placeholders)