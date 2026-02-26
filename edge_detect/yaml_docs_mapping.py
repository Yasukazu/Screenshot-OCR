from dataclasses import fields, dataclass, field
from ruamel.yaml.comments import CommentedMap
from image_filter_main_settings import APP_NAME
import okome
from fancy_dataclass import version
# from dataclass_binder import get_field_docstrings

import ast
from inspect import getsource, cleandoc
from textwrap import dedent# , cleandoc
from typing import Any, Mapping

from image_filter_main_settings import AppSettings


def get_field_docstrings(dataklass: type[Any]) -> Mapping[str, str]:
	"""
	Return a mapping of field name to the docstring for that field.

	Attribute docstrings are not supported by the Python runtime, therefore we must read them from the source code.
	If the source code cannot be found, an empty mapping is returned.
	"""

	try:
		breakpoint()
		source = getsource(dataklass)
	except (OSError, TypeError):
		# According to the documentation only OSError can be raised, but Python 3.10 raises TypeError for
		# sourceless dataclasses.
		#   https://github.com/python/cpython/issues/98239
		raise # return {}

	module_def = ast.parse(dedent(source), "<string>")
	class_def = module_def.body[0]
	assert isinstance(class_def, ast.ClassDef)

	docstrings = {}
	scope = None
	for node in class_def.body:
		match node:
			case ast.AnnAssign(target=ast.Name(id=name)):
				scope = name
			case ast.Expr(value=ast.Constant(value=str(docstring))):
				if scope is None:
					# When using 'scope is not None', Coverage 7.12.0 will consider the 'is None' branch uncovered.
					pass
				else:
					docstrings[scope] = cleandoc(docstring)
	return docstrings


app_settings_field_docstring_map = get_field_docstrings(AppSettings)

'''def field_docs_mapping(cls):
	c = okome.parse(cls)
	return {
		f.name: " ".join(f.comment).strip()
		for f in c.fields
	}'''

# FIELD_DOCS = AppSettings.field_docs_mapping()  # auto from okome

def dataclass_to_commented_map(obj):
	cm = CommentedMap()
	for f in fields(obj):
		cm[f.name] = getattr(obj, f.name)
		doc = FIELD_DOCS.get(f.name)
		if doc:
			cm.yaml_set_comment_before_after_key(f.name, after=doc)
	return cm


# docs = field_docs_mapping(AppSettings)
print(app_settings_field_docstring_map)
# {'host': 'Hostname or IP of the server', 
#  'port': 'Port number to bind (default 8080)'}
