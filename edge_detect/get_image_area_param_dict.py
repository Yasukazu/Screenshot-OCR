	def get_image_area_param_dict(
		image_area_params: SectionProxy = get_image_area_params_section().unwrap(),
		param_dict: dict[ImageAreaParamName, Sequence[int]] = {},
	) -> dict[ImageAreaParamName, Sequence[int]]:
		nonlocal is_param_dict_loaded
		if not is_param_dict_loaded:
			for param_name, v in image_area_params.items():
				try:
					param_enum = ImageAreaParamName(param_name)
					param_dict[param_enum] = [int(p) for p in v.split(",")]
				except (ValueError, TypeError):
					logger.warning(
						"Invalid image area parameter: %s = %s (value type: %s)",
						param_name,
						v,
						type(v).__name__,
					)
			is_param_dict_loaded = True
		return param_dict

	def get_area_param_dict(
		area_param_dict: dict[ImageAreaParamName, ImageAreaParam] = {},
	) -> dict[ImageAreaParamName, ImageAreaParam]:
		nonlocal is_param_dict_loaded
		if (
			not is_param_dict_loaded
			and (image_area_params := get_image_area_params_section()) is not None
		):
			for param_name, v in image_area_params.items():
				try:
					param_enum = ImageAreaParamName(param_name)
					param = [int(p) for p in v.split(",")]
					param_obj = param_enum.to_param_class(*param)
					area_param_dict[param_enum] = param_obj
				except (ValueError, TypeError):
					logger.warning(
						"Invalid image area parameter: %s = %s (value type: %s)",
						param_name,
						v,
						type(v).__name__,
					)
			is_param_dict_loaded = True
		return area_param_dict