	is_app_to_stem_end_set = False
	def get_app_to_stem_end_dict(app_to_stem_end_dict:dict[APP_NAME, set[str]]={}, stem_end_to_app_dict:dict[str, APP_NAME]={}) -> tuple[dict[APP_NAME, set[str]],dict[str, APP_NAME] ]:
		nonlocal is_app_to_ste
		vals = param.as_slice_param()
		try:
			if area_name == 'shift':
				for n, val in enumerate(vals):
					different = (tuple(org_area_dict[area_name][n]) != val)
					if different:
						break
			else:
				different = (tuple(org_area_dict[area_name]) != vals[0])
			if different:
				yn = input(f"Original image area value({org_area_dict[area_name]}) is different from new one ({vals}): overwrite('yes/no'):")
				if yn.lower() not in ('y', 'yes'):
					logger.info("Overwrite skipped for %s", area_name)
					continue
		except KeyError:
			pass
		if different:
			if make_path.exists() and (make_path_size:=make_path.stat().st_size) > 0:
				try:
					yn = input(f"\nThe file path to save the image file area configuration:'{make_path}'\n already exists with a size as {make_path_size} bytes. Overwrite?(Enter 'Yes' or 'Affirmative' if you want to overwrite): ").lower()
				except (EOFError, KeyboardInterrupt): # Ctrl+C or Ctrl+D/ctrl+Z
					yn = ''
				if yn != 'yes' and yn != 'affirmative':
					sys.exit("Exit since the user did not accept overwrite of: %s" % make_path)
			org_area_dict[area_name] = vals if area_name == 'shift' else vals[0] # update
			ocr_filter_table[APP_NAME[args.app].name.lower()] = org_area_dict
			try:
				if config_file:
					config_file.write(org_config)
					logger.info("config file is updated: %s", make_path)
				else:
					# from io import StringIO
					# sio = StringIO()
					toml_file = TOMLFile(make_path)
					toml_file.write(org_config)
					logger.info("config file is created: %s", make_path)
					'''with make_path.open('w') as wf:
						org_config.write(sio)
						sio.seek(0)
						buff = sio.read()
						print(sio.read(), file=wf)'''
			except Exception as e:
				raise ConfigError("Failed to update or create the config file") from e
	# sio.seek(0)

is_param_dict_loaded = False

def select_area_param(
	area_param_name: ImageAreaParamName,
	image: np.ndarray
) -> ImageAreaParam:

	if image is None or image.size == 0:
		logger.error("Image is None or size 0")
		raise ValueError("Image is None or size 0")
	logger.info("Try to get area params [%s] from image: %s", area_param_name, image.shape)
	from mouse_event import get_area, QuitKeyException
	try:
		TL, BR = get_area(area_name.name, image)
	except QuitKeyException:
		logger.warning(
			"Failed to get area from image for %s", area_name.name
		)
		continue
	else:
		return ImageAreaParam(
			TL[1], BR[1] - TL[1], TL[0], BR[0] - TL[0]
		)