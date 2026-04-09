def app_name()-> APP_NAME: # n=0, app_name_list=[]
		try:
			return args.app
		except KeyError:
			pass
		if not args.app: # is not None:
			raise NoAppNameError("No application name specified")
		elif is_app_name_set:
			return app_name_list[n]
		else:
			try:
				_app_name = APP_NAME[args.app.upper()] # app_name_to_enum[args.app]
				logger.info("app_name is set as '%s'", _app_name)
			except KeyError:
				raise ConfigError(f"No such app name:{args.app}")
			else:
				try: # try to extract app name from file name
					_file = get_args_files()[args.nth - 1]
					_app_name = None
					for nm in args.file_stem_end: # APP_NAME:
						if Path(_file).stem.endswith(nm.value):
							_app_name = nm
							if _app_name not in app_name_list:
								app_name_list.append(_app_name)
					if not app_name_list:
						raise NoAppNameError("No app name is found in file names! Available names: {}".format([nm.value for nm in APP_NAME]))	
					logger.info("Application name(s) is/are set as [%s] from file name:%s", [an.name.lower() for an in app_name_list], _file.name)
				except (IndexError, NoAppNameError):
					sys.exit("Needs application name spec. by '--app' option or file name(ending with {}).".format([nm.value for nm in APP_NAME]))
		try:
			image_path_dir = Path(args.image_dir).expanduser() # or get_filter_config()['image-path']['dir'])
		except TypeError:
			logger.info("Image dir keeps None since not args.dir or not get_filter_config")
		except RuntimeError:
			logger.info("Image dir user home expansion(starting with '~') failed.")

