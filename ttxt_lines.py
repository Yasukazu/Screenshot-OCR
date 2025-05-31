class TTxtLines:
    TITLE = 1
    def __init__(self, txt_lines: Sequence[LineBox]):
        self.txt_lines = txt_lines

    def title(self, n=0):
        return self.txt_lines[1].content.replace(' ', '')
    def wages(self):
        return MyOcr.t_wages(self.txt_lines)
    def get_date(self, img_pathset: PathSet, my_ocr: MyOcr) -> tuple[int, MonthDay]:

        for n, txt_line in enumerate(self.txt_lines):
            if txt_line.content.replace(' ', '').startswith('業務開始'):
                break
        if n >= len(self.txt_lines) - 1:
            logger.error("No date found in txt_lines for stem: {}", img_pathset.stem)
            raise ValueError(f"No date found in txt_lines for stem: {img_pathset.stem}")
        date_position = self.txt_lines[n + 1].position
        date_position = date_position[0] + date_position[1]

        img_path = img_pathset.parent / (img_pathset.stem + img_pathset.ext)
        date_image = Image.open(str(img_path)).crop(date_position)

        date_image_dir = img_pathset.parent.parent / 'TMP'
        date_image_dir.mkdir(parents=True, exist_ok=True)
        date_image_fullpath = date_image_dir / f'{img_pathset.stem}.date.png'
        date_image.save(date_image_fullpath, format='PNG')
        logger.info("Saved date image: {}", date_image_fullpath)
        result = my_ocr.run_ocr(path_set=date_image_fullpath, lang='jpn', builder_class=pyocr.builders.TextBuilder, layout=7)
        match result:
            case Success(value):
                no_spc_value = value.replace(' ', '')
                mt = re.match(r"(\d+)月(\d+)日", no_spc_value)
                if mt and len(mt.groups()) == 2:
                    month, day = mt.groups()
                    date = MonthDay(int(month), int(day))
                    return 0, date
                raise ValueError(f"No match string of date!")
            case Failure(_):
                logger.error("No date found in txt_lines for stem: {}", img_pathset.stem)
                raise ValueError(f"No date found in txt_lines for stem: {img_pathset.stem}")


class MTxtLines(TTxtLines):

    def title(self, n: int):
        return ':'.join([self.txt_lines[i].content.replace(' ', '') for i in range(n - 3, n - 1)])

    def wages(self):
        return MyOcr.m_wages(self.txt_lines)

    def get_date(self, img_pathset: PathSet, my_ocr: MyOcr) -> tuple[int, MonthDay]:
        n, date, hrs = my_ocr.check_date(app_type=AppType.M, txt_lines=self.txt_lines)
        if date:
            return n, date
        else: # Retry cropping the image..
            box_pos = [*self.txt_lines[n].position]
            box_pos = list(box_pos[0] + box_pos[1])
            box_pos[0] += box_pos[3] - box_pos[1] # remove leading emoji that is about the same width of the box height
            image = Image.open(img_pathset.parent / (img_pathset.stem + img_pathset.ext))
            if not image:
                logger.error("Failed to open image: {}", img_pathset)
                raise ValueError(f"Failed to open image: {img_pathset}")
            box_img = image.crop(tuple(box_pos))
            if not box_img:
                logger.error("Failed to crop box image: {}", box_pos)
                raise ValueError(f"Failed to crop box image: {box_pos}")
            tmp_img_dir = img_pathset.parent.parent / 'TMP'
            tmp_img_dir.mkdir(parents=True, exist_ok=True)
            box_img_fullpath = tmp_img_dir / f'{img_pathset.stem}.box.png'
            box_img.save(box_img_fullpath, format='PNG')
            logger.info("Saved box image: {}", box_img_fullpath)    
            box_result = my_ocr.run_ocr(box_img_fullpath, lang='jpn', builder_class=pyocr.builders.TextBuilder, layout=7, opt_img=box_img)
            match box_result:
                case Success(value):
                    _n, date, hrs = my_ocr.check_date(app_type=AppType.M, txt_lines=[value]) # len(txt_lines) == 1
                    if date is None:
                        logger.error("Failed to get date from box image: {}", box_pos)
                        raise ValueError(f"Failed to get date from box image: {box_pos}")
                    logger.info("Date by run_ocr with TextBuilder and cropped image: {}", date)
                    return n, date
                case Failure(_):
                    logger.error("Failed to run OCR on box image: {}", box_pos)
                    raise ValueError(f"Failed to run OCR on box image: {box_pos}")
