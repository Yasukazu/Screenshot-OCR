from __future__ import annotations

import re
import sys
from collections import deque
from dataclasses import dataclass
from datetime import date as Date
from itertools import groupby
from pathlib import Path
from re import Match, Pattern
from sys import path as sys_path
from typing import NamedTuple, Optional, Sequence

from returns.result import safe
# from returns.pipeline import is_successful

# Add parent directory to path
cwd = Path(__file__).resolve().parent
sys_path.insert(0, str(cwd.parent))

from image_filter import (
	BreaktimeAreaParam, FigurePart, FromBottomLabelRange, HeadingAreaParam,
	ImageAreaParam, ImageAreaParamName, ImageDictKey, Int4, PaystubAreaParam,
	ShiftAreaParam, XYOffset, XYRange
)
from ocr_filter import OCRFilter, DatePatterns, MonthDay
from set_logger import set_logger
from tool_pyocr import MDateError

logger = set_logger(__name__)


class MercariFilter(OCRFilter):
	M_DATE_PATT = DatePatterns(hours=re.compile(r"(\d\d:\d\d)"),
		month_date=re.compile(r"(1?\d)\s*/\s*([123]?\d)"),
		day_of_week=re.compile(r"\(\s*([日月火水木木金土])\s*\)"))

	@classmethod
	def extract_month_day_and_hours_from_shift_area_text(cls, txt_lines: Sequence[str], year: int = Date.today().year) -> tuple[MonthDay, list[Match]]:
		''' return[0]: MonthDay as (month:int, day:int), return[1]: list[Match] as "hh:mm" '''
		day_of_week = None
		for shift_text in txt_lines:
			if (mt:=cls.M_DATE_PATT.day_of_week.search(shift_text)):
				day_of_week = mt.groups()[0]
				break
		if not day_of_week:
			raise MDateError(f"Could not resolve date! AppType.M txt_lines!:{txt_lines}") 
		if (hours:=cls.M_DATE_PATT.hours.findall(shift_text)):
			m_d = cls.M_DATE_PATT.month_date.match(shift_text)
			if m_d:
				grps = m_d.groups()
				date = MonthDay(int(grps[0]), int(grps[1]))
				return date, hours

		raise MDateError(f"Could not resolve date! AppType.M txt_lines!:{txt_lines}") 

if __name__ == '__main__':
	from sys import argv
	image_path = argv[1]
