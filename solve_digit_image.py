from sympy import symbols, solve

SCALE = 8
PADDING = (2, 2)
LINE_WIDTH = 2

def calc_digit_image_size(scale: int=SCALE, padding: tuple[int, int]=PADDING, line_width=LINE_WIDTH)-> list[float]:
	stroke_offset = [(pad + line_width // 2 or 1 ) for pad in padding]
	size = scale, scale * 2
	return [sz + 2 * stroke_offset[i] + line_width // 2 or 1 for i, sz in enumerate(size)]

def solve_digit_image_scale(size: int, padding: int, line_width: int)-> float:
	offset = padding + line_width / 2
	scale = size - 2 * offset - line_width / 2
	return scale

if __name__ == '__main__':
	from pprint import pp
	slvd = solve_digit_image_scale(12, 2, 2)
	pp(slvd)
